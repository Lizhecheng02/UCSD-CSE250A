import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_vector(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def read_observations(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=int)


def viterbi(pi: np.ndarray, A: np.ndarray, B: np.ndarray, O: np.ndarray) -> np.ndarray:
    N = pi.shape[0]
    T = O.shape[0]

    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    delta = log_pi + log_B[:, O[0]]
    psi = np.empty((T, N), dtype=np.int16)

    for t in range(1, T):
        scores = delta[:, None] + log_A
        psi[t] = scores.argmax(axis=0)
        delta = scores[psi[t], np.arange(N)] + log_B[:, O[t]]

    path = np.empty(T, dtype=np.int16)
    path[-1] = delta.argmax()

    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]

    return path


def collapse_and_decode(states: np.ndarray) -> str:
    alphabet = [chr(ord("a") + i) for i in range(26)] + [" "]
    msg = []
    prev = -1
    for s in states:
        if s != prev:
            msg.append(alphabet[s])
            prev = s
    return "".join(msg)


def plot_states(states: np.ndarray, out_path: Path):
    plt.figure(figsize=(16, 7))
    plt.step(range(len(states)), states + 1, where="post")
    tick_labels = [chr(ord("a") + i) for i in range(26)] + [" "]
    plt.yticks(ticks=np.arange(1, 28), labels=tick_labels)
    plt.xlabel("time step t")
    plt.ylabel("state letter")
    plt.title("Most probable hidden-state sequence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Viterbi decoding for a 27-state HMM.")
    p.add_argument("--pi", default="initialStateDistribution.txt", help="initial-state distribution file")
    p.add_argument("--A", default="transitionMatrix.txt", help="state-transition matrix file")
    p.add_argument("--B", default="emissionMatrix.txt", help="emission matrix file")
    p.add_argument("--obs", default="observations.txt", help="observations file (0/1)")
    args = p.parse_args()

    pi = read_vector(args.pi)
    A = read_matrix(args.A)
    B = read_matrix(args.B)
    O = read_observations(args.obs)

    if pi.size != 27 or A.shape != (27, 27) or B.shape != (27, 2):
        raise ValueError("Model dimensions do not match n=27, k=2.")

    states = viterbi(pi, A, B, O)
    Path("states.txt").write_text("\n".join(map(str, states.tolist())))
    quote = collapse_and_decode(states)
    print("\nDecoded quotation:\n" + quote + "\n")

    plot_states(states, Path("viterbi_states.png"))
    print("State path written to states.txt and plot saved to viterbi_states.png")


if __name__ == "__main__":
    main()
