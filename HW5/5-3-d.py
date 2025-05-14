import numpy as np


def load_spect_data(x_path="spectX.txt", y_path="spectY.txt"):
    X = np.loadtxt(x_path, dtype=int)
    y = np.loadtxt(y_path, dtype=int)
    T, n = X.shape
    assert y.shape[0] == T, "Mismatch between X and y lengths"
    return X, y


def em_noisy_or(X, y, num_iters=256, init_p=0.05, record_iters=None):
    T, n = X.shape
    p = np.full(n, init_p, dtype=float)
    T_i = X.sum(axis=0)
    if record_iters is None:
        record_iters = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    results = {}
    for it in range(num_iters + 1):
        s = np.prod((1 - p) ** X, axis=1)
        P1 = 1 - s

        if it in record_iters:
            eps = 1e-12
            ll = np.mean(y * np.log(P1 + eps) + (1 - y) * np.log(1 - P1 + eps))
            preds = (P1 >= 0.5).astype(int)
            mistakes = int(np.sum(((y == 1) & (preds == 0)) | ((y == 0) & (preds == 1))))
            results[it] = {"mistakes": mistakes, "log_likelihood": ll}
        if it == num_iters:
            break

        numerator = (y[:, None] * X) * p[None, :]
        denom = (1 - s)[:, None]
        denom_safe = np.where(denom == 0, 1.0, denom)
        w = numerator / denom_safe

        p = np.where(T_i > 0, w.sum(axis=0) / T_i, p)

    return results, p


if __name__ == "__main__":
    X, y = load_spect_data()
    results, p_est = em_noisy_or(X, y)

    print(f"{'Iteration':>10} | {'Mistakes':>8} | {'Log-likelihood':>15}")
    print("-" * 40)
    for it in sorted(results):
        m = results[it]["mistakes"]
        ll = results[it]["log_likelihood"]
        print(f"{it:>10} | {m:>8} | {ll:>15.5f}")
