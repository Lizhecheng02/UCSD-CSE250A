import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

vocab_file = "./hw4_vocab.txt"
unigram_file = "./hw4_unigram.txt"
bigram_file = "./hw4_bigram.txt"

with open(vocab_file, "r") as vf:
    vocab = [line.strip() for line in vf.readlines()]

with open(unigram_file, "r") as uf:
    unigram_counts = [int(line.strip()) for line in uf.readlines()]

total_count = sum(unigram_counts)
unigram_probs = {word: count / total_count for word, count in zip(vocab, unigram_counts)}

bigram_data = pd.read_csv(bigram_file, sep="\t", header=None, names=["w1_idx", "w2_idx", "count"])

sentence_tokens = ["THE", "SIXTEEN", "OFFICIALS", "SOLD", "FIRE", "INSURANCE"]

token_indices = [vocab.index(token) + 1 if token in vocab else vocab.index("<UNK>") + 1 for token in sentence_tokens]

lambdas = np.linspace(0, 1, 100)
log_likelihoods_mixture = []

token_indices_mixture = token_indices

for lam in lambdas:
    prev_index = vocab.index("<s>") + 1
    log_likelihood_m = 0.0

    for idx in token_indices_mixture:
        word = vocab[idx - 1]
        p_u = unigram_probs.get(word, unigram_probs["<UNK>"])

        match = bigram_data[(bigram_data["w1_idx"] == prev_index) & (bigram_data["w2_idx"] == idx)]
        count_bigram = int(match["count"].values[0]) if not match.empty else 0
        prev_count = unigram_counts[prev_index - 1]
        p_b = count_bigram / prev_count if count_bigram > 0 else 1e-10

        p_m = lam * p_u + (1 - lam) * p_b
        log_likelihood_m += np.log(p_m)
        prev_index = idx

    log_likelihoods_mixture.append(log_likelihood_m)

plt.figure(figsize=(8, 5))
plt.plot(lambdas, log_likelihoods_mixture, label="Log-Likelihood L_m", color="red", linewidth=1.5)
plt.xlabel("λ (lambda)")
plt.ylabel("Log-Likelihood L_m")
plt.title("Log-Likelihood of Mixture Model vs. Lambda")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("log_likelihood_mixture.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()

optimal_lambda_index = np.argmax(log_likelihoods_mixture)
optimal_lambda = lambdas[optimal_lambda_index]
optimal_lambda, log_likelihoods_mixture[optimal_lambda_index]

print("Optimal λ (lambda):", optimal_lambda)
print("Log-Likelihood L_m at optimal λ:", log_likelihoods_mixture[optimal_lambda_index])
