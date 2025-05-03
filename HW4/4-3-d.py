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

log_likelihood_unigram = sum(np.log(unigram_probs.get(token, unigram_probs["<UNK>"])) for token in sentence_tokens)

prev_index = vocab.index("<s>") + 1
bigram_log_likelihood = 0.0

unseen_bigrams = []
for idx in token_indices:
    match = bigram_data[(bigram_data["w1_idx"] == prev_index) & (bigram_data["w2_idx"] == idx)]
    count_bigram = int(match["count"].values[0]) if not match.empty else 0
    prev_count = unigram_counts[prev_index - 1]

    if count_bigram > 0:
        prob = count_bigram / prev_count
    else:
        prob = 0.0
        unseen_bigrams.append((vocab[prev_index - 1], vocab[idx - 1]))

    bigram_log_likelihood += np.log(prob)
    prev_index = idx

print("Unigram log-likelihood:", log_likelihood_unigram)
print("Bigram log-likelihood:", bigram_log_likelihood)
print("Unseen bigrams:", unseen_bigrams)
