import pandas as pd
import warnings
warnings.filterwarnings("ignore")

vocab_file = "./hw4_vocab.txt"
bigram_file = "./hw4_bigram.txt"

with open(vocab_file, "r") as vf:
    vocab = [line.strip() for line in vf.readlines()]

bigram_data = pd.read_csv(bigram_file, sep="\t", header=None, names=["w1_idx", "w2_idx", "count"])

the_index = vocab.index("THE") + 1
the_bigrams = bigram_data[bigram_data["w1_idx"] == the_index]

total_the_count = the_bigrams["count"].sum()

the_bigrams["probability"] = the_bigrams["count"] / total_the_count
the_bigrams["w2_token"] = the_bigrams["w2_idx"].apply(lambda idx: vocab[idx - 1])

top_10_the_bigrams = the_bigrams.sort_values(by="probability", ascending=False).head(10)[["w2_token", "probability"]]

print(top_10_the_bigrams)
