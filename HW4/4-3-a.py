import pandas as pd
import warnings
warnings.filterwarnings("ignore")

vocab_file = "./hw4_vocab.txt"
unigram_file = "./hw4_unigram.txt"

with open(vocab_file, "r") as vf:
    vocab = [line.strip() for line in vf.readlines()]

with open(unigram_file, "r") as uf:
    unigram_counts = [int(line.strip()) for line in uf.readlines()]

total_count = sum(unigram_counts)

unigram_probs = {word: count / total_count for word, count in zip(vocab, unigram_counts)}
m_words_probs = {word: prob for word, prob in unigram_probs.items() if word.startswith("M")}


df_m_words_probs = pd.DataFrame(m_words_probs.items(), columns=["Token", "Unigram Probability"])
df_m_words_probs_sorted = df_m_words_probs.sort_values(by="Unigram Probability", ascending=False).reset_index(drop=True)

print(df_m_words_probs_sorted.head(30))
