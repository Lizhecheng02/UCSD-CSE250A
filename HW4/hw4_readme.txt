The language modeling data files for homework 4 are:

  hw4_vocab.txt
  hw4_unigram.txt
  hw4_bigram.txt
  
The file hw4_vocab.txt contains a list of 500 tokens, corresponding to words, punctuation symbols, and other textual markers.

The file hw4_unigram.txt contains the counts of each of these tokens in a large text corpus of Wall Street Journal articles.  The corpus consisted of roughly 3 million sentences.

The file hw4_bigram.txt contains the counts of pairs of adjacent words in this same corpus.  Let count(w1,w2) denote the number of times that word w1 is followed by word w2.  The counts are stored in a simple three column format: 

  index(w1)  index(w2)  count(w1,w2)


