#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0

    # YOUR CODE HERE

    def increment(d, k):
        d[k] = 1 if k not in d else d[k] + 1
    for sent in dataset:
        for i in range(len(sent)):
            increment(unigram_counts, sent[i])
            i > 0 and increment(bigram_counts, (sent[i], sent[i-1]))
            i > 1 and increment(trigram_counts, (sent[i], sent[i-1], sent[i-2]))
            token_count += 1
    # END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    # YOUR CODE HERE
    for sent in eval_dataset:
        qlis = 1
        for i in range(2, len(sent)):
            qli = 0
            try:
                qli += lambda1 * trigram_counts[(sent[i], sent[i-1], sent[i-2])] / bigram_counts[(sent[i-1], sent[i-2])]
            except KeyError: pass
            try:
                qli += lambda2 * bigram_counts[(sent[i], sent[i-1])] / unigram_counts[sent[i-1]]
            except KeyError: pass
            qli += (1-lambda1-lambda2) * unigram_counts[sent[i]] / train_token_count
            qlis *= qli
        l = np.log2(qlis) / len(sent)
        perplexity += 2**-l
    perplexity /= len(eval_dataset)
    # END YOUR CODE
    return perplexity


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    # Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    # YOUR CODE HERE
    d = {}
    for l1 in np.arange(0, 1.1, 0.1):
        for l2 in np.arange(0, 1.1, 0.1):
            if l1+l2 < 1:
                d[(l1, l2)] = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, l1,
                                              l2)
    # END YOUR CODE
    print(d)
    perplexity_max, perplexity_min = max(d, key=d.get), min(d, key=d.get)
    print('max:', d[perplexity_max], 'at:', perplexity_max, 'min:', d[perplexity_min], 'at:', perplexity_min)


if __name__ == "__main__":
    test_ngram()
