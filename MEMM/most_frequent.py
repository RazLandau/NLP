from data import *
import numpy as np


def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    # YOUR CODE HERE
    from collections import Counter, defaultdict
    word_to_counter = defaultdict(Counter)
    for sent in train_data:
        for word, tag in sent:
            word_to_counter[word][tag] += 1
    return dict([
        (k, word_to_counter[k].most_common(1)[0][0]) for k in word_to_counter
    ])
    # END YOUR CODE


def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    # YOUR CODE HERE
    correct = 0.0
    total = 0.0
    for sent in test_set:
        for word, tag in sent:
            total += 1
            correct += tag == pred_tags[word]
    return correct / total
    # END YOUR CODE


if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + str(most_frequent_eval(dev_sents, model))

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + str(most_frequent_eval(test_sents, model))