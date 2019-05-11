from data import *
import time


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    # YOUR CODE HERE

    def increment(d, k):
        d[k] = 1 if k not in d else d[k] + 1
    for sent in sents:
        increment(q_uni_counts, 'START_TAG')
        for i in range(len(sent)):
            total_tokens += 1
            word, tag = sent[i]
            tag_1 = sent[i-1][1] if i > 0 else 'START_TAG'
            tag_2 = sent[i-2][1] if i > 1 else 'START_TAG'
            increment(e_word_tag_counts, (word, tag))
            increment(e_tag_counts, tag)
            increment(q_uni_counts, tag)
            increment(q_bi_counts, (tag, tag_1))
            increment(q_tri_counts, (tag_2, tag_1, tag))
        increment(q_uni_counts, 'STOP_TAG')
        increment(q_bi_counts, (tag, 'STOP_TAG'))
        increment(q_tri_counts, (tag_1, tag, 'STOP_TAG'))
    # END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    # YOUR CODE HERE

    def q(t_2, t_1, t):
        q_val = 0.0
        if (t_2, t_1, t) in q_tri_counts and (t_2, t_1) in q_bi_counts:
            q_val += (lambda1 * q_tri_counts[(t_2, t_1, t)]) / q_bi_counts[(t_2, t_1)]
        if (t_1, t) in q_bi_counts and t_1 in q_uni_counts:
            q_val += (lambda2 * q_bi_counts[(t_1, t)]) / q_uni_counts[t_1]
        q_val += ((1-lambda1-lambda2) * q_uni_counts[t]) / total_tokens
        return q_val

    def e(w, t):
        if (w, t) in e_word_tag_counts and t in e_tag_counts:
            return float(e_word_tag_counts[(w, t)]) / e_tag_counts[t]
        return 0

    def S(k):
        return ['START_TAG'] if k < 0 else list(q_uni_counts.keys())

    pi = {(-1, 'START_TAG', 'START_TAG'): 1}
    bp = dict()
    pruning_const = 0.0
    n = len(sent)
    for k in range(n):
        for u in S(k-1):
            for v in S(k):
                e_xk_v = e(sent[k], v)
                if e_xk_v <= pruning_const:
                    continue
                w_val_max = -1
                for w in S(k-2):
                    if (k-1, w, u) not in pi:
                        continue
                    w_val = pi[(k-1, w, u)] * q(w, u, v) * e_xk_v
                    if w_val > w_val_max:
                        bp[(k, u, v)] = w
                        pi[(k, u, v)] = w_val
                        w_val_max = w_val

    y_n, y_n_1, u_v_val_max = None, None, -1
    for v in S(n):
        for u in S(n-1):
            if (n-1, u, v) in pi:
                u_v_val = pi[(n-1, u, v)] * q(u, v, 'STOP_TAG')
                if u_v_val > u_v_val_max:
                    y_n = v
                    y_n_1 = u
                    u_v_val_max = u_v_val
    predicted_tags[n-1], predicted_tags[n-2] = y_n, y_n_1
    for k in range(n-3, -1, -1):
        predicted_tags[k] = bp[(k+2), predicted_tags[k+1], predicted_tags[k+2]]
    # END YOUR CODE
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    # YOUR CODE HERE

    def lambda_grid_search(should_run=False):
        if not should_run:
            return 0.2, 0.6
        for l1 in range(0, 11, 2):
            for l2 in range(0, 11, 2):
                lambda1, lambda2 = l1 / 10.0, l2 / 10.0
                if lambda1 + lambda2 > 1.0:
                    continue
                acc_viterbi = 0.0
                total = 0
                for idx, words_and_tags in enumerate(test_data):
                    sent = [wt[0] for wt in words_and_tags]
                    tags = [wt[1] for wt in words_and_tags]
                    pred = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                                       e_tag_counts, lambda1, lambda2)
                    for i in range(len(tags)):
                        total += 1
                        acc_viterbi += pred[i] == tags[i]
                acc_viterbi /= len(test_data)
                print('(%.1f,%.1f,%.1f),%.4f' % (lambda1, lambda2, 1-lambda1-lambda2, acc_viterbi))
    lambda1, lambda2 = lambda_grid_search()

    total = 0
    for idx, words_and_tags in enumerate(test_data):
        # TODO fine tune the lambdas values
        sent = [wt[0] for wt in words_and_tags]
        tags = [wt[1] for wt in words_and_tags]
        pred = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts,
                           lambda1, lambda2
        )
        for i in range(len(tags)):
            total += 1
            acc_viterbi += pred[i] == tags[i]
    acc_viterbi /= total
    # END YOUR CODE

    return acc_viterbi


if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "HMM DEV accuracy: " + str(acc_viterbi)

    train_dev_end_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds"

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "HMM TEST accuracy: " + str(acc_viterbi)
        full_flow_end_time = time.time()
        print "Full flow elapsed: " + str(full_flow_end_time - start_time) + " seconds"