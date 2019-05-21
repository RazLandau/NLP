from PCFG import PCFG
import math

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents

def cnf_cky(pcfg, sent):
    ### YOUR CODE HERE
    sent = sent.split()
    n_words = len(sent)
    pi = dict()
    bp = dict()
    # Init
    for i in range(1, n_words + 1):
        for lhs in pcfg._rules: # Why is the rules field private?
            for rule in pcfg._rules[lhs]:
                if pcfg.is_terminal(rule[0][0]) and rule[0][0] == sent[i-1]:
                    pi[(i, i, lhs)] = (rule[1]/pcfg._sums[lhs])
                    bp[(i, i, lhs)] = (rule, 0)
                    break
                else:
                    pi[(i, i, lhs)] = 0.0

    # Algorithm
    for l in range(1, n_words):
        for i in range(1, n_words - l + 1):
            j = i + l
            for lhs in pcfg._rules:  # Why is the rules field private?
                for rule in pcfg._rules[lhs]:
                    if not pcfg.is_terminal(rule[0][0]):
                        pi_val_max = 0
                        for s in range(i, j):
                            pi_val = (rule[1] / pcfg._sums[lhs]) * pi[(i, s, rule[0][0])] * pi[(s+1, j, rule[0][1])]
                            if pi_val > pi_val_max:
                                pi[(i, j, lhs)] = pi_val
                                bp[(i, j, lhs)] = (rule, s)
                                pi_val_max = pi_val
                            if (i, j, lhs) not in pi:
                                pi[(i, j, lhs)] = 0.0
                    else:
                        if (i, j, lhs) not in pi:
                            pi[(i, j, lhs)] = 0.0

    if (1,n_words, 'ROOT') not in bp:
        return "FAILED TO PARSE!"
    else:
        return print_tree(bp, (1,n_words), 'ROOT')
    ### END YOUR CODE


def non_cnf_cky(pcfg, sent):
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return "FAILED TO PARSE!"


def print_tree(cky_dict, subtreerange, tree_root):
    range_high_right = subtreerange[1]
    range_low_right = cky_dict[(subtreerange[0], subtreerange[1], tree_root)][1] + 1
    range_high_left = cky_dict[(subtreerange[0], subtreerange[1], tree_root)][1]
    range_low_left = subtreerange[0]
    if range_low_left == range_high_right:
        return tree_root, cky_dict[(subtreerange[0], subtreerange[1], tree_root)][0][0][0]
    left_root = cky_dict[(subtreerange[0], subtreerange[1], tree_root)][0][0][0]
    right_root = cky_dict[(subtreerange[0], subtreerange[1], tree_root)][0][0][1]
    return "(" + tree_root + " " + str(print_tree(cky_dict, (range_low_left, range_high_left), left_root)) + " " \
          + str(print_tree(cky_dict, (range_low_right, range_high_right), right_root))  + ")"


if __name__ == '__main__':
    import sys
    cnf_pcfg = PCFG.from_file_assert("cnf_grammar.txt", assert_cnf=True)
    # non_cnf_pcfg = PCFG.from_file_assert(sys.argv[2])
    sents_to_parse = load_sents_to_parse("sents.txt")
    for sent in sents_to_parse:
        print cnf_cky(cnf_pcfg, sent)
        # print non_cnf_cky(non_cnf_pcfg, sent)
    # print cnf_cky(PCFG.from_file_assert("cnf_grammar.txt"), "a fine sandwich understood the president in a chief on every pickled perplexed delicious fine president")
