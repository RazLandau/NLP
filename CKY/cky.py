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
    # YOUR CODE HERE
    sent = sent.split()
    n = len(sent)
    pi, bp = {}, {}

    def q(X, prob):
        return prob / pcfg._sums[X]

    # Init
    for i in range(1, n + 1):
        for X in pcfg._rules:  # Why is the rules field private?
            for (rhs, prob) in pcfg._rules[X]:
                if rhs[0] == sent[i - 1]:
                    pi[(i, i, X)] = q(X, prob)
                    bp[(i, i, X)] = ((rhs, ), 0)
                    break
                else:
                    pi[(i, i, X)] = 0.0

    # Algorithm
    for l in range(1, n):
        for i in range(1, n - l + 1):
            j = i + l
            for X in pcfg._rules:  # Why is the rules field private?
                for (rhs, prob) in pcfg._rules[X]:
                    if (i, j, X) not in pi:
                        pi[(i, j, X)] = 0.0
                    if pcfg.is_preterminal(rhs):
                        continue
                    Y, Z = rhs[0], rhs[1]
                    s_val_max = 0.0
                    for s in range(i, j):
                        s_val = q(X, prob) * pi[(i, s, Y)] * pi[(s + 1, j, Z)]
                        if s_val > s_val_max:
                            pi[(i, j, X)] = s_val
                            bp[(i, j, X)] = ((rhs, ), s)
                            s_val_max = s_val

    if (1, n, 'ROOT') in bp:
        return print_tree(bp, (1, n), 'ROOT')
    # END YOUR CODE
    return "FAILED TO PARSE!"


def non_cnf_cky(pcfg, sent):
    # YOUR CODE HERE

    def unarize_terminals(lhs, i):
        rhs = pcfg._rules[lhs][i][0]
        if len(rhs) == 1:
            return
        lhs_1 = rhs[0].upper()
        rhs_1 = [rhs[0]]
        lhs_2 = '_'.join(rhs[1:]).upper()
        rhs_2 = rhs[1:]
        pcfg.add_rule(lhs_1, rhs_1, 1)
        pcfg.add_rule(lhs_2, rhs_2, 1)
        pcfg._rules[lhs][i] = ([lhs_1, lhs_2], 1)
        # print(pcfg._rules)
        unarize_terminals(lhs_2, 0)

    for lhs in pcfg._rules.keys():
        for i, (rhs, prob) in enumerate(pcfg._rules[lhs]):
            if pcfg.is_terminal(rhs[0]):
                unarize_terminals(lhs, i)
    res = cnf_cky(pcfg, sent)
    if res != "FAILED TO PARSE!":
        return res
    # END YOUR CODE
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
    return "(" + tree_root + " " + str(print_tree(cky_dict, (range_low_left, range_high_left), left_root)) + " " + \
           str(print_tree(cky_dict, (range_low_right, range_high_right), right_root)) + ")"


if __name__ == '__main__':
    # import sys
    # cnf_pcfg = PCFG.from_file_assert("grammar2-CNF.txt", assert_cnf=True)
    # non_cnf_pcfg = PCFG.from_file_assert("grammar2.txt")
    # sents_to_parse = load_sents_to_parse("sents.txt")
    # for sent in sents_to_parse:
    #     print cnf_cky(cnf_pcfg, sent)
    #     print non_cnf_cky(non_cnf_pcfg, sent)

    print cnf_cky(
        PCFG.from_file_assert("cnf_grammar.txt"),
            "a fine sandwich understood the president in a chief on every pickled perplexed delicious fine president"
    )
    print non_cnf_cky(
        PCFG.from_file_assert("non_cnf_grammar.txt"),
        "a fine sandwich"
    )
