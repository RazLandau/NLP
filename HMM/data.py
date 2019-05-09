import os
MIN_FREQ = 3


def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res


def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents


def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab


def replace_word(word):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    # YOUR CODE HERE
    import re
    # based on Biler et al. Handles mostly numerical quantities and indicative capitalization
    matches = [
        ('^\d{2}$', 'twoDigitNum'),
        ('^\d{4}$', 'fourDigitNum'),
        ('^[0-9-]+$', 'containsDigitAndDash'),
        ('^[0-9,]+$', 'containsDigitAndComma'),
        ('^[0-9.]+$', 'containsDigitAndPeriod'),
        ('^[0-9,.]+$', 'containsDigitAndCommaAndPeriod'),
        ('^[0-9]+$', 'Num'),
        ('^[A-Z]+$', 'AllCaps'),
        ('^[A-Z].$', 'capPeriod'),
    ]
    for pattern, cls in matches:
        if re.match(pattern, word):
            return cls

    # based on http://www.uefap.com/vocab/build/building.html handles predix-suffix
    verb_prefix = 're dis over un mis out be co de fore inter pre sub trans over'
    verb_suffix = 'ise ate fy en'
    noun_prefix = 'anti auto bi co counter dis ex hyper in in inter kilo mal mega mis mini mono neo out poly pseudo ' \
                  're semi sub super sur tele tri ultra under vice'
    noun_suffix = 'tion sion er ment ant ent age al ence ance ery ry er ism ship age ity ness cy'
    adj_suffix = 'al ent ive ous ful less able'
    adj_prefix = 'un im in ir il non dis'
    verb_time_suffix = 'ed ing'
    extra_suffix = 'zation ation'
    extra_prefix = 'pre duo bi'
    prefixes = ' '.join([verb_prefix, noun_prefix, adj_prefix, extra_prefix])
    suffixes = ' '.join([verb_suffix, noun_suffix, adj_suffix, verb_time_suffix, extra_suffix])

    def in_suffixes(w, fixes):
        for s in fixes.split(' '):
            if w.endswith(s):
                return s+'Suffix'
        return None

    def in_prefixes(w, fixes):
        for s in fixes.split(' '):
            if w.startswith(s):
                return s+'Prefix'
        return None

    in_prefix = in_prefixes(word, prefixes)
    if in_prefix:
        return in_prefix
    in_suffix = in_suffixes(word, suffixes)
    if in_suffix:
        return in_suffix
    
    # handles capitalization and simple strings. should catch most words, leaving unk to odd ones
    late_matches = [
        ('^[A-Z]{1}[a-z]*$', 'initCap'),
        ('^[a-z]+$', 'lowerCase')
    ]
    for pattern, cls in late_matches:
        if re.match(pattern, word):
            return cls
    # END YOUR CODE
    return "UNK"


def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print "replaced: " + str(float(replaced)/total)
    return res







