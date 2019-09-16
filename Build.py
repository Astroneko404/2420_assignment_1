from collections import defaultdict
import os
import pickle
import re


def has_num(txt):
    return any(char.isdigit() for char in txt)


def prep(txt):
    """
    :param txt: The input text
    :return: List of tokens that is preprocessed
    """
    # Tokenization
    txt_split = re.split('\u00ad| |\n|\t|,|\.|!|\\?|;|:|-|–|—|~|%|_|\\|/|/|º|¿|¡|<|>|\^|\(|\)|\[|\]|\\|\'|`|"', txt)
    txt_tokenized = [x for x in txt_split if x and not has_num(x)]

    # Lowercase
    txt_lower = [x.lower() for x in txt_tokenized]
    # # Umlaut special case checking for German
    # print([s for s in txt_lower if 'ẞ' in s])
    # print([s for s in txt_lower if 'ß' in s])

    return txt_lower


def print_nested_dict(d):
    """
    :param d: The nested defaultdict to be printed
    :return: None
    """
    for item, keys in d.items():
        print(item + ': ', end='')
        print(dict(keys))
    return


if __name__ == '__main__':

    out_path = 'models/es/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Preprocess
    training = open('data/training.es', 'r')
    text = ' '.join(training.readlines())
    training.close()
    text_tokenized = prep(text)

    # Unigram
    uni_count = defaultdict(lambda: 0)
    uni = defaultdict(lambda: 0.0)
    total_de = 0
    for item in text_tokenized:
        for c in item:
            uni_count[c] += 1
            total_de += 1

    for token, count in uni_count.items():
        uni[token] = float(count / total_de)

    pickle.dump(dict(uni), open(out_path + 'unigram.pkl', 'wb'))

    # Bigram
    bi_count_1 = defaultdict(lambda: 0)  # C(w_{i-1})
    bi_count_2 = defaultdict(lambda: 0)  # C(w_{i-1}, w_i)
    bi = defaultdict(lambda: defaultdict(lambda: 0.0))  # The model

    for token in text_tokenized:  # Build dict count
        n = len(token)

        for i in range(n+1):
            c = token[i] if i < n else '</s>'
            c_prev = token[i - 1] if i > 0 else '<s>'
            bi_count_1[c_prev] += 1
            bi_count_2[(c_prev, c)] += 1

    for token in text_tokenized:  # Build model
        n = len(token)

        for i in range(n + 1):
            c = token[i] if i < n else '</s>'
            c_prev = token[i - 1] if i > 0 else '<s>'
            bi[c][c_prev] = float(bi_count_2[(c_prev, c)] / bi_count_1[c_prev])

    # print(bi_count_2)
    for key, value in bi.items():  # Format transfer
        bi[key] = dict(value)

    pickle.dump(dict(bi), open(out_path + 'bigram.pkl', 'wb'))

    # Trigram (no smoothing)
    tri_count_3 = defaultdict(lambda: 0)  # C(w_{i-2}, w_{i-1}, w_i)
    tri_count_2 = defaultdict(lambda: 0)  # C(w_{i-2}, w_{i-1})
    tri_no_smoothing = defaultdict(lambda: defaultdict(lambda: 0.0))  # The model

    for token in text_tokenized:  # Build dict count
        n = len(token)
        for i in range(1, n+1):
            c = token[i] if i < n else '</s>'
            prev_2 = token[i-2] if i >= 2 else '<s>'
            prev_1 = token[i-1]

            tri_count_2[(prev_2, prev_1)] += 1
            tri_count_3[(prev_2, prev_1, c)] += 1

    for token in text_tokenized:  # Build model
        n = len(token)
        for i in range(1, n+1):
            c = token[i] if i < n else '</s>'
            prev_2 = token[i-2] if i >= 2 else '<s>'
            prev_1 = token[i-1]
            tri_no_smoothing[c][(prev_2, prev_1)] = float(
                tri_count_3[(prev_2, prev_1, c)] / tri_count_2[(prev_2, prev_1)]
            )

    for key, value in tri_no_smoothing.items():  # Format transfer
        tri_no_smoothing[key] = dict(value)

    pickle.dump(dict(tri_no_smoothing), open(out_path + 'trigram_no_smoothing.pkl', 'wb'))
