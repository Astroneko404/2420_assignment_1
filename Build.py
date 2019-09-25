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
    txt_split = re.split('\u00ad| |\n|\t|,|\.|!|\\?|;|:|-|–|—|~|%|_|\\|/|/|º|¿|¡|<|>|\^|\(|\)|\[|\]|\\|\'|\'|`|"', txt)
    txt_tokenized = [x for x in txt_split if x and not has_num(x)]

    # Lowercase
    txt_lower = [x.lower() for x in txt_tokenized]
    # # Umlaut special case checking for German
    # print([s for s in txt_lower if 'ẞ' in s])
    # print([s for s in txt_lower if 'ß' in s])

    return txt_lower


def unigram_build(text_tokenized):
    uni_count = defaultdict(lambda: 0)
    uni = defaultdict(lambda: 0.0)
    total_de = 0
    for item in text_tokenized:
        uni_count['<s>'] += 1
        uni_count['</s>'] += 1
        for c in item:
            uni_count[c] += 1
            total_de += 1

    for token, count in uni_count.items():
        uni[token] = float(count / total_de)

    return dict(uni)


def bigram_build(text_tokenized):
    bi_count_1 = defaultdict(lambda: 0)  # C(w_{i-1})
    bi_count_2 = defaultdict(lambda: 0)  # C(w_{i-1}, w_i)
    bi = defaultdict(lambda: defaultdict(lambda: 0.0))  # The model

    for token in text_tokenized:  # Build dict count
        n = len(token)

        for i in range(n + 1):
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

    return dict(bi)


def print_nested_dict(d):
    """
    :param d: The nested defaultdict to be printed
    :return: None
    """
    for item, keys in d.items():
        print(item, end='')
        print(': ')
        for subitem, prob in keys.items():
            print(subitem, prob)
        print()
    return


if __name__ == '__main__':

    out_path = 'models/en/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    training = open('data/training.en', 'r')
    text = ' '.join(training.readlines())
    training.close()

    ########################
    # Preprocess
    ########################
    txt_tokenized = prep(text)

    ########################
    # Unigram
    ########################
    # unigram = unigram_build(txt_tokenized)
    # pickle.dump(unigram, open(out_path + 'unigram.pkl', 'wb'))

    ########################
    # Bigram
    ########################
    # bigram = bigram_build(txt_tokenized)
    # pickle.dump(dict(bigram), open(out_path + 'bigram.pkl', 'wb'))

    ########################
    # Trigram (no smoothing)
    ########################
    tri_count_3 = defaultdict(lambda: 0)  # C(w_{i-2}, w_{i-1}, w_i)
    tri_count_2 = defaultdict(lambda: 0)  # C(w_{i-2}, w_{i-1})
    tri_no_smoothing = defaultdict(lambda: defaultdict(lambda: 0.0))  # The model

    for token in txt_tokenized:  # Build dict count
        n = len(token)
        for i in range(1, n + 1):
            c = token[i] if i < n else '</s>'
            prev_2 = token[i - 2] if i >= 2 else '<s>'
            prev_1 = token[i - 1]

            tri_count_2[(prev_2, prev_1)] += 1
            tri_count_3[(prev_2, prev_1, c)] += 1

    for token in txt_tokenized:  # Build model
        n = len(token)
        for i in range(1, n + 1):
            c = token[i] if i < n else '</s>'
            prev_2 = token[i - 2] if i >= 2 else '<s>'
            prev_1 = token[i - 1]
            tri_no_smoothing[c][(prev_2, prev_1)] = float(
                tri_count_3[(prev_2, prev_1, c)] / tri_count_2[(prev_2, prev_1)]
            )

    for key, value in tri_no_smoothing.items():  # Format transfer
        tri_no_smoothing[key] = dict(value)

    # print_nested_dict(tri_no_smoothing)

    # pickle.dump(dict(tri_no_smoothing), open(out_path + 'trigram_no_smoothing.pkl', 'wb'))

    ########################
    # Trigram (Laplace)
    ########################
    unigram = pickle.load(open(out_path + 'unigram.pkl', 'rb'))
    v = len(unigram)
    tri_laplace = defaultdict(lambda: defaultdict(lambda: 0.0))  # The model

    for context, context_count in tri_count_2.items():  # Build model using dictionaries of count from previous part
        for c, _ in unigram.items():
            full = (context[0], context[1], c)
            total_count = tri_count_3[full] + 1
            new_denominator = context_count + v
            tri_laplace[c][context] = float(total_count / new_denominator)

    for key, value in tri_laplace.items():
        tri_laplace[key] = dict(value)

    # print_nested_dict(tri_laplace)

    # pickle.dump(dict(tri_laplace), open(out_path + 'trigram_laplace.pkl', 'wb'))

    ########################
    # Linear Interpolation
    ########################
    lam = float(1/3)  # Lambda parameters are equally weighted
    unigram = pickle.load(open(out_path + 'unigram.pkl', 'rb'))
    bigram = pickle.load(open(out_path + 'bigram.pkl', 'rb'))
    trigram = pickle.load(open(out_path + 'trigram_no_smoothing.pkl', 'rb'))
    bigram_inverted = pickle.load(open(out_path + 'bigram_inverted.pkl', 'rb'))
    trigram_inverted = pickle.load(open(out_path + 'trigram_inverted.pkl', 'rb'))
    tri_lin_interpolation = defaultdict(lambda: defaultdict(lambda: 0.0))

    for context, context_dict in trigram_inverted.items():  # Build model using dictionaries of count from previous part
        for c, _ in unigram.items():
            # if c == '<s>':
            #     continue
            prev_2 = context[0]
            prev_1 = context[1]
            score_tri = trigram[c][context] if c in trigram_inverted[context] else 0.0
            score_bi = bigram[c][prev_1] if c in bigram_inverted[prev_1] else 0.0
            score_uni = unigram[c]
            total_score = (score_tri + score_bi + score_uni) / 3
            tri_lin_interpolation[c][context] = total_score

    for key, value in tri_lin_interpolation.items():
        tri_lin_interpolation[key] = dict(value)

    tri_lin_interpolation = dict(tri_lin_interpolation)

    # print_nested_dict(tri_lin_interpolation)

    pickle.dump(tri_lin_interpolation, open(out_path + 'trigram_interpolation.pkl', 'wb'))

    ########################
    # Backoff
    ########################
    unigram = pickle.load(open(out_path + 'unigram.pkl', 'rb'))
    bigram = pickle.load(open(out_path + 'bigram.pkl', 'rb'))
    trigram = pickle.load(open(out_path + 'trigram_no_smoothing.pkl', 'rb'))
    tri_backoff = defaultdict(lambda: defaultdict(lambda: 0.0))

    for context, context_count in tri_count_2.items():  # Build model using dictionaries of count from previous part
        for c, _ in unigram.items():
            if c == '<s>':
                continue
            prev_2 = context[0]
            prev_1 = context[1]
            score_tri = trigram[c][context] if c in trigram and context in trigram[c] else None
            score_bi = bigram[c][prev_1] if c in bigram and prev_1 in bigram[c] else None
            score_uni = unigram[c]

            final_score = 0.0
            alpha = 0.85

            if score_tri:
                final_score = score_tri
            elif score_bi:
                final_score = alpha * score_bi
            else:
                final_score = alpha**10 * score_uni

            tri_backoff[c][context] = final_score

    for key, value in tri_backoff.items():
        tri_backoff[key] = dict(value)
    tri_backoff = dict(tri_backoff)

    # print_nested_dict(tri_backoff)

    pickle.dump(tri_backoff, open(out_path + 'trigram_backoff.pkl', 'wb'))

    ########################
    # Trigram (Katz Backoff)
    ########################
    unigram = pickle.load(open(out_path + 'unigram.pkl', 'rb'))
    bigram = pickle.load(open(out_path + 'bigram.pkl', 'rb'))
    trigram = pickle.load(open(out_path + 'trigram_no_smoothing.pkl', 'rb'))
    bigram_inverted = pickle.load(open(out_path + 'bigram_inverted.pkl', 'rb'))
    trigram_inverted = pickle.load(open(out_path + 'trigram_inverted.pkl', 'rb'))
    tri_katz_backoff = defaultdict(lambda: defaultdict(lambda: 0.0))

    # # Build inverted index for bigram
    # bigram_inverted = defaultdict(lambda: defaultdict(lambda: 0.0))
    # for token, token_dict in bigram.items():
    #     for context, prob in token_dict.items():
    #         bigram_inverted[context][token] = prob
    # for key, value in bigram_inverted.items():
    #     bigram_inverted[key] = dict(value)
    # bigram_inverted = dict(bigram_inverted)
    # print_nested_dict(bigram_inverted)
    # pickle.dump(bigram_inverted, open(out_path + 'bigram_inverted.pkl', 'wb'))

    # # Build inverted index for trigram (no smoothing)
    # trigram_inverted = defaultdict(lambda: defaultdict(lambda: 0.0))
    # for token, token_dict in trigram.items():
    #     for context, prob in token_dict.items():
    #         trigram_inverted[context][token] = prob
    # for key, value in trigram_inverted.items():
    #     trigram_inverted[key] = dict(value)
    # trigram_inverted = dict(trigram_inverted)
    # pickle.dump(trigram_inverted, open(out_path + 'trigram_inverted.pkl', 'wb'))
    # print_nested_dict(trigram_inverted)

    lam = 0.98

    # Calculate all P_{katz}(z|y)
    katz_bi = defaultdict(lambda: defaultdict(lambda: 0.0))  # To store P_{katz}(z|y)
    for y, y_dict in bigram_inverted.items():
        beta_y = 1.0 - lam
        # zero_list = [z for z in unigram if z != '<s>' and y not in bigram[z]]
        denom = [lam * unigram[z] for z in bigram if z != '<s>' and y not in bigram[z]]
        alpha_y = beta_y / sum(denom)

        for z, _ in unigram.items():
            if z == '<s>':
                continue

            final_score = 0.0
            if z in y_dict:
                final_score = lam * bigram[z][y]
            else:
                final_score = alpha_y * (lam * unigram[z])
            katz_bi[z][y] = final_score

    # Calculate all P_{katz}(z|xy)
    katz_tri = defaultdict(lambda: defaultdict(lambda: 0.0))  # To store P_{katz}(z|xy)
    for (x, y), xy_dict in trigram_inverted.items():
        beta_xy = 1.0 - lam
        beta_y = 1.0 - lam
        denom_xy = [katz_bi[z][y] for z in unigram if z != '<s>' and (x, y) not in trigram[z]]
        denom_y = [lam * unigram[z] for z in bigram if z != '<s>' and y not in bigram[z]]
        alpha_xy = beta_xy / sum(denom_xy)
        alpha_y = beta_y / sum(denom_y)

        for z, _ in unigram.items():
            if z == '<s>':
                continue

            final_score = 0.0
            if z in xy_dict:
                final_score = lam * trigram[z][(x, y)]
                # final_score = trigram[z][(x, y)]
            elif y in bigram[z]:
                final_score = alpha_xy * katz_bi[z][y]
            else:
                final_score = alpha_y * lam * unigram[z]
            katz_tri[z][(x, y)] = final_score

    for key, value in katz_tri.items():
        katz_tri[key] = dict(value)
    katz_tri = dict(katz_tri)

    # pickle.dump(katz_tri, open(out_path + 'trigram_katz_backoff.pkl', 'wb'))
