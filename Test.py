import csv
import math
import pickle
from Build import prep


if __name__ == '__main__':

    ################
    # Preprocess
    ################
    with open('data/test') as test_file:
        content = test_file.readlines()
    content = [x.strip() for x in content]
    test_file.close()
    test_tokenized = []
    for test in content:
        line = prep(test)
        test_tokenized.append(line)

    ################
    # Models
    ################
    out_path = 'models/en/'
    unigram = pickle.load(open(out_path + 'unigram.pkl', 'rb'))
    bigram = pickle.load(open(out_path + 'bigram.pkl', 'rb'))
    trigram_no_smoothing = pickle.load(open(out_path + 'trigram_no_smoothing.pkl', 'rb'))
    trigram_laplace = pickle.load(open(out_path + 'trigram_laplace.pkl', 'rb'))
    trigram_backoff = pickle.load(open(out_path + 'trigram_backoff.pkl', 'rb'))
    trigram_interpolation = pickle.load(open(out_path + 'trigram_interpolation.pkl', 'rb'))
    trigram_katz_backoff = pickle.load(open(out_path + 'trigram_katz_backoff.pkl', 'rb'))

    ################
    # Perplexity
    ################
    s = 100  # Value for probability that is 0

    # Unigram
    perplexity_unigram = []
    for line in test_tokenized:
        total = 1.0
        for word in line:
            char_list = ['<s>'] + [c for c in word] + ['</s>']
            N = len(char_list)
            product = 1.0

            for i in range(1, N):  # Skip the <s> tag
                prob = 1 / unigram[char_list[i]]
                # prob = math.log(1 / unigram[word[i]], 10)  # Use logarithm to scale
                product *= prob
            total += product ** (1. / N)
        perplexity_unigram.append(total)
    assert len(perplexity_unigram) == 100

    # Bigram
    perplexity_bigram = []
    for line in test_tokenized:
        total = 1.0
        for word in line:
            char_list = ['<s>'] + [c for c in word] + ['</s>']
            N = len(char_list)
            product = 1.0

            for i in range(1, N):
                c = char_list[i]
                prev_1 = char_list[i-1]
                prob = 1 / bigram[c][prev_1] if prev_1 in bigram[c] else s
                product *= prob
            total += product ** (1. / N)
        perplexity_bigram.append(total)
    assert len(perplexity_bigram) == 100

    # Trigram (no smoothing)
    perplexity_trigram_no_smoothing = []
    for line in test_tokenized:
        total = 1.0
        for word in line:
            char_list = ['<s>'] + [c for c in word] + ['</s>']
            N = len(char_list)
            product = 1.0

            for i in range(2, N):  # For trigram we start from 2
                c = char_list[i]
                prev_1 = char_list[i-1]
                prev_2 = char_list[i-2]
                context = (prev_2, prev_1)
                prob = 1 / trigram_no_smoothing[c][context] if context in trigram_no_smoothing[c] else s
                product *= prob
            total += product ** (1. / N)
        perplexity_trigram_no_smoothing.append(total)
    assert len(perplexity_trigram_no_smoothing) == 100

    # Trigram (Laplace)
    perplexity_trigram_laplace = []
    for line in test_tokenized:
        total = 1.0
        for word in line:
            char_list = ['<s>'] + [c for c in word] + ['</s>']
            N = len(char_list)
            product = 1.0

            for i in range(2, N):  # For trigram we start from 2
                c = char_list[i]
                prev_1 = char_list[i - 1]
                prev_2 = char_list[i - 2]
                context = (prev_2, prev_1)
                prob = 1 / trigram_laplace[c][context] if context in trigram_laplace[c] else s
                product *= prob
            total += product ** (1. / N)
        perplexity_trigram_laplace.append(total)
    assert len(perplexity_trigram_laplace) == 100

    # Trigram (Backoff)
    perplexity_trigram_backoff = []
    for line in test_tokenized:
        total = 1.0
        for word in line:
            char_list = ['<s>'] + [c for c in word] + ['</s>']
            N = len(char_list)
            product = 1.0

            for i in range(2, N):  # For trigram we start from 2
                c = char_list[i]
                prev_1 = char_list[i - 1]
                prev_2 = char_list[i - 2]
                context = (prev_2, prev_1)
                prob = 1 / trigram_backoff[c][context] if context in trigram_backoff[c] else s
                product *= prob
            total += product ** (1. / N)
        perplexity_trigram_backoff.append(total)
    assert len(perplexity_trigram_backoff) == 100

    # Trigram (Linear Interpolation)
    perplexity_trigram_interpolation = []
    for line in test_tokenized:
        total = 1.0
        for word in line:
            char_list = ['<s>'] + [c for c in word] + ['</s>']
            N = len(char_list)
            product = 1.0

            for i in range(2, N):  # For trigram we start from 2
                c = char_list[i]
                prev_1 = char_list[i - 1]
                prev_2 = char_list[i - 2]
                context = (prev_2, prev_1)
                prob = 1 / trigram_interpolation[c][context] if context in trigram_interpolation[c] else s
                product *= prob
            total += product ** (1. / N)
        perplexity_trigram_interpolation.append(total)
    assert len(perplexity_trigram_interpolation) == 100

    # Trigram (Katz Backoff)
    perplexity_trigram_katz_backoff = []
    for line in test_tokenized:
        total = 1.0
        for word in line:
            char_list = ['<s>'] + [c for c in word] + ['</s>']
            N = len(char_list)
            product = 1.0

            for i in range(2, N):  # For trigram we start from 2
                c = char_list[i]
                prev_1 = char_list[i - 1]
                prev_2 = char_list[i - 2]
                context = (prev_2, prev_1)
                prob = 1 / trigram_katz_backoff[c][context] if context in trigram_katz_backoff[c] else s
                product *= prob
            total += product ** (1. / N)
        perplexity_trigram_katz_backoff.append(total)
    assert len(perplexity_trigram_katz_backoff) == 100

    # Result output
    rows = zip(perplexity_unigram, perplexity_bigram, perplexity_trigram_no_smoothing, perplexity_trigram_laplace,
               perplexity_trigram_backoff, perplexity_trigram_interpolation,perplexity_trigram_katz_backoff)
    with open('en_perplexity.csv', "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    ################
    # Result
    ################
    # print('Unigram result:')
    # print(perplexity_unigram)
    # print()
    #
    # print('Bigram result:')
    # print(perplexity_bigram)
    # print()

    print('Trigram (no smoothing) result:')
    print(perplexity_trigram_no_smoothing)
    print()
    #
    # print('Trigram (Laplace) result:')
    # print(perplexity_trigram_laplace)
    # print()
    #
    print('Trigram (Backoff) result:')
    print(perplexity_trigram_backoff)
    print()

    print('Trigram (Katz backoff) result:')
    print(perplexity_trigram_katz_backoff)
    print()

    print('Trigram (Linear Interpolation) result:')
    print(perplexity_trigram_interpolation)
    print()
