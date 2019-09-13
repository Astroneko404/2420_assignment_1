from collections import defaultdict
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
    for item, keys in d.items():
        print(item + ': ', end='')
        print(dict(keys))
    return


if __name__ == '__main__':
    ###############
    # German
    ###############
    # Preprocess
    training_de = open('data/training.de', 'r')
    text_de = ' '.join(training_de.readlines())
    training_de.close()
    text_de_tokenized = prep(text_de)

    # # Unigram
    # uni_de_count = defaultdict(lambda: 0)
    # uni_de = defaultdict(lambda: 0.0)
    # total_de = 0
    # for item in text_de_tokenized:
    #     for c in item:
    #         uni_de_count[c] += 1
    #         total_de += 1
    # for token, count in uni_de_count.items():
    #     uni_de[token] = float(count / total_de)
    # # pickle.dump(dict(uni_de), open('models/uni_de.pkl', 'wb'))

    # Bigram
    de_total_count = defaultdict(lambda: 0)
    bi_de_count = defaultdict(lambda: defaultdict(lambda: 0))
    bi_de = defaultdict(lambda: defaultdict(lambda: 0.0))
    for token in text_de_tokenized:
        de_total_count['<s>'] += 1
        de_total_count['</s>'] += 1
        for i in range(len(token)):
            c = token[i]
            c_prev = token[i-1] if i > 0 else '<s>'
            bi_de_count[c][c_prev] += 1
            de_total_count[c] += 1
        last_char = token[len(token)-1]
        bi_de_count['</s>'][last_char] += 1
    for c, bi_dict in bi_de_count.items():
        for c_prev, bi_count in bi_dict.items():
            prob = float(bi_count / de_total_count[c_prev])
            bi_de[c][c_prev] = prob
    bi_de['<s>']['</s>'] = 1.0
    for key, value in bi_de.items():
        bi_de[key] = dict(value)
    print_nested_dict(bi_de)
    pickle.dump(dict(bi_de), open('models/bi.de.pkl', 'wb'))

    ###############
    # English
    ###############
    # training_en = open('data/training.en', 'r')
    # text_en = ' '.join(training_en.readlines())
    # training_en.close()
    # text_en_tokenized = prep(text_en)
    #
    # uni_en = defaultdict(lambda: 0.0)
    # total_en = 0
    # for item in text_en_tokenized:
    #     for c in item:
    #         uni_en[c] += 1
    #         total_en += 1
    # for token, count in uni_en.items():
    #     uni_en[token] = float(count / total_en)
    # # pickle.dump(dict(uni_en), open('models/uni_en.pkl', 'wb'))

    ###############
    # Spanish
    ###############
    # training_es = open('data/training.es', 'r')
    # text_es = ' '.join(training_es.readlines())
    # training_es.close()
    # text_es_tokenized = prep(text_es)
    #
    # uni_es = defaultdict(lambda: 0.0)
    # total_es = 0
    # for item in text_es_tokenized:
    #     for c in item:
    #         uni_es[c] += 1
    #         total_es += 1
    # for token, count in uni_es.items():
    #     uni_es[token] = float(count / total_es)
    # # pickle.dump(dict(uni_es), open('models/uni_es.pkl', 'wb'))

    #########################
    # Bigram Model
    #########################
    # German

