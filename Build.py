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
    # # Urlaub special case checking for German
    # print([s for s in txt_lower if 'ẞ' in s])
    # print([s for s in txt_lower if 'ß' in s])

    return txt_lower


#########################
# Unigram Model
#########################
# German
# training_de = open('data/training.de', 'r')
# text_de = ' '.join(training_de.readlines())
# training_de.close()
# text_de_tokenized = prep(text_de)
#
# uni_de = defaultdict(lambda: 0.0)
# total_de = 0
# for item in text_de_tokenized:
#     for c in item:
#         uni_de[c] += 1
#         total_de += 1
# for token, count in uni_de.items():
#     uni_de[token] = float(count / total_de)
# pickle.dump(dict(uni_de), open('models/uni_de.pkl', 'wb'))

# English
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
# pickle.dump(dict(uni_en), open('models/uni_en.pkl', 'wb'))

# Spanish
training_es = open('data/training.es', 'r')
text_es = ' '.join(training_es.readlines())
training_es.close()
text_es_tokenized = prep(text_es)

uni_es = defaultdict(lambda: 0.0)
total_es = 0
for item in text_es_tokenized:
    for c in item:
        uni_es[c] += 1
        total_es += 1
for token, count in uni_es.items():
    uni_es[token] = float(count / total_es)
pickle.dump(dict(uni_es), open('models/uni_es.pkl', 'wb'))
