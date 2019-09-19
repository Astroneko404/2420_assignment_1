import pickle
from Build import prep


if __name__ == '__main__':

    # Preprocess
    test_file = open('data/test', 'r')
    test = ' '.join(test_file.readlines())
    test_file.close()
    test_tokenized = prep(test)

    # Models
    out_path = 'models/de/'
    unigram = pickle.load(open(out_path + 'unigram.pkl', 'rb'))
    bigram = pickle.load(open(out_path + 'bigram.pkl', 'rb'))
