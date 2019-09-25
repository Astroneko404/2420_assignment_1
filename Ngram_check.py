import csv
import pickle


# This is for csv output only
if __name__ == '__main__':
    out_path = 'models/en/'
    trigram_no_smoothing = pickle.load(open(out_path + 'trigram_no_smoothing.pkl', 'rb'))
    trigram_laplace = pickle.load(open(out_path + 'trigram_laplace.pkl', 'rb'))
    trigram_backoff = pickle.load(open(out_path + 'trigram_backoff.pkl', 'rb'))
    trigram_interpolation = pickle.load(open(out_path + 'trigram_interpolation.pkl', 'rb'))
    trigram_katz = pickle.load(open(out_path + 'trigram_katz_backoff.pkl', 'rb'))

    # CSV output for Google Sheets
    token = []
    prob_list = []
    for c in trigram_interpolation:
        t = 'th' + c
        prob = trigram_interpolation[c][('t', 'h')] if ('t', 'h') in trigram_interpolation[c] else 0.0
        token.append(t)
        prob_list.append(prob)
    rows = zip(token, prob_list)
    with open('interpolation.csv', "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

