import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def find_the_top_ngrams_from_count_vect(data, ngram_range, stop_words=None, topn=None):
    """"""
    vec = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range).fit(data)
    bag_of_words = vec.transform(data)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    common_words = words_freq[:topn]
    words = []
    frequencies = []
    for word, freq in common_words:
        words.append(word)
        frequencies.append(freq)

    dfxa = pd.DataFrame({'Word': words, 'Freq': frequencies})
    return dfxa


def get_top_words(topn, dataframe, column_name):
    """"""
    stop_words = 'english'
    unigrams = find_the_top_ngrams_from_count_vect(dataframe[column_name],
                                                   (1, 1), stop_words, topn)
    bigrams = find_the_top_ngrams_from_count_vect(dataframe[column_name],
                                                  (2, 2), stop_words, topn)
    trigrams = find_the_top_ngrams_from_count_vect(dataframe[column_name],
                                                   (3, 3), stop_words, topn)

    return unigrams, bigrams, trigrams

