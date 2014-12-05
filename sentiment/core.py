__author__ = 'Ian Smith, Lucas Hure'

import pandas as pd
from abc import ABCMeta, abstractmethod


# def single_word(ws):
#     def s_match(w):
#         return w.str.match(r'\A[\w-]+\Z')
#     try:
#         return s_match(ws)
#     except AttributeError:
#         return [s_match(w) for w in ws]


def lreduce(fun, ls, z):

    def lreduce_r(fun, ls, acc):
        if len(ls) > 0:
            # print ls
            # print acc
            return lreduce_r(fun, ls[1:], fun(acc, ls[0]))
        else:
            return acc

    return lreduce_r(fun, ls, z)


class SentModel(object):
    """
    An abstract base class that defines an interface for sentiment models.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError


class StatsRating(pd.DataFrame):
    def __init__(self):
        self.accuracy = 0.


def load_reviews(url=None):
    url = 'https://sites.google.com/site/sentananianlucas/data-and-other-material/train.tsv?attredirects=0&d=1' \
        if url is None else url
    revs = pd.read_table(url)
    revs.rename(columns={k: k.lower() for k in revs.columns.values}, inplace=True)
    revs.phrase = revs.phrase.str.lower()
    return revs


def build_one_word_sentiment_dict(reviews_df):
    """
    Returns a dictionary of
    :param reviews_df: the "reviews" data frame as returned by 'load_reviews'
    :return: one-word phrase : sentiment dictionary
    """
    return {p.phrase: p.sentiment for _, p
            in reviews_df[reviews_df.phrase.str.match(r'\A[\w-]+\Z')].iterrows()}


def prediction_report(predictions_df, test_df):
    '''
    Function
    --------
    Check predicted ratings accuracy when testing on our training set

    Returns
    -------
    An accuracy percentage

    '''
    df = pd.DataFrame()

    #
    # for p in range(i):
    #     if predictions['Sentiment'][p] == test_set['Sentiment'][p]:
    #         hits += 1
    #     else:
    #
    # return (hits/i)