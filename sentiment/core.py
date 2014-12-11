__author__ = 'Ian Smith, Lucas Hure'

import pandas as pd
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

neg_dict = {"not": 1,
            "neither": 1,
            "can't": 1,
            "isn't": 1,
            "aint": 1}


def scatter_exp_vs_pred(df=None, expected=None, predicted=None):
    """
    A function that takes a data frame with 'expected' and
    'predicted' columns and creates a scatter plot for those
    columns
    :param df:
    :return: None.
    """
    if df is None:
        df = pd.DataFrame()
        df['expected'] = expected
        df['predicted'] = predicted

    # res_df = pd.DataFrame(data=np.transpose([pred, XY_all[3]]), columns=["predicted", "expected"])
    rstats = df.groupby(["predicted", "expected"]).size().\
        reset_index().rename(columns={0: "counts"})

    plt.figure(figsize=(12, 8))
    plt.title("Predicted vs. Expected Sentiment Values")
    plt.ylabel("Predicted Sentiment")
    plt.xlabel("Expected Sentiment")
    plt.scatter(rstats.expected, rstats.predicted, s=[0.1*c for c in rstats.counts], alpha=0.6)
    plt.show()

def lreduce(fun, ls, z):
    """
    A function following the functional paradigm. It is equivalent to
    reduce left.
    :param fun: A function that operates on an accumulator and
     one element of the list and returns an accumulator. Has the
     signature f(accumulator, element) => accumulator
    :param ls: A list of elements
    :param z: The base value for the accumulator
    :return: The result of applying fun to successive elements of the list
    """
    acc = z
    for e in ls:
        acc = fun(acc, e)
    return acc


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


def load_negation_dict(path='../resources/neg_words.csv'):
    ndf = pd.read_csv(path, names=['neg_word'], index_col=0)
    ndf['Value'] = True
    return ndf.to_dict()['Value']


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


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def vectorize_phrases(phrase_list):
    v = CountVectorizer(min_df=0.00, binary=True)
    v.fit(phrase_list)
    return v.transform(phrase_list).tocsc()