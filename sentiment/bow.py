__author__ = 'Ian Smith'
__author__ = 'Lucas Hure'

import numpy as np
import core
from abc import ABCMeta, abstractmethod


class CategoryMapper(object):
    """
    An abstract class that defines an interface for mapping floating-point
    values to integer categories.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, val):
        return np.round(val)


class DefaultMapper(CategoryMapper):
    def __init__(self):
        super.__init__(self)

    def map(self, val):
        return np.round(val)


class MovieMapper(CategoryMapper):
    def __init__(self):
        super.__init__(self)

    def map(self, val):
        if 1.7 < val < 2.7:
            return 2
        elif val < 8:
            return 1
        else:
            return super.map()


class BagOfWordsModel(core.SentModel):
    def __init__(self, mapper=DefaultMapper()):
        self.model = {}
        self.mapper = mapper

    def set_mapper(self, mapper):
        self.mapper = mapper

    def train(self, X, y):
        self.model = core.build_sentiment_dict(X)

    def predict(self, X):
        raise NotImplementedError


'''
Function
--------
Rate reviews using Bag of Words model

Returns
-------
A new DataFrame with a "Sentiment" column containing the predicted sentiments for each phrase

'''
# def predict(reviews):
#     df = reviews.copy()
#     sentiments = []
#     not_found = []
#     for p in reviews['Phrase']:
#         accum = 0
#         words_in_dic = 0
#         mean = 0
#         split = p.split() # split phrase into words
#         for s in split:
#             if s in dic:
#                 words_in_dic += 1
#                 accum += dic[s]
#         # check for case where no words were found in our dictionary
#         if words_in_dic == 0:
#             mean = random.randint(0, 4) # generate random sentiment from 0 to 4
#             randoms += 1
#             not_found.append(p)
#         else:
#             mean = round(accum / words_in_dic)
#
#         sentiments.append(mean)
#
#     # add 'Sentiment' column containing predicted sentiments to new dataframe
#     df['Sentiment'] = sentiments
#     # print "randomly rated:", randoms
#
#     return df, not_found

