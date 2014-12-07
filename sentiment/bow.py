__author__ = 'Ian Smith, Lucas Hure'

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
        return int(np.round(val))


class DefaultMapper(CategoryMapper):
    """
    Maps a floating-point to a category via simple rounding.
    """
    def __init__(self):
        CategoryMapper.__init__(self)

    def map(self, val):
        return np.round(val)


class MovieMapper(CategoryMapper):
    """

    """
    def __init__(self, round_up=0.1):
        CategoryMapper.__init__(self)
        self.round_up = round_up

    def map(self, val):
        return np.round(val + self.round_up)


class BagOfWordsModel(core.SentModel):
    def __init__(self, mapper=DefaultMapper()):
        self.model = None
        self.mapper = mapper
        self._not_found = 0  # a value re-initialized for each invocation of 'predict'

    def set_mapper(self, mapper):
        self.mapper = mapper

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        # sw = re.compile(r'\A[\w-]+\Z')
        # self.model = {xs: ys for xs, ys in zip(X, y) if sw.match(xs)}
        self.model = {xs: ys for xs, ys in zip(X, y)}

    def predict(self, X, default_caty=2, weight_map={0: 5, 1: 3, 2: 1, 3: 3, 4: 5}):
        self._not_found = 0
        predictions = []
        for phrase in X:
            words = phrase.split()
            sent_accum = 0.0
            weight = 0
            for w in words:
                if w in self.model:
                    sent_val = self.model[w]
                    sent_accum += sent_val * weight_map[sent_val]
                    weight += weight_map[sent_val]
            if weight != 0:
                caty = self.mapper.map(sent_accum / weight)
                predictions.append(caty)
            else:
                self._not_found += 1
                predictions.append(default_caty)
        return predictions


class Ensemble(core.sentMod)