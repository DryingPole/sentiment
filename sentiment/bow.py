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


class Ensemble(core.SentModel):
    def __init__(self, models):
        self.models = models[:]
        self.fitted_models = None
        # self.res_disc_fun = self._resolve_discord

    def _resolve_discord(self, m1prob, m2prob, m1p, m2p):
        # scores = np.ndarray(dtype=int,shape=len())
        return m1p if m1prob[m1p] + m2prob[m1p] > m1prob[m2p] + m2prob[m2p] else m2p

    def fit(self, X, y):
        self.fitted_models = [m.fit(X, y) for m in self.models]

    def predict(self, X):
        preds = [m.predict(X) for m in self.fitted_models]
        hy_pred = [] # A combined array of the first two models' predictions

        m1 = self.fitted_models[0]
        m2 = self.fitted_models[1]

        discords = filter(lambda x: x is not None,
                          map(lambda a, b, i: (a, b, i) if a != b else None,
                              preds[0], preds[1], range(len(preds[0]))))

        m1prob = m1.predict_proba(X)[:]
        m2prob = m2.predict_proba(X)[:]

        # For disagreeing predictions, take the predicted category with the highes
        # associated confidence.
        for m1p, m2p, ix in discords:
            hy_pred[ix] = self._resolve_discord(m1prob[ix], m2prob[ix], m1p, m2p)

        return hy_pred


