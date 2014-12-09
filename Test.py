__author__ = 'imksmith'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import pandas as pd

import imp
source_path = './sentiment/'
core = imp.load_source('core', source_path + 'core.py')
bow = imp.load_source('bow', source_path + 'bow.py')
bayes = imp.load_source('bayes', source_path + 'bayes.py')


reviews = core.load_reviews('./resources/train.tsv')
rarray = reviews.phrase.tolist()
vectorizer = CountVectorizer(min_df=0.00, binary=True)
vectorizer.fit(rarray)
X = vectorizer.transform(rarray).tocsc()
Y = reviews.sentiment.tolist()

xtrain, xtest, ytrain, ytest = train_test_split(X, Y)
ensemble = bow.Ensemble([MultinomialNB(),
                         BernoulliNB(),
                         LogisticRegression(C=2.4, class_weight={0: 2.5,
                                                                 1: 2,
                                                                 2: 1,
                                                                 3: 2,
                                                                 4: 2})])
ensemble.fit(xtrain, ytrain)
Y_pred = ensemble.predict(xtest)

# XT_df = core.load_reviews('./resources/test.tsv')
# X_Test = core.vectorize_phrases(XT_df.phrase.tolist())

# print X_Test.shape
# print X.shape

# ensemble = bow.Ensemble([MultinomialNB(), BernoulliNB()])
# ensemble.fit(X, Y)
# Y_pred = ensemble.predict(X_Test)
# XT_df['Sentiment'] = Y_pred
# XT_df[['phraseid','Sentiment']].to_csv('./resources/kaggle_sub.csv')

print "accuracy is:"
print sum(map(lambda x, y: 1 if x == y else 0, Y_pred, ytest)) / float(len(ytest))
