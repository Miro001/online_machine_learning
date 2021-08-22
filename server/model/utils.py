from sklearn import preprocessing as prep
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

import numpy as np
np.random.seed(2302)


def column_selector(column_name):
    return prep.FunctionTransformer(
        lambda X: X[column_name], validate=False)


def init_model():
    # let's build classifiers
    # bag of words for the text feature
    vectorizer = CountVectorizer(max_features=10000)
    # we need a few OneHotEncoders - but I don't like the interface for that and this has the same effect
    # bag of words when all texts are max of one word => one-hot encoding
    amount_encoder = CountVectorizer(max_features=50)
    companyId_encoder = CountVectorizer(max_features=500)
    # combine before doing the regression - feature union requires all features to have the same interface,
    # so to make this work we need to project onto a single column first for each
    all_features = FeatureUnion(
        [
            ['company', make_pipeline(column_selector('CompanyId'), companyId_encoder)],
            ['text', make_pipeline(column_selector('BankEntryText'), vectorizer)],
            ['amount', make_pipeline(column_selector('BankEntryAmount'), amount_encoder)],
        ])
    classifier = MultinomialNB()
    # pipeline the whole thing
    model = Pipeline([('features', all_features), ('nb', classifier)])
    return model

