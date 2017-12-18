import pandas
from sklearn.base import BaseEstimator
from slang_collections import *
__author__ = 'pravesh'

slang_regex = ur"|".join(slangs)
fake_suffix_regex = ur"|".join(fake_suffixes)
true_suffix_regex = ur"|".join(true_suffixes)

class TurkishFeatureExtractor(BaseEstimator):
    """
    Extract additional features from text.
    """
    def __init__(self, slang=False, suffixes=False, avg_sentence=False, punctuations=False):
        self.slang = slang
        self.suffixes = suffixes
        self.avg_sentences = avg_sentence
        self.punctuations = punctuations

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # transform the given data frame.. concretely, add new columns with the features
        if self.slang:
            X["slangs"] = X["NewsTitle"].str.count(slang_regex) + X["News"].str.count(slang_regex)
            # print X["slangs"]

        if self.suffixes:
            X["fake_suffix_count"] = X["NewsTitle"].str.count(fake_suffix_regex) + X["News"].str.count(fake_suffix_regex)
            X["true_suffix_count"] = X["NewsTitle"].str.count(true_suffix_regex) + X["News"].str.count(true_suffix_regex)

        if self.avg_sentences:
            X["sentences_count"] = X["News"].str.count(r"\.") + 1

        if self.punctuations:
            X["bang_count"] = X["NewsTitle"].str.count("!") + X["News"].str.count("!")
            print X["bang_count"]
            # print pandas.concat(X["bang_count"], axis=1)
        return X