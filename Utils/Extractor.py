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
    def __init__(self, slang=False, suffixes=False, sentences_count=False, punctuations=False):
        self.slang = slang
        self.suffixes = suffixes
        self.sentences_count = sentences_count
        self.punctuations = punctuations

    def fit(self, X, y=None):
        self.y_values = y
        return self

    def transform(self, X, y=None):
        # transform the given data frame.. concretely, add new columns with the features
        if self.slang:
            X["slangs"] = X["NewsTitle"].str.count(slang_regex) + X["News"].str.count(slang_regex)
            print zip(self.y_values, X["slangs"].values)

        if self.suffixes:
            X["fake_suffix_count"] = X["NewsTitle"].str.count(fake_suffix_regex) + X["News"].str.count(fake_suffix_regex)
            X["true_suffix_count"] = X["NewsTitle"].str.count(true_suffix_regex) + X["News"].str.count(true_suffix_regex)

        if self.sentences_count:
            X["sentences_count"] = X["News"].str.count(r"\.") + 1

        if self.punctuations:
            X["bang_count"] = X["NewsTitle"].str.count(r"\!") + X["News"].str.count(r"\!")
        return X