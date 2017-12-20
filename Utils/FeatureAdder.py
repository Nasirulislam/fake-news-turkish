import numpy
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, MinMaxScaler

__author__ = 'pravesh'


class TurkishFeatureAdder(BaseEstimator):
    # original df
    def __init__(self, n_components=None, n_iter=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.pca = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=13)

    def fit(self, X, y=None):
        self.pca.n_components = self.n_components
        self.pca.n_iter = self.n_iter

        self.pca.fit(getattr(X, "__X"), y)
        return self

    def transform(self, X, y=None):
        result = self.pca.transform(getattr(X,"__X"))
        combined_feature = [result]

        for features in ["slangs", "fake_suffix_count", "true_suffix_count", "bang_count"]:
            if features in X.columns:
                combined_feature.append(X[[features]])


        # print "Captured variance is %.2f" % self.pca.explained_variance_ratio_.sum()
        # stack them together
        result = MinMaxScaler().fit_transform(numpy.hstack(combined_feature))
        return result