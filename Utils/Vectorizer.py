import scipy as sp
from sklearn.preprocessing import normalize
import pandas as pd


class TurkishVectorizer:
    def __init__(self, vectorizer, ngram_range=(1,1)):
        self.vectorizer = vectorizer
        self.vectorizer.ngram_range = ngram_range

    def fit(self, df, y=None):
        corpus = list(df["NewsTitle"].str.split()) + list(df["News"].str.split())
        # corpus = reduce(list.__add__, corpus)
        # self.vectorizer.vocabulary = set(corpus)
        self.vectorizer.fit(pd.concat([df["NewsTitle"], df["News"]]))
        return self

    def transform(self, df, y=None):
        news_title_sparse = self.vectorizer.transform(df["NewsTitle"])
        news_sparse = self.vectorizer.transform(df["News"])
        setattr(df, "__X", sp.sparse.hstack((news_title_sparse, news_sparse), format='csr'))
        return df

    def get_params(self, deep=True):
        return {"vectorizer": self.vectorizer}

    def set_params(self, vectorizer=None, **kwargs):
        if vectorizer is not None:
            self.vectorizer = vectorizer
        self.vectorizer.set_params(**kwargs)
