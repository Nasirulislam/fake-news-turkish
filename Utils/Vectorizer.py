import scipy as sp
from sklearn.preprocessing import normalize

class TurkishVectorizer:
    def __init__(self, vectorizer, ngram_range=(1,1)):
        self.vectorizer = vectorizer
        self.vectorizer.ngram_range = ngram_range

    def fit(self, df, y=None):
        corpus = list(df["NewsTitle"].str.split()) + list(df["News"].str.split())
        corpus = reduce(list.__add__, corpus)
        self.vectorizer.vocabulary = set(corpus)
        return self

    def transform(self, df, y=None):
        news_title_sparse = self.vectorizer.transform(df["NewsTitle"])
        news_sparse = self.vectorizer.transform(df["News"])
        X = sp.sparse.hstack((news_title_sparse, news_sparse), format='csr')
        return normalize(X)

    def get_params(self, deep=True):
        return {"vectorizer": self.vectorizer}

    def set_params(self, vectorizer=None, ngram_range=None):
        if vectorizer is not None:
            self.vectorizer = vectorizer
        if ngram_range is not None:
            self.vectorizer.ngram_range = ngram_range
