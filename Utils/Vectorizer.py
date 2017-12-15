import scipy as sp
from sklearn.preprocessing import normalize

class TurkishVectorizer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, df, y=None):
        corpus = list(df["NewsTitle"].str.split()) + list(df["News"].str.split())
        corpus = reduce(list.__add__, corpus)
        return self.vectorizer.fit_transform(corpus)

    def transform(self, df, y=None):
        news_title_sparse = self.vectorizer.transform(df["NewsTitle"])
        news_sparse = self.vectorizer.transform(df["News"])
        X = sp.sparse.hstack((news_title_sparse, news_sparse), format='csr')
        return normalize(X)
