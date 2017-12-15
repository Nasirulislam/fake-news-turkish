
class TurkishVectorizer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X):
        # no state or model to train here, so simply return this same class
        self.vectorizer.fit(X)

    def transform(self, df):
        df["NewsTitle"] = df["NewsTitle"].apply(lambda t: stem_turkish_text(self.stemmer, t))
        df["News"] = df["News"].apply(lambda t: stem_turkish_text(self.stemmer, t))
        return self