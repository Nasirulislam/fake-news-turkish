from nltk import RegexpTokenizer
from sklearn.base import BaseEstimator
from Utils.StopWords import StopWords

stop_word = StopWords()

def stem_turkish_text(stemmer, text):
    text = text.lower()
    tokenized = RegexpTokenizer(r"\w+").tokenize(text)
    tokenized = stop_word.remove_stop_words(tokenized)
    return u" ".join(stemmer.stemWords(tokenized))

class TurkishPreprocessor(BaseEstimator):
    def __init__(self, stemmer):
        self.stemmer = stemmer

    def fit(self, X, y=None):
        # no state or model to train here, so simply return this same class
        return self

    def transform(self, df, y=None):
        df["NewsTitle"] = df["NewsTitle"].apply(lambda t: stem_turkish_text(self.stemmer, t))
        df["News"] = df["News"].apply(lambda t: stem_turkish_text(self.stemmer, t))
        return df

    def get_params(self, deep=True):
        return {"stemmer": self.stemmer}

    def set_params(self, **params):
        self.stemmer = params.get("stemmer", self.stemmer)