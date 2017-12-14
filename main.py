# coding=utf-8
"""
Fake news detector for Turkish language
"""
from urlparse import urlparse
import scipy as sp
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import snowballstemmer
from nltk.tokenize import RegexpTokenizer
from Utils.StopWords import StopWords

stop_word = StopWords()


def read_file_xlsx(file_path):
    # check that the file is indeed an excel file
    assert file_path[-5:] == ".xlsx"
    df = pd.read_excel(file_path, index_col=0, encoding="utf-8")
    return df

def read_file_csv(file_path):
    return pd.read_csv(file_path, index_col=0, encoding="utf-8")


def stem_turkish_text(stemmer, text):
    text = text.lower()
    tokenized = RegexpTokenizer(r"\w+").tokenize(text)
    tokenized = stop_word.remove_stop_words(tokenized)
    return u" ".join(stemmer.stemWords(tokenized))


def transform_df(df):
    """
    Takes input a pandas data frame and returns a pre-processed data frame to be used for feature extraction.
    :param df:
    :return:
    """
    # we work with only these columns for now
    df = df.loc[:, ["NewsDate", "Url", "NewsTitle", "News", "Value"]]

    # transform NewsDate column to a boolean indicating whether the date is present or not
    values = (df["NewsDate"] != pd.NaT) * 1
    df["NewsDate"] = values

    # transform Url column to just the hostname or "Social" if the news source is social media
    df["Url"] = df["Url"].apply(lambda url: "Social" if url == "Social Media Virals" else
                                urlparse(url).hostname.replace("www.", "").replace(".", "") if "http" in url else url)

    # transform Value from FAKE / TRUE to True and False depending if the news is true or fake
    df.loc[df["Value"] == "FAKE", "Value"] = False
    df.loc[df["Value"] == "TRUE", "Value"] = True

    # stem the turkish words
    stemmer = snowballstemmer.TurkishStemmer()
    df["NewsTitle"] = df["NewsTitle"].apply(lambda t: stem_turkish_text(stemmer, t))
    return df


def to_features(df, feature):
    """
    Convert the df to features..
    NewsTitle and News to either n-gram or TF-IDF
    :param feature: Either 'ngram' or 'tfidf'. For now, ngram corresponds to bigram
    :param df:
    :return:
    """
    # first retrieve the corpus
    corpus=list(df["NewsTitle"].str.split()) + list(df["News"].str.split())
    corpus = reduce(list.__add__, corpus)
    vectorizer = CountVectorizer().fit(corpus)
    news_title_sparse = vectorizer.transform(df["NewsTitle"])
    news_sparse = vectorizer.transform(df["News"])
    X = sp.sparse.hstack((news_title_sparse, news_sparse), format='csr')
    return normalize(X)


if __name__ == "__main__":
    df = read_file_csv("TDFFN.csv")
    df = transform_df(df)
    X =  to_features(df, "")
    pca = TruncatedSVD(20, n_iter=10).fit(X)
    print pca.explained_variance_ratio_.sum() * 100
    X = normalize(pca.transform(X))
    # exit()
    # train

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % 'recall')
    Y = df["Value"].values
    clf.fit(X, Y)
    print clf.best_params_
    print
    print "Grid scores on development set:"
    print
    print clf.best_estimator_
    print clf.best_score_