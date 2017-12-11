# coding=utf-8
"""
Fake news detector for Turkish language
"""
from urlparse import urlparse

import pandas as pd
import sklearn
import pytest
import snowballstemmer
from nltk.tokenize import RegexpTokenizer

def read_file_xlsx(file_path):
    # check that the file is indeed an excel file
    assert file_path[-5:] == ".xlsx"
    df = pd.read_excel(file_path, index_col=0)
    return df


def stem_turkish_text(stemmer, text):
    text = text.lower()
    tokenized = RegexpTokenizer(r"\w+").tokenize(text)
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
    values = df["NewsDate"] != pd.NaT
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

if __name__ == "__main__":
    df = read_file_xlsx("TDFFN.xlsx")
    df = transform_df(df)
    print df.head(10)
