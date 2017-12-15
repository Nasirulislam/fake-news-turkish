import snowballstemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from Utils.PreProcessor import TurkishPreprocessor
from Utils.Vectorizer import TurkishVectorizer

FEATURES_TERM_FREQUENCY = "term frequency"
FEATURES_TFIDF = "tfidf"
FEATURES_NGRAMS = "ngrams"

MODELS_SVM = "svm"
MODELS_RANDOM_FOREST = "random forest"

STEMMER_SNOWBALL = "snowball"
STEMMER_ZEMBREK = "zembrek"

PARAM_PRECISION = "precision"
PARAM_RECALL = "recall"

from urlparse import urlparse


class TurkishFakeNewsClassifier:
    model_name_to_class = {
        MODELS_SVM: SVC(),
        MODELS_RANDOM_FOREST: RandomForestClassifier()
    }

    feature_name_to_class = {
        FEATURES_TERM_FREQUENCY: CountVectorizer(),
        FEATURES_TFIDF: TfidfVectorizer(),
    }

    stemmer_name_to_method = {
        STEMMER_SNOWBALL: snowballstemmer.TurkishStemmer()
    }

    def __init__(self, columns, model=MODELS_SVM, feature=FEATURES_TERM_FREQUENCY, use_pca=True,
                 training_param=PARAM_PRECISION, stemmer_method = STEMMER_SNOWBALL):
        """

        :param columns: Which training columns to use. Default all
        :param model: Which model to use, default PCA
        :param feature: Which feature to apply on texts. Default Term Frequency
        :param use_pca: Whether or not to use PCA. Default true
        :param training_param: Which parameter to train on. Default Precision
        """
        self.columns = columns or []
        self.model = model
        self.feature = feature
        self.use_pca = use_pca
        self.training_param = training_param
        self.stemmer_method = stemmer_method

    def fit(self, X):
        # Extract columns
        col = self.transform_column(X)
        self.pipeline = self.get_pipeline()
        return self.pipeline.fit(col)

    def transform(self, X):
        pass

    def transform_column(self, X):
        if not self.columns:
            # select all columns
            df = X
        else:
            df = X.loc[:, self.columns]

        # transform NewsDate column to a boolean indicating whether the date is present or not
        if "NewsDate" in self.columns:
            values = (df["NewsDate"] != pd.NaT) * 1
            df["NewsDate"] = values

        # transform Url column to just the hostname or "Social" if the news source is social media
        if "Url" in self.columns:
            df["Url"] = df["Url"].apply(lambda url: "Social" if url == "Social Media Virals" else
            urlparse(url).hostname.replace("www.", "").replace(".", "") if "http" in url else url)

        if "Value" in self.columns:
            # transform Value from FAKE / TRUE to True and False depending if the news is true or fake
            df.loc[df["Value"] == "FAKE", "Value"] = False
            df.loc[df["Value"] == "TRUE", "Value"] = True
        # Todo add other features here
        return df

    def get_pipeline(self):
        """
        Return the pipeline used to train the final model. Also return the various associated model parameters for
        cross-validation purposes
        :return: pipeline, parameters
        """
        steps = [
            # first the pre-processor
            ("preprocessor", TurkishPreprocessor(self.stemmer_name_to_method[self.stemmer_method])),
            ("vectorizer", TurkishVectorizer(self.feature_name_to_class[self.feature])),
            # use pca
            ("pca", TruncatedSVD(n_iter=10)),
            ("model", self.model_name_to_class[self.model])
        ]
        return Pipeline(steps)

    def select_best_model(self, df):
        """
        Select the best model by using grid search method.
        :return:
        """
        params = {
            # check whether unigrams give good results or bigrams.
            "vectorizer__vectorizer": [self.feature_name_to_class[self.feature]],
            "vectorizer__ngram_range": [(1,1), (2,2)],
            # check pca parameters
            "pca__n_components": [20, 25, 30, 50],
            # stemmer to use for preprocessing
            "preprocessor__stemmer": [self.stemmer_name_to_method[self.stemmer_method]]

        }
        # select the tunable parameters according to the model
        if self.model == MODELS_SVM:
            params.update({
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': [1e-3, 1e-4],
                'model__C': [1, 10, 100, 1000]
            })
        elif self.model == MODELS_RANDOM_FOREST:
            params.update({

            })
        clf = GridSearchCV(self.get_pipeline(), params, cv=5,
                           scoring='%s_macro' % self.training_param)
        X = df.drop(["Value"], axis=1)
        Y = df["Value"].values
        clf.fit(X, Y)
        print clf.best_params_
        print clf.best_estimator_
        print clf.best_score_



