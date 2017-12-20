from sklearn.linear_model import LogisticRegression
import snowballstemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, \
    roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from Utils.Extractor import TurkishFeatureExtractor
from Utils.FeatureAdder import TurkishFeatureAdder
from Utils.PreProcessor import TurkishPreprocessor
from Utils.Vectorizer import TurkishVectorizer
import matplotlib.pyplot as plt

FEATURES_TERM_FREQUENCY = "term frequency"
FEATURES_TFIDF = "tfidf"
FEATURES_NGRAMS = "ngrams"

MODELS_SVM = "svm"
MODELS_RANDOM_FOREST = "random forest"
MODELS_LOGISTIC_REGRESSION = "logistic regression"
MODELS_NAIVE_BAYES = "Naive Bayes"

STEMMER_SNOWBALL = "snowball"
STEMMER_ZEMBREK = "zembrek"

PARAM_PRECISION = "precision"
PARAM_RECALL = "recall"

from urlparse import urlparse


class TurkishFakeNewsClassifier:
    model_name_to_class = {
        MODELS_SVM: SVC(),
        MODELS_RANDOM_FOREST: RandomForestClassifier(),
        MODELS_LOGISTIC_REGRESSION: LogisticRegression(),
        MODELS_NAIVE_BAYES: MultinomialNB()
    }

    feature_name_to_class = {
        FEATURES_TERM_FREQUENCY: CountVectorizer(),
        FEATURES_TFIDF: TfidfVectorizer(),
    }

    stemmer_name_to_method = {
        STEMMER_SNOWBALL: snowballstemmer.TurkishStemmer()
    }

    def __init__(self, columns, model=MODELS_SVM, feature=FEATURES_TERM_FREQUENCY, use_pca=True,
                 training_param=PARAM_PRECISION, stemmer_method = STEMMER_SNOWBALL, random_state=0):
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
        self.random_state = random_state

    def get_test_train_split(self, df, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(["Value"], axis=1),
            df["Value"],
            test_size=test_size,
            random_state=1,
            stratify=df["Value"])
        return X_train, X_test, y_train, y_test

    def plot_or_save(self, plt, save_image, file_name):
        if not save_image:
            plt.show()
        else:
            # save this plot
            plt.savefig("static/plots/%s" % file_name)

    def get_precision_recall_f1(self):
        assert hasattr(self, "y_test") and hasattr(self, "y_score"), "Call this function only after train has been called"
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        return precision, recall, f1

    def plot_precision_recall(self, save_img=False, file_name="pr_rc_plot.png"):
        plt.figure()
        assert hasattr(self, "y_test") and hasattr(self, "y_score"), "Call this function only after train has been called"
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_score)

        plt.step(recall, precision, color='b' , # alpha=0.2,
                 where='post')
        # plt.fill_between(recall, precision, step='post', alpha=0.2,
        #                  color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        average_precision = average_precision_score(self.y_test, self.y_score)
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        self.plot_or_save(plt, save_img, file_name)

    def plot_roc(self, save_img=False, file_name="roc.png"):
        plt.figure()
        assert hasattr(self, "y_test") and hasattr(self, "y_score"), "Call this function only after train has been called"

        fpr, tpr, _ = roc_curve(self.y_test, self.y_score,
                                      pos_label=1)
        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        self.plot_or_save(plt, save_img, file_name)


    def train(self, X, test_size=0.3, pipeline_params=None, threshold=0.5):
        """
        Automatically split training data and test data and present the results in the form of a dictionary.
        :param test_size: How much data to use for train and test. 0.3 default means that 30% used for test, 70% for train.
        :param pipeline_params:
        :param folds: number of folds in k fold cross validation
        :param X:
        :return: this trains the pipeline
        """
        data = self.transform_column(X)
        pipeline_params = pipeline_params or {}
        assert isinstance(pipeline_params, dict)
        X_train, X_test, y_train, y_test = self.get_test_train_split(data, test_size)
        self.get_pipeline().set_params(**pipeline_params)
        self.pipeline.fit(X_train, y_train.as_matrix())
        # now calculate scores.
        accuracy = self.pipeline.score(X_test, y_test)
        print "Accuracy %f" % accuracy
        if hasattr(self.pipeline, "decision_function") and callable(self.pipeline.decision_function):
            y_scores = self.pipeline.decision_function(X_test)
        else:
            y_scores = self.pipeline.predict_proba(X_test)[:, 1]
        self.y_test = y_test
        self.y_score = y_scores
        self.y_pred = self.pipeline.predict(X_test)
        # self.y_pred = self.pipeline.predict_proba(X_test)
        # self.y_pred = self.y_pred[:, 1] > threshold

    def fit(self, X):
        # Extract columns
        col = self.transform_column(X)
        return self.get_pipeline().fit(col)

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
            df.loc[df["Value"] == "FAKE", "Value"] = 1
            df.loc[df["Value"] == "TRUE", "Value"] = 0
        df["Value"] = df["Value"].astype('int')
        # Todo add other features here
        return df

    def get_pipeline(self):
        """
        Return the pipeline used to train the final model. Also return the various associated model parameters for
        cross-validation purposes
        :return: pipeline, parameters
        """
        if hasattr(self, "pipeline"):
            return self.pipeline
        steps = [
            # before preprocessor, comes the feature extractor
            ('extractor', TurkishFeatureExtractor()),
            # first the pre-processor
            ("preprocessor", TurkishPreprocessor(self.stemmer_name_to_method[self.stemmer_method])),
            ("vectorizer", TurkishVectorizer(self.feature_name_to_class[self.feature])),
            # use pca
            # ("pca", TruncatedSVD(n_components=20, n_iter=10)),
            ("adder", TurkishFeatureAdder(n_components=20, n_iter=10)),
            ("model", self.model_name_to_class[self.model])
        ]
        self.pipeline = Pipeline(steps)
        return self.pipeline

    def select_best_model(self, df):
        """
        Select the best model by using grid search method.
        :return:
        """
        params = {
            # check whether unigrams give good results or bigrams.
            "vectorizer__vectorizer": [self.feature_name_to_class[self.feature]],
            "vectorizer__ngram_range": [(1,1)],
            # check pca parameters
            "pca__n_components": [30, 40, 50],
            # stemmer to use for preprocessing
            "preprocessor__stemmer": [self.stemmer_name_to_method[self.stemmer_method]]

        }
        # select the tunable parameters according to the model
        if self.model == MODELS_SVM:
            params.update({
                'model__kernel': ['linear'],
                'model__gamma': [1e-3, 1e-4],
                'model__C': [0.5, 1, 10]
            })
        elif self.model == MODELS_RANDOM_FOREST:
            params.update({
                'model__n_estimators': [5, 10, 15]
            })
        elif self.model == MODELS_LOGISTIC_REGRESSION:
            params.update({
                'model__C': [1.0, 10],
                'model__tol': [0.001, 0.01, 0.1]
            })
        clf = GridSearchCV(self.get_pipeline(), params, cv=5,
                           scoring='%s_macro' % self.training_param)
        X = df.drop(["Value"], axis=1)
        Y = df["Value"].values
        clf.fit(X, Y)
        print clf.best_params_
        # print clf.best_estimator_
        print clf.best_score_



