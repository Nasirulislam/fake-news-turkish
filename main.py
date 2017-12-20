# coding=utf-8
"""
Fake news detector for Turkish language
"""
from sklearn.feature_selection import SelectKBest

from Training.FakeNews import *


def read_file_xlsx(file_path):
    # check that the file is indeed an excel file
    assert file_path[-5:] == ".xlsx"
    df = pd.read_excel(file_path, index_col=0, encoding="utf-8")
    return df

def read_file_csv(file_path):
    return pd.read_csv(file_path, index_col=0, encoding="utf-8")


def get_baseline_svm(**kwargs):
    classifier = TurkishFakeNewsClassifier(columns=["NewsDate", "Url", "NewsTitle", "News", "Value"],
                                           model=MODELS_SVM, feature=FEATURES_TERM_FREQUENCY)
    pipeline_params = {
                           "model__C": 10,
                           'model__kernel': 'linear',
                           'adder__n_components': 30,
                           'adder__n_iter': 10,
                       }
    pipeline_params.update(kwargs)
    classifier.train(df, pipeline_params=pipeline_params)
    return classifier


def get_baseline_nb(**kwargs):
    classifier = TurkishFakeNewsClassifier(columns=["NewsDate", "Url", "NewsTitle", "News", "Value"],
                                           model=MODELS_NAIVE_BAYES, feature=FEATURES_TERM_FREQUENCY)
    pipeline_params={
                         "model__alpha": 1,
                         'model__fit_prior': False,
                         'adder__n_components': 30,
                         'adder__n_iter': 10
    }
    pipeline_params.update(kwargs)
    classifier.train(df, pipeline_params=pipeline_params)
    return classifier


def get_all_svm():
    classifier = get_baseline_svm(**{# 'vectorizer__sublinear_tf': True,
                           # 'vectorizer__smooth_idf': True,
                           # 'vectorizer__ngram_range' : (1,2),
                           "extractor__slang": True,
                           'extractor__suffixes': True,
                           'extractor__sentences_count': True,
                           'extractor__punctuations': True
                           })
    return classifier


def get_all_nb():
    classifier = get_baseline_nb(**{# 'vectorizer__sublinear_tf': True,
                           # 'vectorizer__smooth_idf': True,
                           # 'vectorizer__ngram_range' : (1,2),
                           "extractor__slang": True,
                           'extractor__suffixes': True,
                           'extractor__sentences_count': True,
                           'extractor__punctuations': True
                           })
    return classifier

if __name__ == "__main__":
    df = read_file_csv("TDFFN.csv")

    # import random
    # random.seed(10)

    classifier = get_baseline_svm()
    classifier.plot_precision_recall(True, "pr_rc_plot.png")
    classifier.plot_roc(True, "roc_svm.png")

    # precision, recall and f1 score for baseline
    p_b, r_b, f_b = classifier.get_precision_recall_f1()
    print p_b, r_b, f_b

    # Naive Bayes

    classifier = get_baseline_nb()
    classifier.plot_precision_recall(True, file_name="pr_rc_nb.png")
    classifier.plot_roc(True, "roc_nb.png")

    # precision, recall and f1 score for baseline
    p_b, r_b, f_b = classifier.get_precision_recall_f1()
    print p_b, r_b, f_b


    classifier = get_all_svm()
    classifier.plot_precision_recall(True, file_name="pr_rc_svm_all.png")
    classifier.plot_roc(True, "roc_svm_all.png")

    # precision, recall and f1 score for baseline
    p_b, r_b, f_b = classifier.get_precision_recall_f1()
    print p_b, r_b, f_b


    classifier = get_all_nb()
    classifier.plot_precision_recall(True, file_name="pr_rc_nb_all.png")
    classifier.plot_roc(True, "roc_nb_all.png")

    # precision, recall and f1 score for baseline
    p_b, r_b, f_b = classifier.get_precision_recall_f1()
    print p_b, r_b, f_b