# coding=utf-8
"""
Fake news detector for Turkish language
"""

from Training.FakeNews import *


def read_file_xlsx(file_path):
    # check that the file is indeed an excel file
    assert file_path[-5:] == ".xlsx"
    df = pd.read_excel(file_path, index_col=0, encoding="utf-8")
    return df

def read_file_csv(file_path):
    return pd.read_csv(file_path, index_col=0, encoding="utf-8")


if __name__ == "__main__":
    df = read_file_csv("TDFFN.csv")

    classifier = TurkishFakeNewsClassifier(columns=["NewsDate", "Url", "NewsTitle", "News", "Value"],
                                           model=MODELS_SVM)
    print classifier.train(df, pipeline_params={"model__C": 1, 'model__kernel': 'linear', 'model__gamma': 0.001})
    classifier.plot_precision_recall(True)
