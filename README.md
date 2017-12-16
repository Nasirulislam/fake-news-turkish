# Turkish Fake News Detection (with UI)

## Requirements
-  python 2.7

## Installation

-  Clone the repository or download as zip file.
-  You may create a new virtual environment.
-  run `pip install -r requirement.txt`. This will install all the dependencies for the project.
-  run `python web-main.py` for the UI server.
-  open `localhost:8080` in your browser.
 
## Training the model

The main class for this project is the `TurkishFakeNewsClassifier` in Training/FakeNews.py file. The class initialization
signature is as follows:

`def __init__(self, columns, model=MODELS_SVM, feature=FEATURES_TERM_FREQUENCY, use_pca=True,
                 training_param=PARAM_PRECISION, stemmer_method = STEMMER_SNOWBALL)`
                 
The parameters are as follows:

-  `columns: List` The list of columns to use for feature extraction. If this parameter is None or an empty list, all
 features are considered for training.
-  `model: One of MODELS_SVM, MODELS_RANDOM_FOREST, MODELS_LOGISTIC_REGRESSION` Specify which classifier to use for training. Default is _MODELS_SVM_ i.e. Support Vector Classifier
- `feature: One of FEATURES_TERM_FREQUENCY, FEATURES_TF_IDF` Either use term frequency or tf-idf for text representation
- `use_pca: Boolean` Whether to use Truncated SVD for dimensionality reduction. Considering LSI/LDA for next phase.
- `training_param: One of PARAM_PRECISION, PARAM_RECALL` Which score to maximize while training the model.
- `stemmer_method: One of STEMMER_SNOWBALL, STEMMER_ZEMBREK` Which stemmer to stem while pre-processing the text. For now, only snowball stemmer is available.

_Note: This document is undergoing continued update_