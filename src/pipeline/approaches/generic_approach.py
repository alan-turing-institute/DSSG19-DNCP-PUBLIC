from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

# Imports
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd


def fit(args, train_dict):
    """
    Parameters
    ----------
    args: dictionary
        Approach configuration coming from the experiment yaml

    train_dict: dictionary
        Dictionary containing the train features DataFrame and the train labels DataFrame.


    Return
    ------
    Dictionary
        'chosen_model': Model fitted using the train data.
        'hotencoding': OneHotEncoder
        'standardscaler': StandardScaler
    """
    model = {}


    model_list = {'random_forest': RandomForestClassifier(),
        'extra_trees': ExtraTreesClassifier(),
        'ada_boost': AdaBoostClassifier(),
        'logistic_regression': LogisticRegression(),
        'svm': svm.SVC(),
        'gradient_boosting': GradientBoostingClassifier(),
        'gaussian_nb': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(),
        'sgd': SGDClassifier(),
        'knn': KNeighborsClassifier(),
        'xgb': XGBClassifier(),
        'lgbm': LGBMClassifier()
            }

    chosen_model = model_list[args['approach_name']] # initiate the corresponding model
    p = args['hyperparameters']
    chosen_model.set_params(**p) # set hyperparameters; if empty default is used.
    model['chosen_model'] = chosen_model.fit(
                    X = train_dict['features'],
                    y = train_dict['labels']['target'])

    return model

def predict(model, test_features):
    """
    predict test set.

    Parameters
    ----------
    model: Dictionary from fit function
        'chosen_model': Model fitted using the train data.
        'hotencoding': OneHotEncoder
        'standardscaler': StandardScaler
    test_features: DataFrame
        Pandas DataFrame containing features to be used for predicting.

    Return
    ------
    List
        DataFrame of predictions (0s or 1s).
    """

    # Predicting
    prediction_index = list(test_features.index.values)
    prediction_values = model['chosen_model'].predict_proba(test_features)[:, 1]
    # print(prediction_values[:10])

    predictions = pd.DataFrame(data = {'prediction': prediction_values,
                                      'id_llamado' : prediction_index})
    predictions = predictions.set_index('id_llamado')

    return predictions
