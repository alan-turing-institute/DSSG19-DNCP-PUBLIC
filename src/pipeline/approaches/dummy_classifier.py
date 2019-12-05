'''
Dummy classifier.

sdfsd
'''
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath('dummy_classifier.py')).parent.parent.parent)
if source_path not in sys.path:
    sys.path.insert(0, source_path)

import numpy as np
import pandas as pd


def fit(approach, train_dict):
    """
    Given a train data (features and labels), fit a dummy classifier.
    Since we can actually not fit a model, it returns the probability of complaint.

    Parameters
    ----------
    approach: dictionary
        Approach configuration coming from the experiment yaml
    train_dict: dictionary
        Dictionary containing the train features DataFrame and the train labels DataFrame.

    Return
    ------
    Class
        Model fitted using the train data.
    """

    model = approach['hyperparameters']['perc_complaint']
    return model

def predict(model, test_features):
    """
    Given a fitted model, predict on some test features.

    Parameters
    ----------
    model: Class
        Fitted model. In this case, dummy classifier.
    test_features: DataFrame
        Pandas DataFrame containing features to be used for predicting.

    Return
    ------
    List
        List of predictions (0s or 1s).
    """

    test_features['prediction'] = [int(prediction)
        for prediction in np.random.randint(0, 100, len(test_features)) < model]
    
    return test_features
