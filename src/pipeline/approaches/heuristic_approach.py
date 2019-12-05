from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

import numpy as np
import pandas as pd
from sklearn import preprocessing


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

    model = {'chosen_model': None}
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

    # Normalize values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = pd.DataFrame(min_max_scaler.fit_transform(np.array(test_features.log_amt_planned_real_guarani).reshape(-1, 1)))

    prediction_index = list(test_features.index.values)
    prediction_values = x_scaled[0].tolist()

    predictions = pd.DataFrame(data = {'prediction': prediction_values,
                                      'id_llamado' : prediction_index})
    predictions = predictions.set_index('id_llamado')

    return predictions
