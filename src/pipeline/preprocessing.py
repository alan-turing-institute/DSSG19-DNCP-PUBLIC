from sklearn import preprocessing
from pipeline.data_persistence import persist_local
import copy
from collections import defaultdict
import pandas as pd
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)


def split_by_type(data, numeric_types=[
                  'int64', 'float64'], object_types=['object']):

    return {'numeric': data.select_dtypes(numeric_types),
            'object': data.select_dtypes(object_types),
            'other': data.select_dtypes(exclude=numeric_types + object_types)
            }


def new_columns_names(df, columns):

    rep = dict(zip([f'x{i}' for i in range(len(df.columns))], df.columns))

    for k, v in rep.items():

        columns = [i.replace(k, v) for i in columns]

    return columns


def apply_preprocessor(data, preprocessor, fit, encoder=None):

    if preprocessor == 'StandardScaler':

        if fit:
            encoder = getattr(preprocessing, preprocessor)()
            encoder.fit(data)

        encoded_data = pd.DataFrame(
            encoder.transform(data),
            columns=data.columns,
            index=data.index)

    elif preprocessor == 'OneHotEncoder':

        if fit:
            encoder = getattr(
                preprocessing,
                preprocessor)(
                handle_unknown='ignore')
            encoder.fit(data)

        encoded_data = pd.DataFrame(encoder.transform(data).toarray(),
                                    columns=new_columns_names(
                                        data, encoder.get_feature_names()),
                                    index=data.index)

    else:
        raise 'Error'

    return encoded_data, encoder


def run(preprocessors, data, preprocessing=defaultdict(lambda: None), fit=True):
    """Applies preprocessing to data. It currently suppoerts StandardScaler and
    OneHotEncoding

    Parameters
    ----------
    preprocessors : list
        preprocessors to be applied
    data : pd.DataFrame
        data to be preprocessed
    preprocessing : dict, optional
        encoders of each preprocessor, by default defaultdict(lambda: None)
    fit : bool, optional
        if False, it applies to current encoder, by default True

    Returns
    -------
    pd.DataFrame dict
        preprocessed data and preprocessors used
    """

    scaler_to_data_type = {
        'StandardScaler': 'numeric',
        'OneHotEncoder': 'object'}

    if len(preprocessors) == 0:
        return data, preprocessing

    preprocessor = preprocessors[0]
    data_type = scaler_to_data_type[preprocessor]

    splited_data = split_by_type(data)

    splited_data[data_type], preprocessing[preprocessor] = \
        apply_preprocessor(splited_data[data_type],
                           preprocessor,
                           fit=fit,
                           encoder=preprocessing[preprocessor])

    processed_data = pd.concat(splited_data.values(), axis=1)

    return run(preprocessors[1:], processed_data, preprocessing, fit)


def apply_preprocessing(approach, original_train_dict,
                        original_test_dict, args):
    """Generic preprocessing implementation.

    It currently supports StandardScaler and OneHotEncoder

    Parameters
    ----------
    approach : dict
        approach variables
    original_train_dict : dict
        contains a dataframe with features and labels
    original_test_dict : dict
        contains a dataframe with features and labels
    args : dict
        generic variables of the pipeline

    Returns
    -------
    dict dict
        modified train and test dict
    """

    train_dict, test_dict = copy.deepcopy(
        original_train_dict), copy.deepcopy(original_test_dict)

    if 'preprocessors' in approach:

        train_dict['features'], preprocessing = run(
            approach['preprocessors'], train_dict['features'])

        test_dict['features'], preprocessing = run(
            approach['preprocessors'], test_dict['features'], preprocessing, fit=False)

    persist_local(preprocessing, args, 'preprocessing', [
                  'experiment_id', 'approach_id', 'fold_name'], '.dill')

    return train_dict, test_dict
