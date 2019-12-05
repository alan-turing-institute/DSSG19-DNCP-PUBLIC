# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath('joaoc-feature-importance')).parent.parent / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)


# Packages
import pandas as pd
pd.options.display.max_columns = 999

import warnings
warnings.filterwarnings('ignore')
# -

from model_selection import plotting, select_model

# +
from sklearn import preprocessing 
import pandas as pd
from collections import defaultdict
import copy

from pipeline.data_persistence import persist_local
# -

encoder = getattr(preprocessing, 'OneHotEncoder')(handle_unknown='ignore')

df = pd.DataFrame(data={'a': ['a', 'b', 'c', 'a'], 'b': ['a', 'b', 'c', 'a'] })
df1 = pd.DataFrame(data={'a': ['a', 'b', 'c', 'z'], 'b': ['a', 'b', 'c', 'a'] })

encoder.fit(df)
res = encoder.transform(df)

old_columns = encoder.get_feature_names()



new_columns_names(df, old_columns)


# +
def split_by_type(data, numeric_types=['int64','float64'], object_types=['object']):
    
    return {'numeric': data.select_dtypes(numeric_types),
            'object': data.select_dtypes(object_types),
            'other': data.select_dtypes(exclude=numeric_types+object_types)
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
    
        encoded_data = pd.DataFrame(encoder.transform(data), columns=data.columns, index=data.index)
        
        
    elif preprocessor == 'OneHotEncoder':
        
        if fit:
            encoder = getattr(preprocessing, preprocessor)(handle_unknown='ignore')
            encoder.fit(data)
            
        encoded_data = pd.DataFrame(encoder.transform(data).toarray(), 
                                    columns=new_columns_names(data, encoder.get_feature_names()), 
                                    index=data.index)
    
    else:
        raise 'Error'
    
    return encoded_data, encoder


def run(approach, data, preprocessing=defaultdict(lambda: None), fit=True):

    scaler_to_data_type = {'StandardScaler': 'numeric', 'OneHotEncoder': 'object'}

    if len(approach) == 0:
        return data, preprocessing
    
    preprocessor = approach[0]
    data_type = scaler_to_data_type[preprocessor]
    
    splited_data = split_by_type(data)
    
    splited_data[data_type], preprocessing[preprocessor] = \
        apply_preprocessor(splited_data[data_type],
                            preprocessor,
                            fit=fit,
                            encoder=preprocessing[preprocessor])


    processed_data = pd.concat(splited_data.values(), axis=1)
    
    return run(approach[1:], processed_data, preprocessing, fit)

def apply_preprocessing(approach, original_train_dict, original_test_dict, args):
    

    train_dict, test_dict = copy.deepcopy(original_train_dict), copy.deepcopy(original_test_dict)

    if 'preprocessors' in approach:

        train_dict['features'], preprocessing = run(approach['preprocessors'], train_dict['features'])

        test_dict['features'], preprocessing = run(approach['preprocessors'], test_dict['features'], preprocessing, fit=False)
    
    persist_local(preprocessing, args, 'preprocessing', ['experiment_id', 'approach_id', 'fold'], '.dill')
    
    return train_dict, test_dict
# -

dill.load(open('/data/persist/preprocessing/5492726176_101282_2013-01-01.dill', 'rb'))

from pipeline.pipeline import generate_folds_matrices
from pipeline.data_persistence import get_local, get_approaches

# +
experiment_id = 1127901537
args = dict(experiment_id=experiment_id, approach_id='1234', fold=123)
features = get_local({'experiment_id': 1127901537}, 'features').set_index('id_llamado')
labels = get_local({'experiment_id': 1127901537}, 'labels').set_index('id_llamado')

approaches = get_approaches(args['experiment_id'])
folds = [{'train': list(features.index[:1000]), 'test': list(features.index[1000:1200])}]

features['test'] = (features['log_amt_planned_real_guarani'] > 6).apply(str)
original_train_dict, original_test_dict = generate_folds_matrices(features, labels, folds[0])
# -

approach = {'preprocessors': ['OneHotEncoder']}

args

import dill

dill

train_dict, test_dict = apply_preprocessing(approach, original_train_dict, original_test_dict, args)

test_dict['features'].columns




