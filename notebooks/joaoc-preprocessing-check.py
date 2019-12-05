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
source_path = str(Path(os.path.abspath('joaoc-preprocessing-check')).parent.parent / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)


# Packages
import pandas as pd
pd.options.display.max_columns = 999

import warnings
warnings.filterwarnings('ignore')

import yaml
# -

import pipeline.preprocessing as pr
import pipeline.pipeline as pi
import pipeline.data_persistence as dp
import pipeline.model_data_prep as mp
import utils.utils as utils

from sklearn.preprocessing import StandardScaler, OneHotEncoder
scaler = StandardScaler()

from sklearn import preprocessing

args = dict(experiment_id=3113067571)

folds = mp.create_folds(args)

features = dp.get_local(args, 'features').set_index('id_llamado')
labels = dp.get_local(args, 'labels').set_index('id_llamado')[['target']]

approaches = dp.get_approaches(args['experiment_id'])

approach = approaches[0]


# +
def split_by_type(data, numeric_types=['int64','float64'], object_types=['object']):
    
    return {'numeric': data.select_dtypes(numeric_types),
            'object': data.select_dtypes(object_types),
            'other': data.select_dtypes(exclude=numeric_types+object_types)
           }

def apply_preprocessor(data, preprocessor, fit, encoder=None):
    
    if fit:
        encoder = getattr(preprocessing, preprocessor)()
        encoder.fit(data)
    
    encoded_data = pd.DataFrame(encoder.transform(data), columns=data.columns, index=data.index)
    
    return encoded_data, encoder


def run(approach, data, preprocessing=defaultdict(lambda: None), fit=True):
    
    splited_data = split_by_type(data)
    
    for preprocessor in approach['preprocessors']:
        
    
        for data_type in splited_data.keys():
            
            try:
    
                splited_data[data_type], preprocessing[preprocessor] = \
                    apply_preprocessor(splited_data[data_type],
                                          preprocessor,
                                          fit=fit,
                                          encoder=preprocessing[preprocessor])
            
            except Exception as e:
                print(preprocessor, data_type, e)
                continue
    
    processed_data = pd.concat(splited_data.values(), axis=1)
    
    return processed_data, preprocessing


def apply_preprocessing(approach, train_dict, test_dict):
    
    if 'preprocessing' not in approach:
        approach['preprocessing'] = []
    
    print('train')
    train_data, preprocessing = run(approach, copy.deepcopy(train_dict['features']))
    
    print('test')
    test_data, preprocessing = run(approach,  copy.deepcopy(test_dict['features']), preprocessing, fit=False)
    
#     persist_local(....)
    
    return {'features': train_data, 'labels': train_dict['labels']}, {'features': test_data, 'labels': test_dict['labels']}


# +
train_dict, test_dict = pi.generate_folds_matrices(features, labels, folds[0])

approach['preprocessors'] = ['OrdinalEncoder', 'StandardScaler']

train, test = apply_preprocessing(approach, train_dict, test_dict)
# -

test['features']

len(train['features']) == len(train['labels'])

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier

clf = AdaBoostClassifier()

train['labels'].info()

test['labels']

len(train['features'])

train_dict['features']

clf.fit(train['features'].fillna(train['features'].mean()), train['labels'])


