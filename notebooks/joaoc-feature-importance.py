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

import eli5
import pickle
import glob
import tqdm

glob.glob('/data/persist/models/1127901537_*.p')

# +
path =  '/data/persist/models/1127901537_210336_5574798346.p'
model = pickle.load(open(path, 'rb'))

def get_feature_importance(model):
    
    feature_importances = eli5.explain_weights_df(model['chosen_model'])
    
    if feature_importances is None:
        return list()
    
    else:     
        return feature_importances.to_dict(orient='records')
        

get_feature_importance(model)
# -
model['chosen_model'].predict_proba()


from pipeline.data_persistence import *

features = get_local({'experiment_id': 1127901537}, 'features').set_index('id_llamado')

labels = get_local({'experiment_id': 1127901537}, 'labels').set_index('id_llamado')['target']

prediction_values = model['chosen_model'].predict_proba(features)

predictions = pd.DataFrame(data = {'prediction': pd.DataFrame(prediction_values)[0],
                                  'id_llamado' : features.index.values})
predictions = predictions.set_index('id_llamado')

predictions

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve


eval_metrics = {'metrics': ['recall', 'f1', 'precision'],
                'from': 0.5,
                'to': 0.5,
                'steps': 1,
                'scaler': 10
               }


def evaluate_at(obs, pred, eval_metrics_list, thresh):
    '''
    Evaluate goodness of fit.

    Parameters
    ----------
    obs: List or array
        Real observations.
    pred: List or array
        Predicted values.
    eval_metrics_list: List
        List of wanted evaluation metrics to calculate.

    Return
    ------
    Dictionary
        Dictionary containing all evaluation metrics and scores.
    '''
    eval_dict = {}
    
    obs = list(obs)
    pred = list(pred)
    
    for metric in eval_metrics_list:
        if metric == 'accuracy':
            accuracy = accuracy_score(obs, pred)
            eval_dict[f'accuracy@{thresh}'] = accuracy
        
        elif metric == 'recall':
            recall = recall_score(obs, pred)
            eval_dict[f'recall@{thresh}'] = recall
        
        elif metric == 'precision':
            precision = precision_score(obs, pred)
            eval_dict[f'precision@{thresh}'] = precision

        elif metric == 'f1':
            f1 = f1_score(obs, pred)
            eval_dict[f'f1@{thresh}'] = f1
        
        else:
            logging.warning(f'Metric {metric} does not exists')
            continue


    return eval_dict


def evaluate(obs, predictions, eval_metrics):
    
    evaluations = {}
   
    for thresh in get_range(eval_metrics):
        
        thresh = thresh / eval_metrics['scaler']
        
        print(thresh)
        
        pred = (predictions['prediction'] > thresh).apply(int)
        
        evaluations.update(evaluate_at(obs, pred, eval_metrics['metrics'], thresh))
        
    return evaluations
    


def get_range(eval_metrics):
    to = int(eval_metrics['to'] * eval_metrics['scaler']) + 1
    fro = int(eval_metrics['from'] * eval_metrics['scaler']) 
    step_size = int((eval_metrics['to'] - eval_metrics['from']) / eval_metrics['steps'] * eval_metrics['scaler']) if eval_metrics['steps'] > 1 else 1
    print(to, fro, step_size)
    return range(fro, to, step_size)


evaluate(labels, predictions, eval_metrics)

list(range(eval_metrics['from'], eval_metrics['to'], eval_metrics['steps']))


precision, recall, _ = precision_recall_curve(labels, predictions)


pd.DataFrame(data={'pre': precision, 're': recall}).plot(x='re', y='pre')


