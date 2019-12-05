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
source_path = str(Path(os.path.abspath('joaoc-evaluation@%')).parent.parent / 'src')
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
import yaml

from pipeline.data_persistence import persist_local, get_local
# -

evaluation = """
evaluation:
    metrics: [recall, precision, f1, missed, size]
    parameters: 
        at%: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        lower_bound: 0
        upper_bound: 300
    groups:
        CD: director1
        CO: director2
        LPN: director3
        LPI: director3
        LC: director3
        CE: director3
    
"""

evaluation = yaml.load(evaluation)

experiment_id = 7136875758

features = get_local({'experiment_id': experiment_id}, 'features').set_index('id_llamado')
labels = get_local({'experiment_id': experiment_id}, 'labels').set_index('id_llamado')

length = 10000
labels = labels.sort_values(by='reception_date', ascending=False)
predictions = labels[:length][[]]
predictions['prediction'] = [random.random() for i in range(length)]
observertions =  labels[:length]


# +
def classifing(df, args):
    
    res = []
    for i in args['at%']:
        
        size = min(args['upper_bound'], max(args['lower_bound'], int(i / 100 * len(df))))
        
        # store (perc, missed complaints, size)
        res.append((i, df[size:].sum(), size,))
    
    return res

def adding_up(df):
    
    res = pd.DataFrame()
    for i, a in df.iterrows():
        
        new = pd.DataFrame(a['target'], columns=['%', 'missed', 'size']).set_index('%')
        
        if len(res) == 0:
            res = new
            
        else:
            res['missed'] = new['missed'] + res['missed']
            res['size'] = new['size'] + res['size']
            
    return res

  
def generate_name(row, metric):
    
    return  '{}@{}@{}'.format(metric, row["%"], row["groups"])


def apply_metric(df, metric):
    
    metrics_func = dict(
        recall = lambda x: (1 - x['missed'] / x['target']),
        precision = lambda x: (x['target'] - x['missed']) / x['size'],
        f1 = lambda x: 1,
        missed = lambda x: x['missed'],
        size = lambda x: x['size']
        # 2 * ((x['target'] - x['missed']) / x['size'] * (1 - x['missed'] / x['target'])) / ((x['target'] - x['missed']) / x['size'] + (1 - x['missed'] / x['target']))
    )
    
    return df.apply(lambda row: {'name': generate_name(row, metric),
                                 'value': metrics_func[metric](row)}, 1).tolist()

def calculate_metrics(df, results, metrics):

    df_temp = df.query("groups != 'overall'").groupby('%').sum()
    df_temp['groups'] = 'overall_groups'
    df = pd.concat([df.reset_index(), df_temp.reset_index()])
    
    total_complaints = results.groupby('groups')['target'].sum()
    soma = total_complaints.sum() / 2
    total_complaints['overall_groups'] = soma
    total_complaints['overall'] = soma
    
    df = df.reset_index().merge(total_complaints.reset_index(), on='groups')
    
    results = []
    
    for metric in metrics:

        results = results + apply_metric(df, metric)

    return results

def apply_daily_evaluation(predictions, observations, evaluation):
    
    results = predictions.merge(observations, right_index=True, left_index=True)\
                          .sort_values(by='prediction', ascending=False)
    
    results['groups'] = results['tipo_procedimiento_codigo'].apply(lambda x: evaluation['groups'][x])
    
    temp = results.copy()
    temp['groups'] = 'overall'
    results = pd.concat([temp, results])
    
    df = results.groupby([results['reception_date'].dt.date, 'groups'])['target']\
        .apply(classifing, args=evaluation['parameters']).to_frame()

    df = df.reset_index().groupby('groups').apply(adding_up)
    
    results = calculate_metrics(df, results, evaluation['metrics'])
    
    return results

def get_params_for_precision(obs, pred, precision_at):
    
    precision, recall, thresh = precision_recall_curve(obs['target'], pred['prediction'])
    df = pd.DataFrame({'precision': precision[:-1], 'thresh': thresh, 'recall': recall[:-1]}) 
    
    res = []
    for prec in precision_at:
        
        if len(df[df['precision'] >= prec]['recall']) > 0:

            res.append({'name': f'precision@{prec}@recall',
                        'value': df[df['precision'] >= prec]['recall'].values[0]})
        else:
            res.append({'name': f'precision@{prec}@recall',
                        'value': 0})
        
        res.append({'name': f'precision@{prec}@%',
                    'value': len(df[df['precision'] >= prec]) / len(df) * 100})
        
    return res

def evaluate(obs, pred, evaluation):

    evaluations = apply_daily_evaluation(pred, obs, evaluation)

    evaluations = evaluations + get_params_for_precision(obs, pred, evaluation['parameters']['precision_at'])

    return evaluations


# -

from copy import deepcopy
df = apply_daily_evaluation(predictions, observertions, evaluation['evaluation'])

df

total_complaints = results.groupby('groups')['target'].sum()
total_complaints['overall_groups'] = total_complaints.sum()
total_complaints['overall'] = total_complaints['overall_groups']

total_complaints

ax = df.query('groups == "overall"').plot(x='%', y='recall', label='overall')
ax = df.query('groups == "director1"').plot(x='%', y='recall', ax=ax, label='director1')
ax = df.query('groups == "director2"').plot(x='%', y='recall', ax=ax, label='director2')
ax = df.query('groups == "director3"').plot(x='%', y='recall', ax=ax, label='director3')
ax = pd.DataFrame({'%': [0,100], 'recall': [0,1]}).plot(x='%', y='recall', ax=ax, label='straight')


# +
def predict_given_at_1(df, args):
    df['pred_target'] = df.index.isin(df.nlargest(int(args[0] / 100 * len(df)), # percentage of list
                                                  columns='prediction').index)
    df['pred_target'] = df['pred_target'].apply(int)
    
    return df

def predict_given_at_2(df, args):
    df['pred_target'] = df.index.isin(df.iloc[:(int(50 / 100 * len(df)))].index)
    df['pred_target'] = df['pred_target'].apply(int)
    
    return df

def predict_given_at_pass(df, args):
    
    return df

def predict_given_multiple(df, args):
    
    for perc in args[0]:
        df[f'pred_target_{perc}'] = df.index.isin(df.iloc[:(int(perc / 100 * len(df)))].index)
#         df[f'pred_target_{perc}'] = df[f'pred_target_{perc}'].apply(int)
    
    return df


# -

# %timeit results.groupby([results['reception_date'].dt.date, 'groups']).apply(predict_given_at_1, args=(50,))

# %timeit results.groupby([results['reception_date'].dt.date, 'groups']).apply(predict_given_at_2, args=(50,))

# %timeit results.groupby([results['reception_date'].dt.date, 'groups']).apply(predict_given_at_pass, args=(50,))

args = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# %timeit results.groupby([results['reception_date'].dt.date, 'groups']).apply(predict_given_multiple, args=(args,))

import numpy as np


def return_numbers(df):
    
    res = []
    for i in range(10):
        size = int(i / 10 * len(df))
        res.append((df[size:].sum(), df[:size].sum() / size,))
    
    return res


def return_numbers_1(df):
    
    size = int(50 / 100 * len(df))
    
    return df[size:].sum(), df[:size].sum() / size


def return_numbers_multiple(df, args):
    
    res = []
    for i in args['%']:
        
        size = min(args['upper_bound'], max(args['lower_bound'], int(i / 100 * len(df))))
        
        # store {perc: (missed complaints, size)}
        res.append({i: (df[size:].sum(), size)})
    
    return res


def return_0(df):
    
    return 0


# %timeit results.groupby([results['reception_date'].dt.date, 'groups'])['target'].apply(return_numbers_1).to_frame()

# +
args = {'%': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
'lower_bound': 1,
'upper_bound': 30}

# %timeit results.groupby([results['reception_date'].dt.date, 'groups'])['target'].apply(return_numbers_multiple, args=args).to_frame()
# -

results.groupby([results['reception_date'].dt.date, 'groups'])['target'].apply(return_numbers_multiple, args={'%': args,
                                                                                                             'lower_bound': 1,
                                                                                                             'upper_bound': 30}).to_frame()

results.groupby([results['reception_date'].dt.date, 'groups'])['target'].apply(return_numbers_multiple, args=(args,)).to_frame()

# %timeit results.groupby([results['reception_date'].dt.date, 'groups'])['target'].apply(return_numbers).to_frame()

# %timeit results.groupby([results['reception_date'].dt.date, 'groups'])['target'].apply(return_0).to_frame()

results.groupby([results['reception_date'].dt.date, 'groups'])['target'].apply(return_0).to_frame()

time = 0.295
4.3 * 8 * 11 * 40 / 60 / 60


def recall(results):

    total_complaints = (results['target'] == 1).sum()

    if total_complaints:

        recall = sum((results['pred_target'] == 1) & (results['target'] == 1)) / total_complaints

    else:

        recall = 1
    
    return recall


def precision(results):

    return sum((results['pred_target'] == 1) & (results['target'] == 1)) / sum(results['pred_target'] == 1)


from sklearn.metrics import precision_recall_curve

import random
import numpy as np

y_true = pd.Series([random.choice([0,1]) for i in range(1000)])
probas_pred = pd.Series([uniform(0,1) for i in range(1000)])






def get_params_for_precision(obs, pred, precision_at):
    
    precision, recall, thresh = precision_recall_curve(obs['target'], pred['prediction'])
    df = pd.DataFrame({'precision': precision[1:], 'thresh': thresh, 'recall': recall[1:]}) 
    
    res = []
    for prec in precision_at:
        
        res.append({'metric': f'precision@{prec}@recall',
             'value': df[df['precision'] >= prec]['recall'].values[0]})
        
        res.append({'metric': f'precision@{prec}@%',
             'value': len(df[df['precision'] >= prec]) / len(df) * 100})
        
    return res    


'asdf'['a']

ax = pd.DataFrame({'x':thresh, 'y':precision[1:]}).plot(x='x', y='y')
pd.DataFrame({'x':thresh, 'y':recall[1:]}).plot(x='x', y='y', ax=ax)

pd.DataFrame({'x':recall[1:], 'y':precision[1:]}).plot(x='x', y='y')


