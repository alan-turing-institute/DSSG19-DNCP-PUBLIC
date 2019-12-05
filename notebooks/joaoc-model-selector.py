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

# ## selector

# +
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath('joaoc-model-selector')).parent.parent/ 'src')
pipeline_path = str(Path(os.path.abspath('joaoc-model-selector')).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)
# -

import yaml

selector_config = """

experiment_id: 3124936745


# List of rules to filter the table experiments.evaluations.
#
# One rule orders a list of learners according to an statistic that is 
# applied to all folds of a given experiment. Then, it orders it given
# the higher_is_better paramenter. Finally, the top is used to cut the list.
#
# Parameters:
# ----------
# metric: string
#       It accepts any implemented metric, i.e., precision, recall, acurracy.
#       See pipeline/evaluation.py.
#
# top: integer
#       Any integer higher then 1. This parameter sets the size of the cut
#
# statistic: string
#       Any statistic accepted by pd.Groupby.agg function.
#       Some examples are: mean, median, std, sum, min, max, skew, kurt, sem, var
#
# higher_is_better: boolean
#       Teels the algorithm how to order the list given the statistic of the metric.
#       If the metric is good with high values, then, this parameter should be true

rules:
    - 
        metric: 'recall'
        filter: 
            type: threshold
            value: 0.5
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall'
        filter: 
            type: threshold
            value: 0.3
        statistic: min
        higher_is_better: true
    -
        metric: 'precision'
        filter: 
            type: top
            value: 5
        statistic: mean
        higher_is_better: true
        
# Selector is a set of rules to be applied to an experiment result.         
# One can build multiple selectors with different rules. 
# 
# To build a selector, you need to declare the order that you want the rules
# above to be applied. For instance, a selector can use the rules in the following order:
#  1 -> 3 -> 2. Or it can be as simple as one rule: 1
#
# This chains are defined by a list, [1, 3, 2] or [1], that has to be added to the order key
# for each selector.

selectors:
    -
        order: [1, 2, 3]
    -
        order: [1]
"""

# +
experiment_id = 3124936745

selector_config = yaml.load(selector_config)

selector_config['experiment_id'] = experiment_id

data = fetch_data(selector_config)

# +
import fire
import logging
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# from model_selection.plotting import plot_metric_by_fold
from pipeline.data_persistence import generate_id, persist_local
from utils.utils import connect_to_database, open_yaml

logging.basicConfig(level=logging.ERROR)

def apply_statistics(rule, data):
    
    return data[data['eval_metric'] == rule["metric"]]\
            .groupby('learner_id')\
            .agg(rule['statistic'])

def filter_by(rule, data):
    
    if rule['filter']['type'] == 'threshold':
        if rule['higher_is_better']:
            return data[data['score'] >= rule['filter']['value']]
        else:
            return data[data['score'] <= rule['filter']['value']]
        
    elif rule['filter']['type'] == 'top':
        data = data.sort_values(by='score', ascending=not rule['higher_is_better'])
        return data.iloc[:rule['filter']['value']]
        

def apply_selection(rules, filtered_ids, data):
    
    if len(rules) == 0:
        return filtered_ids, data
    
    # apply stats
    stats = apply_statistics(rules[0], data)
    
    # filter
    stats = filter_by(rules[0], stats)
    
    # rename
    column_name = '_'.join([rules[0]['statistic'], rules[0]['metric']])
    stats = stats.rename(columns={'score': column_name})
    
    # append stats
    if filtered_ids is None:  
        filtered_ids = stats
    else:
        filtered_ids = filtered_ids.merge(stats[column_name], right_index=True, left_index=True, how='right')
        
        
    # filter data
    data = data[data['learner_id'].isin(filtered_ids.index)]
        
    
    return apply_selection(rules[1:], filtered_ids, data)


def generate_rule_name(rule):
    
    return '_'.join(map(str, [rule['statistic'], rule["metric"], 
                              rule['filter']['type'], rule['filter']['value']]))


def run_selectors(selector_config, data):
    
    for selector in selector_config['selectors']:
        
        selector['rules'] = [selector_config['rules'][i-1] for i in selector['order']]
        selector['name'] = '->'.join(map(generate_rule_name, selector['rules']))

        learner_ids, data_filtered = apply_selection(selector['rules'], None, data)
        
        selector['learner_stats'] = learner_ids.set_index(['experiment_id', 'approach_id'], append=True)
        selector['data'] = data_filtered
            
    return selector_config


def fetch_data(selector_config):
    
    con = connect_to_database()

    query = f"""
    select evaluation.*, approach.name
    from (select *
    from experiments.evaluations
    where experiment_id = {selector_config['experiment_id']}) evaluation
    left join experiments.approaches approach
    on evaluation.experiment_id = approach.experiment_id
    and evaluation.approach_id = approach.approach_id
    """

    data = pd.read_sql_query(query, con)[['learner_id', 'fold', 'name',
                                         'eval_metric', 'score', 'experiment_id', 'approach_id']]

    return data



# -

selector_results = run_selectors(selector_config, data)


a = selector_results['selectors'][0]['data']
a

a.set_index()

# ### Creating Plot

# # +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath('joaoc-model-selector.py')).parent.parent / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)
# sys.path.insert(0, '../utils')

import model_selection.select_model as sm

# !pwd

res = sm.run(3124936745, '/home/joao.carabetta/Documents/dncp/model_selectors/template.yaml')

# +
import pickle
# Packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns
# from pipeline.data_persistence import persist_local
from itertools import cycle

color = cycle(cm.get_cmap('tab20', 20).colors)

from pipeline.pipeline import generate_folds_matrices
from pipeline.data_persistence import get_local


# -

def fetch_data(experiment_id):
    
    con = utils.utils.connect_to_database()

    query = f"""
    select *
    from experiments.evaluations
    where experiment_id = {experiment_id}
    """

    return pd.read_sql_query(query, con)[['learner_id', 'fold', 'eval_metric', 'score', 'experiment_id']]


res = pickle.load(open('/data/persist/selector_results/3124936745_1117241490.p', 'rb'))

# +
args = dict(experiment_id='7657456883')

9805273013 # w/ preprocessing
6370062941 # wo/ preprocessing
# -

overall_performance_per_fold(args)


# +
def overall_performance_per_fold(args):
    
    title = f'Experiment : {args["experiment_id"]}'
    
    def complaints_per_fold(args, data):
    
        folds = pickle.load(open(f'/data/persist/folds/{args["experiment_id"]}.p', 'rb'))
        labels = get_local(args, 'labels').set_index('id_llamado')[['target']]

        i = 0
        complaints = []
        for fold in folds:

            if fold['name'] in list(data['fold'].unique()):
                complaints.append({
                    'complaints': 100 * labels.loc[fold['test']].sum().values[0] / len(labels.loc[fold['test']]),
                    'fold': fold['name']})

        return pd.DataFrame(complaints)
    
    data = fetch_data(args['experiment_id'])

    eval_metrics = data['eval_metric'].unique()

    nrows = len(eval_metrics)  # Number of rows
    ncols = 1 # Number of columns
    fontsize = 14 # General fontsize
                                 
    grid = plt.GridSpec(nrows * 2 + 2, ncols)

    fig = plt.figure(figsize=(15, nrows * 7))
    
    fig.suptitle(title, fontsize=18, y=0.9)
                                 
    # Percentage of Complaints

    axis = plt.subplot(grid[1, 0])

    complaints = complaints_per_fold(args, data)
    axis.bar(complaints['fold'], complaints['complaints'], label='% of true labels per fold', align='edge')

    axis.get_xaxis().set_visible(False)

    axis.set_ylabel('%', 
                        fontsize=fontsize)

    axis.legend()

    for row, eval_metric in enumerate(eval_metrics):

        axis = plt.subplot(grid[2 + row * 2: 2 + (row + 1) * 2, 0])

        df = data.query(f'eval_metric == "{eval_metric}"')

        # Plot something, it can be more than one 
        axis = sns.boxplot(x='fold', y='score', hue='eval_metric', data=df, ax=axis,
                           boxprops=dict(alpha=0.3)
                          )

        if nrows - 1 > row:
            axis.get_xaxis().set_visible(False)

        # Set tick params size
        axis.tick_params(axis='both', labelsize=12)
        axis.tick_params(axis='x', rotation=45)


    fig.tight_layout()
                                                                                
    persist_local(
        data=fig, 
        args={'experiment_id': args['experiment_id'], 
              'title': title, 'eval_metric': metric}, 
        folder='evaluation_plots', 
        id_keys=['experiment_id', 'title', 'eval_metric'], 
        as_type='.png')
                                 
    # path = utils.path_to_shared('joaoc', 'imgs', 'processo_fechas_diff', 'png')


# +
def plot_metric_by_fold(selector):
    
    
    data = selector['data']
    experiment_id = data['experiment_id'].unique()[0]
    selector_name = selector['name']
    
    metrics = data['eval_metric'].unique()
    
    nrows = 1  # Number of rows
    ncols = 1 # Number of columns
    fontsize = 14 # General fontsize
    
    title = """
Experiment: {experiment_id} 
Metric: {metric}
Selector: {selector_name}"""

    # Loop to create one fig per metric and a line per learner
    for metric in metrics:
        
        fig, axis = plt.subplots(figsize = (15,8))
        
        axis.set_title(title.format(experiment_id=experiment_id,
                                    selector_name=selector_name,
                                    metric=metric
                                   ))
        axis.set_ylabel('score')
        axis.legend()

        #check if it is k-fold or temporal-fold
        if '-' in  data['fold'].iloc[0]:
            axis.set_xlabel('time')
            axis.tick_params(axis='both', labelsize=12)
            axis.tick_params(axis='x', rotation=45)
        else:
            axis.set_xlabel('fold')
            
        for i, learner in enumerate(data['learner_id'].unique()):

            data_to_plot = data[(data['learner_id'] == learner) & 
                                (data['eval_metric'] == metric)]

            axis.plot(data_to_plot['fold'], data_to_plot['score'], 
                      color=next(color), label=data['name'].unique()[0] + '_' + str(learner))


        axis.legend()
        
        persist_local(
        data=fig, 
        args={'experiment_id': experiment_id, 
              'title': title, 'eval_metric': metric}, 
        folder='evaluation_plots', 
        id_keys=['experiment_id', 'title', 'eval_metric'], 
        as_type='.png')
                


# -
import model_selection.select_model as sm

model_selector = 'template'

selectors = sm.run(7657456883, 
                   Path().cwd().parent / 'model_selectors' / (model_selector + '.yaml'),
                  False)

selector = selectors['selectors'][0]

selector.keys()

selector['data']['learner_id'].unique()

selector['name']

plot_metric_by_fold(selector)



