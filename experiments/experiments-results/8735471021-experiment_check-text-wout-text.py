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

# # Setup

# ## Imports

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath('joaoc-experiment-checks')).parent.parent.parent / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)


# Packages
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = 999
import seaborn as sns
import yaml
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

from model_selection import plotting, select_model
from utils.utils import connect_to_database, path_to_shared

# %matplotlib inline
# -

con = connect_to_database()

# ## Matplotlib functions and config

# +
fontsize = 20 # General fontsize

def get_index_of_line(line, df, forbiden='forest'):
    line = [i for i in df.unique() if ((line in i) & (forbiden not in i))][0]
    return [i.get_text() for i in axis.lines[0].axes.get_legend().texts].index(line) - 1


# -

# ## >> Insert Macro Experiment ID >>

# Add here your parameters:
macro_experiment_id = 8735471021
args = dict(experiment_id=macro_experiment_id)

# ## Count Errors

query = f"""
select *
from experiments.errors
where experiment_id in (
    select experiment_id 
    from experiments.parallelization 
    where macro_experiment_id = {macro_experiment_id})
order by created_on desc
"""
errors = pd.read_sql_query(query, con)
print('number of errors: ', len(pd.read_sql_query(query, con)))

# ## Download Data

# +
# This is just for this experiment
# It adds baselines to dataframe

query = f""" 
   select evaluation.*, approach.name
    from (select *
    from experiments.evaluations
    where experiment_id in (
    select experiment_id 
    from experiments.parallelization 
    where macro_experiment_id = 5431381010)) evaluation
    left join experiments.approaches approach
    on evaluation.experiment_id = approach.experiment_id
    and evaluation.approach_id = approach.approach_id"""

data1 = pd.read_sql_query(query, con)
data1 = data1.query('name in ("higher_than_x", "random")')

# +
query = f""" 
   select evaluation.*, approach.name
    from (select *
    from experiments.evaluations
    where experiment_id in (
    select experiment_id 
    from experiments.parallelization 
    where macro_experiment_id = {macro_experiment_id})) evaluation
    left join experiments.approaches approach
    on evaluation.experiment_id = approach.experiment_id
    and evaluation.approach_id = approach.approach_id"""

data = pd.read_sql_query(query, con)

data = pd.concat([data, data1])

def explode(df):

    df['metric'] = df['eval_metric'].apply(lambda x: x.split('@')[0])
    df['at'] = df['eval_metric'].apply(lambda x: float(x.split('@')[1]))
    df['type_procurement'] = df['eval_metric'].apply(lambda x: x.split('@')[2])
    
    return df

data = explode(data)
# -

# # First Data Selection: Performance Metrics
#
# It selects models according to overall performance metrics such as `recall` and `precision`.

# +
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
# filter: dict
#       type: str
#          It can be `threshold` or `top`
#       value: number
#           A number to filter by
#
#       It filters the statistic of the metrix by threshold or top. 
#
# statistic: string
#       Any statistic accepted by pd.Groupby.agg function.
#       Some examples are: mean, median, std, sum, min, max, skew, kurt, sem, var
#
# higher_is_better: boolean
#       Teels the algorithm how to order the list given the statistic of the metric.
#       If the metric is good with high values, then, this parameter should be true

# Selector is a set of rules to be applied to an experiment result.         
# One can build multiple selectors with different rules. 
# 
# To build a selector, you need to declare the order that you want the rules
# above to be applied. For instance, a selector can use the rules in the following order:
#  1 -> 3 -> 2. Or it can be as simple as one rule: 1
#
# This chains are defined by a list, [1, 3, 2] or [1], that has to be added to the order key
# for each selector.
# -

selector = \
"""
rules:
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 10
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 1
        statistic: std
        higher_is_better: false
    - 
        metric: 'recall@10@overall'
        filter: 
            type: top
            value: 300
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@10@overall'
        filter: 
            type: top
            value: 1
        statistic: std
        higher_is_better: false
    - 
        metric: 'recall@10@overall_groups'
        filter: 
            type: top
            value: 300
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@10@overall_groups'
        filter: 
            type: top
            value: 1
        statistic: std
        higher_is_better: false
    - 
        metric: 'recall@30@overall_groups'
        filter: 
            type: top
            value: 300
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@overall_groups'
        filter: 
            type: top
            value: 1
        statistic: std
        higher_is_better: false
    - 
        metric: 'recall@30@director3'
        filter: 
            type: top
            value: 300
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@director3'
        filter: 
            type: top
            value: 1
        statistic: std
        higher_is_better: false
    - 
        metric: 'recall@30@director2'
        filter: 
            type: top
            value: 300
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@director2'
        filter: 
            type: top
            value: 1
        statistic: std
        higher_is_better: false
    - 
        metric: 'recall@30@director1'
        filter: 
            type: top
            value: 300
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@director1'
        filter: 
            type: top
            value: 1
        statistic: std
        higher_is_better: false
        

selectors:
    -
        order: [1] #overall avg recall at 30%
    -
        order: [3, 4] #overall avg recall at 10%
    -
        order: [5, 6] #overall_groups at 10%
    -
        order: [7, 8] #overall_groups at 30%
    -
        order: [9, 10] #director3 at 30%
    -
        order: [11, 12] #director2 at 30%
    -
        order: [13, 14] #director1 at 30%
"""

res = select_model.run(args['experiment_id'], yaml.load(selector), data.query('fold != "2017-12-26"'))


# ## Recall by Quality Reviews

# +
selector = res['selectors'][0]

for selector in res['selectors']:

    fig, axis = plt.subplots(figsize=(15,  10),
                             ncols=1, nrows=1)
    
    df = explode(selector['data'])
    df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
    
    df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
    defaultdict
    translate = defaultdict(lambda: 'Selected Models')
    translate.update({'higher_than_x': 'Ordered by Value Baseline',
               'random': 'Random Baseline'
               })
                      
    df['Models'] = df['name'].apply(lambda x: a[x])

    axis = sns.lineplot(x="at", 
                        y="score",
                        hue="Models",
                        # hue_order=['Random Baseline', model_name, 'Ordered by Value Baseline'],
                        ci=100,
                        data=df.query('type_procurement == "overall"')\
                                .query('metric == "recall"'),
                        ax=axis)

    higher_idx = get_index_of_line('Ordered', df['Models'])
    axis.lines[higher_idx].set_linestyle("--")
    axis.lines[higher_idx].set_linewidth(10)
    axis.lines[higher_idx].set_color('black')

    random_idx = get_index_of_line('Random', df['Models'])
    axis.lines[random_idx].set_linestyle(":")
    axis.lines[random_idx].set_linewidth(10)
    axis.lines[random_idx].set_color('black')

    axis.set_ylabel('Recall', fontsize=fontsize)
    axis.set_xlabel('% of quality reviews', fontsize=fontsize)
    axis.tick_params(axis='both', labelsize=fontsize)
    axis.set_title(f"""
    Best models for {selector['name']}""".format(selector['name']), fontdict={'fontsize': fontsize})
    axis.legend(prop={'size': fontsize})
    fig.tight_layout()
    path = path_to_shared('user', 'imgs', selector['name'], 'png')
    fig.savefig(path) 

# -

# ## Precision by Quality Reviews

# +
selector = res['selectors'][0]

for selector in res['selectors']:

    fig, axis = plt.subplots(figsize=(15,  10),
                             ncols=1, nrows=1)
    
    df = explode(selector['data'])
    df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
    
    df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
    defaultdict
    translate = defaultdict(lambda: 'Selected Models')
    translate.update({'higher_than_x': 'Ordered by Value Baseline',
               'random': 'Random Baseline'
               })
                      
    df['Models'] = df['name'].apply(lambda x: a[x])

    axis = sns.lineplot(x="at", 
                        y="score",
                        hue="Models",
                        # hue_order=['Random Baseline', model_name, 'Ordered by Value Baseline'],
                        ci=100,
                        data=df.query('type_procurement == "overall"')\
                                .query('metric == "precision"'),
                        ax=axis)

    higher_idx = get_index_of_line('Ordered', df['Models'])
    axis.lines[higher_idx].set_linestyle("--")
    axis.lines[higher_idx].set_linewidth(10)
    axis.lines[higher_idx].set_color('black')

    random_idx = get_index_of_line('Random', df['Models'])
    axis.lines[random_idx].set_linestyle(":")
    axis.lines[random_idx].set_linewidth(10)
    axis.lines[random_idx].set_color('black')

    axis.set_ylabel('Precision', fontsize=fontsize)
    axis.set_xlabel('% of quality reviews', fontsize=fontsize)
    axis.tick_params(axis='both', labelsize=fontsize)
    axis.set_title(f"""
    Best models for {selector['name']}""".format(selector['name']), fontdict={'fontsize': fontsize})
    axis.legend(prop={'size': fontsize})
    fig.tight_layout()
    path = path_to_shared('user', 'imgs', selector['name'], 'png')
    fig.savefig(path) 

# -

# # Second Data Selection: Bias and Fairness Metrics
#
# In development



# # Feature Importance

learner_ids = res['selectors'][0]['learner_stats'].reset_index()['learner_id'].tolist()


# learner_ids = set(best_models.index)

def get_features(learner_ids):
       
    query = """
    select learner_id, feature, importance
    from experiments.feature_importances
    where learner_id in ({learner_ids})
    """.format(learner_ids=','.join(map(lambda x: str(x), learner_ids)))
    
    return pd.read_sql_query(query, con)


def compare_best_features_per_learner(features):
    
    features['importance'] = features['importance'].apply(abs)
    a = features.groupby(['learner_id', 'feature']).mean().reset_index()
    
    features_imp = pd.DataFrame()
    for lid in a['learner_id'].unique():
        features_imp[lid] = a[a['learner_id'] == lid].sort_values(by='importance', ascending=False)\
                            .reset_index()['feature']
    
    return features_imp


def get_feature_importance_distribution_for_learner(features, learner_id, top=20):
    
    f = features[features['learner_id'] == learner_id].groupby(['learner_id', 'feature']).mean().reset_index()
    f.nlargest(10, 'importance')

    return f.nlargest(top, 'importance').plot(y='importance', x='feature', kind='barh', figsize=(8, top*0.5))


features = get_features(learner_ids)

# ## Feature distribution for given learner

get_feature_importance_distribution_for_learner(features, list(learner_ids)[0], top=10)


# ## Compare Feature Importance Across Models

compare_best_features_per_learner(features).head(10).T

# ## Useless Features

useless = features.groupby(['learner_id', 'feature']).mean().query('importance == 0')\
    .reset_index().groupby('feature').count()['importance'].sort_values(ascending=False).to_frame()

len(useless)

useless


