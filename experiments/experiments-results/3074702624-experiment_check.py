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
source_path = str(Path(os.path.abspath('joaoc-experiment-checks')).parent.parent.parent / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)


# Packages
import pandas as pd
pd.options.display.max_columns = 999

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline
# -

from model_selection import plotting, select_model
from utils.utils import connect_to_database

con = connect_to_database()

# Add here your parameters:
experiment_id = 3074702624
model_selector = 'good_selector'

args = dict(experiment_id=experiment_id)

query = f"""
select *
from experiments.errors
where experiment_id = {experiment_id}
order by created_on desc
"""
errors = pd.read_sql_query(query, con)
print('number of errors: ', len(pd.read_sql_query(query, con)))

# +
query = f""" 
   select evaluation.*, approach.name
    from (select *
    from experiments.evaluations
    where experiment_id = {experiment_id}) evaluation
    left join experiments.approaches approach
    on evaluation.experiment_id = approach.experiment_id
    and evaluation.approach_id = approach.approach_id"""

data = pd.read_sql_query(query, con)

data['metric'] = data['eval_metric'].apply(lambda x: x.split('@')[0])
data['at'] = data['eval_metric'].apply(lambda x: float(x.split('@')[1]))
data['type_procurement'] = data['eval_metric'].apply(lambda x: x.split('@')[2])

# +
import seaborn as sns

def get_index_of_line(line):
    return [i.get_text() for i in axis.lines[0].axes.get_legend().texts].index(line) - 1

fontsize = 20# General fontsize

fig, axis = plt.subplots(figsize=(15,10), ncols=1, nrows=1)

axis = sns.lineplot(x="at", 
                    y="score",
                    hue="name",
                    ci=10,
#                     dashes=['higher_than_x'],
                    data=data.query('type_procurement == "overall"').query('metric == "recall"'), ax=axis)

axis.lines[get_index_of_line('higher_than_x')].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x')].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x')].set_color('black')
axis.lines[get_index_of_line('random')].set_linestyle(":")
axis.lines[get_index_of_line('random')].set_linewidth(3)
axis.lines[get_index_of_line('random')].set_color('black')

axis.set_xlabel('%', fontsize=fontsize)
axis.set_ylabel('Recall', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title('Recall for Overall Directors', fontdict={'fontsize': fontsize})
axis.legend(prop={'size': 15})

# +
fig, axis = plt.subplots(figsize=(15,10), ncols=1, nrows=1)

axis = sns.lineplot(x="at", y="score",
             hue="name", ci=0,
             data=data.query('type_procurement == "overall"').query('metric == "precision"'), ax=axis)

axis.lines[get_index_of_line('higher_than_x')].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x')].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x')].set_color('black')
axis.lines[get_index_of_line('random')].set_linestyle(":")
axis.lines[get_index_of_line('random')].set_linewidth(3)
axis.lines[get_index_of_line('random')].set_color('black')

axis.set_xlabel('%', fontsize=fontsize)
axis.set_ylabel('Precision', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title('Precision for Overall Directors', fontdict={'fontsize': fontsize})
axis.legend(prop={'size': 15})

# +
fig, axis = plt.subplots(figsize=(15,10), ncols=1, nrows=1)

axis = sns.lineplot(x="fold", y="score",
             hue="name", ci=95,
             data=data.query('type_procurement == "overall"')\
                      .query('metric == "recall"')\
                      .query('at == 1')
                    , ax=axis)

axis.lines[get_index_of_line('higher_than_x')].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x')].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x')].set_color('black')
axis.lines[get_index_of_line('random')].set_linestyle(":")
axis.lines[get_index_of_line('random')].set_linewidth(3)
axis.lines[get_index_of_line('random')].set_color('black')

axis.set_xlabel('%', fontsize=fontsize)
axis.set_ylabel('Recall', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.tick_params(axis='x', rotation=90)
axis.set_title('Recall @ 1% for Overall Directors', fontdict={'fontsize': fontsize})
axis.legend(prop={'size': 15})

# +
fig, axis = plt.subplots(figsize=(15,10), ncols=1, nrows=1)

axis = sns.lineplot(x="fold", y="score",
             hue="name", ci=95,
             data=data.query('type_procurement == "overall"')\
                      .query('metric == "precision"')\
                      .query('at == 1')
                    , ax=axis)

axis.lines[get_index_of_line('higher_than_x')].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x')].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x')].set_color('black')
axis.lines[get_index_of_line('random')].set_linestyle(":")
axis.lines[get_index_of_line('random')].set_linewidth(3)
axis.lines[get_index_of_line('random')].set_color('black')

axis.set_xlabel('%', fontsize=fontsize)
axis.set_ylabel('Precision', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.tick_params(axis='x', rotation=90)
axis.set_title('Precision @ 1% for Overall Directors', fontdict={'fontsize': fontsize})
axis.legend(prop={'size': 15})
# -

res = select_model.run(args['experiment_id'], Path().cwd().parent.parent / 'model_selectors' / (model_selector + '.yaml'))


best_models, markdown = select_model.get_result_stats(res)
best_models

plotting.plot_approaches_by_fold(experiment_id, best_models.index[0])

plotting.plot_approaches_by_fold_avg_std(experiment_id)

print(markdown)

# ## Feature Importance

learner_ids = set(best_models.index)


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


