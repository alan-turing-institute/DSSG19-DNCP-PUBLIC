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
import matplotlib.pyplot as plt
pd.options.display.max_columns = 999

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline
# -

from model_selection import plotting, select_model
from utils.utils import connect_to_database

con = connect_to_database()

# Add here your parameters:
macro_experiment_id = 3481383696
model_selector = 'good_selector'

args = dict(experiment_id=macro_experiment_id)

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

def explode(df):

    df['metric'] = df['eval_metric'].apply(lambda x: x.split('@')[0])
    df['at'] = df['eval_metric'].apply(lambda x: float(x.split('@')[1]))
    df['type_procurement'] = df['eval_metric'].apply(lambda x: x.split('@')[2])
    
    return df

data = explode(data)

# +
import seaborn as sns

def get_index_of_line(line, df, forbiden='forest'):
    line = [i for i in df.unique() if ((line in i) & (forbiden not in i))][0]
    return [i.get_text() for i in axis.lines[0].axes.get_legend().texts].index(line) - 1

fontsize = 20# General fontsize

fig, axis = plt.subplots(figsize=(15,10), ncols=1, nrows=1)

axis = sns.lineplot(x="at", 
                    y="score",
                    hue="name",
                    ci=95,
#                     dashes=['higher_than_x'],
                    data=data.query('type_procurement == "overall"')\
                    .query('metric == "recall"'), ax=axis)

axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_color('black')
axis.lines[get_index_of_line('random', data['name'])].set_linestyle(":")
axis.lines[get_index_of_line('random', data['name'])].set_linewidth(3)
axis.lines[get_index_of_line('random', data['name'])].set_color('black')

axis.set_xlabel('%', fontsize=fontsize)
axis.set_ylabel('Recall', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title('Recall for Overall Directors', fontdict={'fontsize': fontsize})
axis.legend(prop={'size': 15})

# +
fig, axis = plt.subplots(figsize=(15,10), ncols=1, nrows=1)

axis = sns.lineplot(x="at", y="score",
             hue="name", ci=95,
             data=data.query('type_procurement == "overall"').query('metric == "precision"'), ax=axis)

axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_color('black')
axis.lines[get_index_of_line('random', data['name'])].set_linestyle(":")
axis.lines[get_index_of_line('random', data['name'])].set_linewidth(3)

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

axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_color('black')
axis.lines[get_index_of_line('random', data['name'])].set_linestyle(":")
axis.lines[get_index_of_line('random', data['name'])].set_linewidth(3)

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

axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x', data['name'])].set_color('black')
axis.lines[get_index_of_line('random', data['name'])].set_linestyle(":")
axis.lines[get_index_of_line('random', data['name'])].set_linewidth(3)

axis.set_xlabel('%', fontsize=fontsize)
axis.set_ylabel('Precision', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.tick_params(axis='x', rotation=90)
axis.set_title('Precision @ 1% for Overall Directors', fontdict={'fontsize': fontsize})
axis.legend(prop={'size': 15})
# -

selector = \
"""
rules:
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 100
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
        order: [1] #overall at 30%
    -
        order: [3, 4] #overall at 10%
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

import yaml
res = select_model.run(args['experiment_id'], yaml.load(selector), data.query('fold != "2017-12-26"'))


df = res['selectors'][0]['data']

learners = df[df['fold'] == '2015-06-30']['learner_id'].unique()

stats = res['selectors'][0]['learner_stats'].reset_index()

best_learner = stats[stats['learner_id'].isin(learners)]['learner_id'].values[0]

df = df[df['learner_id'] == best_learner]

# +
selector = res['selectors'][0]

fontsize = 25

fig, axis = plt.subplots(figsize=(15,  10),
                         ncols=1, nrows=1)

df = explode(selector['data'])
df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
translate = {'higher_than_x': 'Ordered by Value Baseline',
                   'random': 'Random Baseline',
                   'xgb': 'Best Model'
                  }
df['Models'] = df['name'].apply(lambda x: translate[x])

axis = sns.lineplot(x="at", 
                    y="score",
                    hue="Models",
                    hue_order=['Random Baseline'],
                    ci=95,
                    data=df.query('type_procurement == "overall"')\
                            .query('metric == "recall"')\
                            .query('Models == "Random Baseline"'),
                    ax=axis)


# higher_idx = get_index_of_line('Ordered', df['Models'])
# axis.lines[higher_idx].set_linestyle("--")
# axis.lines[higher_idx].set_linewidth(5)
# axis.lines[higher_idx].set_color('black')

# random_idx = get_index_of_line('Random', df['Models'])
# axis.lines[random_idx].set_linestyle(":")
# axis.lines[random_idx].set_linewidth(10)
# axis.lines[random_idx].set_color('black')

axis.set_ylabel('Recall', fontsize=fontsize)
axis.set_xlabel('% of quality reviews', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Best model for Recall at 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': fontsize}, loc='lower right')
fig.tight_layout()
fig.savefig('random_baseline.png')

# +
selector = res['selectors'][0]

fig, axis = plt.subplots(figsize=(15,  10),
                         ncols=1, nrows=1)

df = explode(selector['data'])
df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
translate = {'higher_than_x': 'Ordered by Value Baseline',
                   'random': 'Random Baseline',
                   'xgb': 'Best Model'
                  }
df['Models'] = df['name'].apply(lambda x: translate[x])

axis = sns.lineplot(x="at", 
                    y="score",
                    hue="Models",
                    hue_order=['Random Baseline', 'Best Model'],
                    ci=95,
                    data=df.query('type_procurement == "overall"')\
                            .query('metric == "recall"')\
                            .query('Models != "Ordered by Value Baseline"'),
                    ax=axis)


# higher_idx = get_index_of_line('Ordered', df['Models'])
# axis.lines[higher_idx].set_linestyle("--")
# axis.lines[higher_idx].set_linewidth(5)
# axis.lines[higher_idx].set_color('black')

random_idx = get_index_of_line('Random', df['Models'])
axis.lines[random_idx].set_linestyle(":")
axis.lines[random_idx].set_linewidth(10)
axis.lines[random_idx].set_color('black')

axis.set_ylabel('Recall', fontsize=fontsize)
axis.set_xlabel('% of quality reviews', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Best model for Recall at 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': fontsize}, loc='lower right')
fig.tight_layout()
fig.savefig('random_baseline+best_model.png')

# +
selector = res['selectors'][0]

fig, axis = plt.subplots(figsize=(15,  10),
                         ncols=1, nrows=1)

df = explode(selector['data'])
df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
translate = {'higher_than_x': 'Ordered by Value Baseline',
                   'random': 'Random Baseline',
                   'xgb': 'Best Model'
                  }
df['Models'] = df['name'].apply(lambda x: translate[x])

axis = sns.lineplot(x="at", 
                    y="score",
                    hue="Models",
                    hue_order=['Random Baseline', 'Best Model'],
                    ci=95,
                    data=df.query('type_procurement == "overall"')\
                            .query('metric == "recall"')\
                            .query('Models != "Ordered by Value Baseline"'),
                    ax=axis)


# higher_idx = get_index_of_line('Ordered', df['Models'])
# axis.lines[higher_idx].set_linestyle("--")
# axis.lines[higher_idx].set_linewidth(5)
# axis.lines[higher_idx].set_color('black')

random_idx = get_index_of_line('Random', df['Models'])
axis.lines[random_idx].set_linestyle(":")
axis.lines[random_idx].set_linewidth(10)
axis.lines[random_idx].set_color('black')

axis.axvline(x=30, c='red')

axis.set_ylabel('Recall', fontsize=fontsize)
axis.set_xlabel('% of quality reviews', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Best model for Recall at 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': fontsize}, loc='lower right')
fig.tight_layout()
fig.savefig('random_baseline+best_model+vline.png')

# +
selector = res['selectors'][0]

fig, axis = plt.subplots(figsize=(15,  10),
                         ncols=1, nrows=1)

df = explode(selector['data'])
df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
# translate = {'higher_than_x': 'Ordered by Value Baseline',
#                    'random': 'Random Baseline',
#                    'xgb': 'Best Model'
#                   }
# df['Models'] = df['name'].apply(lambda x: translate[x])

axis = sns.lineplot(x="at", 
                    y="score",
                    hue="name",
                    ci=95,
                    data=df.query('type_procurement == "overall"')\
                            .query('metric == "recall"'),
                    ax=axis)

axis.axvline(x=30, c='red')

# higher_idx = get_index_of_line('Ordered', df['Models'])
# axis.lines[higher_idx].set_linestyle("--")
# axis.lines[higher_idx].set_linewidth(10)
# axis.lines[higher_idx].set_color('black')

# random_idx = get_index_of_line('Random', df['Models'])
# axis.lines[random_idx].set_linestyle(":")
# axis.lines[random_idx].set_linewidth(10)
# axis.lines[random_idx].set_color('black')

axis.set_ylabel('Recall', fontsize=fontsize)
axis.set_xlabel('% of quality reviews', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Best model for Recall at 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': fontsize})
fig.tight_layout()
fig.savefig('all.png')
# -

fakedf = """
Tender Value,Proportion @ 30% of Good Reviews,Models
Low,60,Random Baseline
Medium,30,Random Baseline
High,10,Random Baseline
Low,25,Best Model
Medium,35,Best Model
High,40,Best Model
Low,10,Ordered By Value Baseline
Medium,20,Ordered By Value Baseline
High,70,Ordered By Value Baseline"""
open('fakedf.csv', 'w').write(fakedf)

fakedf = pd.DataFrame.from_csv('fakedf.csv').reset_index(
)

fakedf.columns

# +
fig, axis = plt.subplots(figsize=(15,  10),
                         ncols=1, nrows=1)

df1 = fakedf.query('Models == "Random Baseline"')
axis.bar(x=df1['Tender Value'], 
         height=df1['Proportion @ 30% of Good Reviews'],
        label='Random Baseline', color='blue', alpha=0.3)


axis.set_xlabel('Tender Value', fontsize=fontsize)
axis.set_ylabel('Proportion (%)', fontsize=fontsize)
axis.set_ylim([0, 100])
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Distribution of Labeled Tenders @ 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': fontsize})
fig.tight_layout()
fig.savefig('random+dist.png')

# +
fig, axis = plt.subplots(figsize=(15,  10),
                         ncols=1, nrows=1)

df1 = fakedf.query('Models == "Random Baseline"')
axis.bar(x=df1['Tender Value'], 
         height=df1['Proportion @ 30% of Good Reviews'],
        label='Random Baseline', color='blue', alpha=0.3)

df1 = fakedf.query('Models == "Ordered By Value Baseline"')
axis.bar(x=df1['Tender Value'], 
         height=df1['Proportion @ 30% of Good Reviews'],
        label='Ordered By Value Baseline', color='green', alpha=1,
         width=0.7)

axis.set_xlabel('Tender Value', fontsize=fontsize)
axis.set_ylabel('Proportion (%)', fontsize=fontsize)
axis.set_ylim([0, 100])
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Distribution of Labeled Tenders @ 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': fontsize})
fig.tight_layout()
fig.savefig('random+ordered.png')

# +
fig, axis = plt.subplots(figsize=(15,  10),
                         ncols=1, nrows=1)

df1 = fakedf.query('Models == "Random Baseline"')
axis.bar(x=df1['Tender Value'], 
         height=df1['Proportion @ 30% of Good Reviews'],
        label='Random Baseline', color='blue', alpha=0.3)


df1 = fakedf.query('Models == "Best Model"')
axis.bar(x=df1['Tender Value'], 
         height=df1['Proportion @ 30% of Good Reviews'],
        label='Best Model', color='orange', alpha=1,
         width=0.7)

axis.set_xlabel('Tender Value', fontsize=fontsize)
axis.set_ylabel('Proportion (%)', fontsize=fontsize)
axis.set_ylim([0, 100])
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Distribution of Labeled Tenders @ 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': fontsize})
fig.tight_layout()
fig.savefig('random+best.png')
# -

df.groupby(['name', 'type_procurement'])['score'].mean()



# +
nrows = len(res['selectors'])

fig, ax = plt.subplots(figsize=(15, nrows * 10),
                         ncols=1, nrows=nrows)

for i, selector in enumerate(res['selectors']):
    
    axis = ax[i]

    df = explode(selector['data'])
    df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
    df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
    translate = {'higher_than_x': 'Ordered by Value Baseline',
                       'random': 'Random Baseline',
                       'xgb': 'Best Model'
                      }
    df['Models'] = df['name'].apply(lambda x: translate[x])

    axis = sns.lineplot(x="at", 
                        y="score",
                        hue="Models",
                        ci=95,
                        data=df.query('type_procurement == "overall"')\
                                .query('metric == "recall"'),
                        ax=axis)
    
    
    higher_idx = get_index_of_line('Ordered', df['Models'])
    axis.lines[higher_idx].set_linestyle("--")
    axis.lines[higher_idx].set_linewidth(5)
    axis.lines[higher_idx].set_color('black')
    
    random_idx = get_index_of_line('Random', df['Models'])
    axis.lines[random_idx].set_linestyle(":")
    axis.lines[random_idx].set_linewidth(3)
    axis.lines[random_idx].set_color('black')

    axis.set_ylabel('Recall', fontsize=fontsize)
    axis.set_xlabel('% of quality reviews', fontsize=fontsize)
    axis.tick_params(axis='both', labelsize=fontsize)
    axis.set_title("""
    Best model for Recall at 30% of Quality Reviews""".format(selector['name']), fontdict={'fontsize': fontsize})
    axis.legend(prop={'size': 15})
# -

best_model = res['selectors'][0]['learner_stats'].reset_index().iloc[0]['learner_id']

df = data.query(f'learner_id == {best_model}')            
df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
df = df.query('fold == "2017-12-26"')

df['new_name'].unique()

# +
fig, axis = plt.subplots(figsize=(15,10), ncols=1, nrows=1)

df = data.query(f'learner_id == {best_model}')            
df = pd.concat([df, data.query('name in ("higher_than_x", "random")')])
df['new_name'] = df.apply(lambda x: x['name'] + '_' + str(x['learner_id']), 1)
df = df.query('fold == "2017-12-26"')

axis = sns.lineplot(x="at", 
                    y="score",
                    hue="new_name",
                    ci=95,
#                     dashes=['higher_than_x'],
                    data=df\
                    .query('type_procurement == "overall"')\
                    .query('metric == "recall"'),
                    ax=axis)

axis.lines[get_index_of_line('higher_than_x', df['new_name'])].set_linestyle("--")
axis.lines[get_index_of_line('higher_than_x', df['new_name'])].set_linewidth(5)
axis.lines[get_index_of_line('higher_than_x', df['new_name'])].set_color('black')
axis.lines[get_index_of_line('random', df['new_name'])].set_linestyle(":")
axis.lines[get_index_of_line('random', df['new_name'])].set_linewidth(3)
axis.lines[get_index_of_line('random', df['new_name'])].set_color('black')

axis.set_xlabel('%', fontsize=fontsize)
axis.set_ylabel('Recall', fontsize=fontsize)
axis.tick_params(axis='both', labelsize=fontsize)
axis.set_title("""
Recall for Overall Directors
{}""".format(selector['name']), fontdict={'fontsize': fontsize})
axis.legend(prop={'size': 15})

# + {"active": ""}
# best_models, markdown = select_model.get_result_stats(res)
# best_models
# -

plotting.plot_approaches_by_fold(experiment_id, best_models.index[0])

plotting.plot_approaches_by_fold_avg_std(experiment_id)

print(markdown)

# ## Feature Importance

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


