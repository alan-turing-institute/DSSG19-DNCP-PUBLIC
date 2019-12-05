# -*- coding: utf-8 -*-
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

# # Config

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath('joaoc-evaluation@%')).parent.parent / 'src')
dncp_path = str(Path(os.path.abspath('joaoc-evaluation@%')).parent.parent / 'aequitas' / 'src')
# if source_path not in sys.path:
sys.path.insert(0, dncp_path)
sys.path.insert(0, source_path)
    

# Packages
import pandas as pd
pd.options.display.max_columns = 999

import warnings
warnings.filterwarnings('ignore')

# +
from model_selection import plotting, select_model
from sklearn import preprocessing 
import pandas as pd
from collections import defaultdict
import copy
import yaml
import aequitas

from pipeline.data_persistence import persist_local, get_local
from model_selection import plotting, select_model, select_bias
from utils.utils import connect_to_database, path_to_shared

con = connect_to_database()
# -

# # Metrics Selector

# Add here your parameters:
macro_experiment_id = 8216636716
args = dict(experiment_id=macro_experiment_id)

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
# -

selector = \
"""
rules:
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 3
        statistic: mean
        higher_is_better: true
        

selectors:
    -
        order: [1] #overall at 30%
"""

res = select_model.run(args['experiment_id'], yaml.load(selector), data.query('fold != "2017-12-26"'))
selector = res['selectors'][0]
selected_learners = selector['learner_stats'].index.to_frame(index=False)

bias_selector = yaml.load("""
query: |
    select 
            tenders.id_llamado, 
            tipo_entidad,
            _objeto_licitacion,
            tipo_procedimiento_codigo,
            idh
    from semantic.tenders as tenders
    left join (
    select 
        case when idh::float <= 0.703 then 'Low'
             when idh::float > 0.744 then 'High'
             when (idh::float <= 0.735 and idh::float > 0.703) then 'Medium Low'
             when (idh::float > 0.735 and idh::float <= 0.744) then 'Medium High'
        end idh,
        id_llamado
    from semantic.tenders_demographics) as idh 
    on tenders.id_llamado = idh.id_llamado
    

groups:
    -
        name: 'tipo_entidad'
        new_name: 'Agency Type'
        remove: ['NO CLASIFICADO']
        replace:
            Organismos de la Administración Central: Central Adm.
            Municipalidades: Cities
            Entidades Descentralizadas: Decentralised Entities
    -
        name: 'tipo_procedimiento_codigo'
        new_name: 'Tender Value'
        replace:         
            CD: Low Value
            CO: Medium Value
            LPN: High Value
            LPI: High Value
            LC: High Value
            CE: High Value
    -
        name: '_objeto_licitacion'
        new_name: 'Service Category'
        remove: [LOCINM, LOCMUE, TIERRAS, CON]
        replace:
            SER: Services
            BIEN: Goods
            OBRAS: Infrastructure
            CON: Consultancy
    -
        name: 'idh'
        new_name: 'Human Development Index'
        
selection:
    rules:
        - 
            attribute_name: 'Agency Type'
            metric: 'ppr'
            filter: 
                type: top
                value: 1
            statistic: sum_of_abs_difference
            higher_is_better: false    

    selectors:
        -
            order: [1] #overall at 30%
        -
            order: [1] #overall at 30%
""")

baselines = [{'database_name': 'random',
              'plot_name': 'Random Baseline'},
              {'database_name': 'higher_than_x',
               'plot_name': 'Ordered By Value'}]

# %%capture
results, baselines, selected_data = select_bias.run(bias_selector, data, selected_learners, baselines)

select_bias.plot_bias(results, baselines, selected_data)


# # Prepare Aequitas

# +
def get_data(selected_learners):
    
    final = []
    for args in selected_learners.to_dict('records'):
        
        args['fold_name'] = '2017-12-26'
        
        try: 
        
            final.append(dict(
                learner_id=args['learner_id'],
                experiment_id=args['experiment_id'],
                results=get_local(args, 'predictions', 
                              id_keys=['experiment_id', 'approach_id', 'learner_id', 'fold_name'])\
                            .merge(get_local(args, 'labels').set_index('id_llamado'),
                            left_index=True, right_index=True)[['reception_date', 'prediction', 'target']]
                            ))
        except:
            
            print('debug')

    return final

def label_data(results, minimum_per_day=1, perc_good_quality=0.3):

    df = results.groupby(results['reception_date'].dt.date)\
           .apply(lambda x: x.nlargest(max(minimum_per_day, 
                                           int(len(x) * perc_good_quality)), 
                                       'prediction'))\
           .drop('reception_date', 1).reset_index()
    

    df['score'] = 1
    
    df = results.reset_index().merge(df[['id_llamado', 'score']], 
                                on='id_llamado', how='outer').fillna(0)[['id_llamado', 'target', 'score']]

    return df

def download_bias_data():

    query = f"""
    select 
            tenders.id_llamado, 
            tipo_entidad,
            _objeto_licitacion,
            tipo_procedimiento_codigo,
            idh
    from semantic.tenders as tenders
    left join (
    select 
        case when idh::float <= 0.703 then 'Low'
             when idh::float > 0.744 then 'High'
             when (idh::float <= 0.735 and idh::float > 0.703) then 'Medium Low'
             when (idh::float > 0.735 and idh::float <= 0.744) then 'Medium High'
        end idh,
        id_llamado
    from semantic.tenders_demographics) as idh 
    on tenders.id_llamado = idh.id_llamado
    """

    categories = pd.read_sql_query(query, con)
    
    return categories


def filter_bias_data(df, groups):

    for group in groups:

        if 'replace' in group:

            df[group['name']] = df[group['name']].replace(group['replace'])

        if 'remove' in group:

            df = df[~df[group['name']].isin(group['remove'])]

        if 'new_name' in group:

            df = df.rename(columns={group['name']:group['new_name']})
            
    return df

def add_bias(labels, bias):
    
    return labels.merge(bias, on='id_llamado').rename(columns=
                            {'target': 'label_value', 
                            'id_llamado': 'entity_id',
                           'learner_id': 'model_id'})

def get_aequitas(df):
    
    g = aequitas.group.Group()
    aeq, _ = g.get_crosstabs(df)

    b = aequitas.bias.Bias()
    aeq = b.get_disparity_major_group(aeq, df)
    
    return aeq
    
def melt_aequitas(selected_data):
    
    full_aeq = pd.concat([d['aequitas'] for d in selected_data])
    
    id_vars = ['model_id', 'attribute_name', 'attribute_value']
    value_vars = [a for a in full_aeq if a not in id_vars + ['score_threshold', 'k']]
    
    melted_aequitas = full_aeq.melt(id_vars=id_vars, value_vars=value_vars, var_name='metric')
    melted_aequitas = melted_aequitas[melted_aequitas['value'].apply(lambda x: isinstance(x, float))]
    melted_aequitas['value'] = melted_aequitas['value'].apply(float)
    
    return melted_aequitas


# -

# # Bias Selector

# +
def apply_statistics(rule, data):
    
    statistics = {
        'sum_of_abs_difference': lambda s: sum(abs(1 - s)),
        'maximum': lambda s: max(s)
    }

    return data[(data['attribute_name'] == rule["attribute_name"]) &
                (data['metric'] == rule["metric"])]\
            .groupby('model_id')['value']\
            .apply(statistics[rule['statistic']]).to_frame()

def filter_by(rule, data):
    
    if rule['filter']['type'] == 'threshold':
        if rule['higher_is_better']:
            return data[data['value'] >= rule['filter']['value']]
        else:
            return data[data['value'] <= rule['filter']['value']]
        
    elif rule['filter']['type'] == 'top':
        
        data = data.sort_values(by='value', ascending=not rule['higher_is_better'])
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
    data = data[data['model_id'].isin(filtered_ids.index)]
    
    return apply_selection(rules[1:], filtered_ids, data)

def prepare_to_selection(selected_learners):
    
    selected_data = get_data(selected_learners)

    bias = download_bias_data()
    bias = filter_bias_data(bias, groups)

    for data in selected_data:
        data['labels'] = label_data(data['results'])
        data['bias'] = add_bias(data['labels'], bias)

        data['bias']['model_id'] = data['learner_id']
        data['aequitas'] = get_aequitas(data['bias'], )

    melted_aequitas = melt_aequitas(selected_data)
    
    return melted_aequitas, selected_data
# -



# +
import matplotlib.pyplot as plt
import seaborn as sns

def force_order(df, keys):
    
    return df.set_index('attribute').ix[keys].reset_index()

def plot_bias(df, approach_to_plot='Best Model'):
    
    # Set quantity and position of axis
    nrows = 2 # Number of rows
    ncols = 2 # Number of columns
    fontsize = 20 # General fontsize

    # Create a subplot grid
    fig, axis = plt.subplots(nrows=2, ncols=2,
                           figsize=(20, 12)) # (width, heigth)

    fig.suptitle('Main title', fontsize=30)
    
    # Iterate through plots
    i=0
    for ax, name in zip(axis.flatten(), df['attribute_group'].unique()):
        
        # Define subset of data to plot
        df_to_plot = df.loc[(df['attribute_group'] == name)]
           
        
        if name == 'Tender Value':
            df_to_plot = force_order(df_to_plot, ['Low Value', 'Medium Value', 'High Value'])
            
        if name == 'Human Development Index':
            keys = ['Low', 'Medium Low', 'Medium High', 'High']
            df_to_plot = force_order(df_to_plot, keys)
        
        
        ax = sns.barplot(x='attribute', y='perc_tenders', hue='approach', data=df_to_plot, ax=ax,
                        hue_order=['Random Baseline', 'Ordered By Value', 'Best Model'],
                        palette=['grey', '#1f497d', '#f3b32a'])
        
        # Set title
        ax.set_title(name, fontsize=fontsize)

        ax.set_ylabel('% of procurements to review', 
                        fontsize=fontsize)
        
        ax.set_xlabel('')
        # Set axis limits

        ax.set_ylim(0, 100)

        # Set tick params size
        ax.tick_params(axis='both', labelsize=fontsize)

        # Set legend
        ax.legend(fontsize=fontsize)
 
        i = i + 1


# -

round(3/2)

plot_bias(plot_data)


