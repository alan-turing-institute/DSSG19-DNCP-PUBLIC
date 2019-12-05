from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

import fire
import logging
import pandas as pd
from tqdm import tqdm
import warnings
from tabulate import tabulate
warnings.filterwarnings("ignore")

from model_selection.plotting import plot_metric_by_fold
from pipeline.data_persistence import generate_id, persist_local
from utils.utils import connect_to_database, open_yaml

logging.basicConfig(level=logging.ERROR)

def count_errors(macro_experiment_id):

    con = connect_to_database()

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

def download_data(macro_experiment_id):

    con = connect_to_database()

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
    
    data = explode(data)
    
    return data

def explode(df):

    df['metric'] = df['eval_metric'].apply(lambda x: x.split('@')[0])
    df['at'] = df['eval_metric'].apply(lambda x: float(x.split('@')[1]))
    df['type_procurement'] = df['eval_metric'].apply(lambda x: x.split('@')[2])

    return df

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

def plot(selector_results, experiment_id):

    for selector in selector_results['selectors']:
        plot_metric_by_fold(selector, experiment_id)
        
def get_result_stats(results):
    stats = []
    for result in results['selectors']:
        if len(result['learner_stats']) != 0:
            stat = result['data'].pivot_table(values='score', columns='eval_metric', index=['learner_id', 'fold'])\
                .reset_index().groupby('learner_id').mean()
            stat['selector'] = result['name']
            stat['experiment_id'] = result['data']['experiment_id'].values[0]
            stat['algorithm'] = result['data']['name'].values[0]
            stats.append(stat)

    stats = pd.concat(stats)[['experiment_id', 'algorithm', 'selector', 'f1', 'precision', 'recall']].rename(columns={'f1': 'mean_f1',
                                            'precision': 'mean_precision', 
                                            'recall': 'mean_recall'})

    markdown = tabulate(stats.reset_index(), tablefmt='pipe', headers=stats.reset_index().columns, showindex=False)
    return stats, markdown


def run(macro_experiment_id, selector):

    data = download_data(macro_experiment_id)

    max_fold = max(data['fold'].unique())

    selector_results = run_selectors(selector, data[['learner_id', 'fold', 'name',
                'eval_metric', 'score', 'experiment_id', 
                'approach_id']].query(f'fold != "{max_fold}"'))

    return selector_results, data, max_fold

if __name__ == '__main__':
    # python src/pipeline/pipeline.py run_experiment --experiment_file='dummy_experiment.yaml'
    fire.Fire()
