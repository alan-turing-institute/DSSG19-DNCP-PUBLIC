# Import Scripts
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
dncp_path = str(Path(os.path.abspath(__file__)).parent.parent.parent / 'aequitas' / 'src')
# if source_path not in sys.path:
sys.path.insert(0, dncp_path)
sys.path.insert(0, source_path)

from model_selection import plotting, select_model
import pandas as pd
import copy
import yaml
import aequitas
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline.data_persistence import persist_local, get_local
from utils.utils import connect_to_database, path_to_shared

# Prepare to Selection

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

def download_bias_data(query=None, con=None):

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

# Select by Bias
    
def melt_aequitas(selected_data):
    
    full_aeq = pd.concat([d['aequitas'] for d in selected_data])
    
    id_vars = ['model_id', 'attribute_name', 'attribute_value']
    value_vars = [a for a in full_aeq if a not in id_vars + ['score_threshold', 'k']]
    
    melted_aequitas = full_aeq.melt(id_vars=id_vars, value_vars=value_vars, var_name='metric')
    melted_aequitas = melted_aequitas[melted_aequitas['value'].apply(lambda x: isinstance(x, float))]
    melted_aequitas['value'] = melted_aequitas['value'].apply(float)
    
    return melted_aequitas

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

# Plotting



# Main functions 
def add_baselines(data, selected_learners, baselines):    
    
    selected_learners = pd.concat([
        selected_learners, 
        data[data['name']\
            .isin([x['database_name'] for x in baselines])][['learner_id', 'experiment_id', 'approach_id']]\
            .drop_duplicates()])

    for base in baselines:

        base['id'] =  data[data['name'].isin([base['database_name']])][
                            ['learner_id', 'experiment_id', 'approach_id']]\
                            .drop_duplicates()
        
    return selected_learners, baselines

def prepare_to_selection(selected_learners, configuration):

    con = connect_to_database()
    
    selected_data = get_data(selected_learners)

    bias = download_bias_data(query=configuration['query'], con=con)
    bias = filter_bias_data(bias, configuration['groups'])

    for data in selected_data:
        data['labels'] = label_data(data['results'])
        data['bias'] = add_bias(data['labels'], bias)

        data['bias']['model_id'] = data['learner_id']
        data['aequitas'] = get_aequitas(data['bias'], )

    melted_aequitas = melt_aequitas(selected_data)
    
    return melted_aequitas, selected_data

def generate_rule_name(rule):
    
    return '_'.join(map(str, [rule['statistic'], rule['attribute_name'], rule["metric"], 
                              rule['filter']['type'], rule['filter']['value']]))

def apply_selectors(melted_aequitas, selector_config, baselines):

    results = []

    for selector in selector_config['selectors']:
        
        rules = [selector_config['rules'][i-1] 
                 for i in selector['order']]

        # Exclude baselines from melted aequita
        baseline_learners = [base['id']['learner_id'].iloc[0] for base in baselines]
        learners, data = apply_selection(rules, 
                        None,  
                        melted_aequitas[~melted_aequitas.isin(baseline_learners)])

        name = '->'.join(map(generate_rule_name, rules))

        results.append({'rules': rules, 'learners': learners, 'data': data, 'name': name})

    return results

def run(bias_selector, data, selected_learners, baselines):

    selected_learners, baselines = add_baselines(data, selected_learners, baselines)

    melted_aequitas, selected_data = prepare_to_selection(selected_learners, bias_selector)

    results = apply_selectors(melted_aequitas, 
                            bias_selector['selection'],
                            baselines)

    return results, baselines, selected_data