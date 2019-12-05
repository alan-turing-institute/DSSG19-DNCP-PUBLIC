#
# File containing evaluation functions for the predictions.
#     Input:
#         - List with real observations.
#         - List with predicted values.
#         - List of wanted evaluation metrics.
#     Output:
#         Dictionary with the different metrics or scores.

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import eli5
import numpy as np
import logging
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
if source_path not in sys.path:
    sys.path.insert(0, source_path)


import logging
import numpy as np
import pandas as pd
import eli5
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve


# def evaluate(obs, pred, eval_metrics_list):
#     '''
#     Evaluate goodness of fit.
#     Parameters
#     ----------
#     obs: List or array
#         Real observations.
#     pred: List or array
#         Predicted values.
#     eval_metrics_list: List
#         List of wanted evaluation metrics to calculate.

#     Return
#     ------
#     Dictionary
#         Dictionary containing all evaluation metrics and scores.
#     '''
#     eval_dict = {}

#     obs = list(obs)
#     pred = list(pred)

#     for metric in eval_metrics_list:
#         if metric == 'accuracy':
#             accuracy = accuracy_score(obs, pred)
#             logging.info(f"Accuracy score: {accuracy}")
#             eval_dict['accuracy'] = accuracy
        
#         elif metric == 'recall':
#             recall = recall_score(obs, pred)
#             logging.info(f"Recall score: {recall}")
#             eval_dict['recall'] = recall
        
#         elif metric == 'precision':
#             precision = precision_score(obs, pred)
#             logging.info(f"Precision score: {precision}")
#             eval_dict['precision'] = precision

#         elif metric == 'f1':
#             f1 = f1_score(obs, pred)
#             logging.info(f"F1 score: {precision}")
#             eval_dict['f1'] = f1
        
#         else:
#             logging.warning(f'Metric {metric} does not exists')
#             continue


#     return eval_dict

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
    


def get_feature_importance(model, features):
    """Calculates feature importance of classifier using eli5 `explain_weights` method

    Parameters
    ----------
    model : dict
        Carries the model class

    Returns
    -------
    list of dicts
        Response structured as list of dicts with keys: 'feature' and 'weight'
    """

    feature_importances = eli5.explain_weights_df(
        model['chosen_model'], feature_names=list(features.columns))

    if feature_importances is None:
        return list()

    else:
        return feature_importances.to_dict(orient='records')
