from utils import utils
from pipeline.data_persistence import get_experiment, persist_local, check_if_local_exists, get_local
from pipeline.config_loading import load_experiment
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
if source_path not in sys.path:
    sys.path.insert(0, source_path)


# Import external packages

# Import internal python scripts


def create_labels(args):
    """
    Function to obtain a dataframe of labels from experiment file corresponding
    to cohort

    Parameters
    ----------
    experiment: dict
        Experiment file with model parameters

    Return
    ---------
    pd.DataFrame
        Dataframe of IDs and labels
    """

    experiment = get_experiment(args['experiment_id'])
    features = get_local(args, 'features')['id_llamado']

    query ="""
        select distinct labels.id_llamado as id_llamado, tipo_procedimiento_codigo, 
        labels.reception_date, {label_target} as target
        from semantic.labels labels
        join semantic.tenders tenders
        on labels.id_llamado = tenders.id_llamado
        where labels.id_llamado in ({cohort})
    """.format(cohort=experiment['cohort_config']['query'],
               label_target=experiment['label_config']['query'])

    con = utils.connect_to_database()
    labels = pd.read_sql_query(query, con)

    labels = labels[labels['id_llamado'].isin(features)]

    persist_local(labels, args, 'labels')


def create_features(args):
    """
    Function to obtain features specified in the experiment file.
    Function will loop over all the features.

    Parameters:
    ------------
    experiment: dict
        Experiment file with model parameters

    Return:
    ------------
    pd.DataFrame
        A dataframe of features corresponding to each cohort
    """

    experiment = get_experiment(args['experiment_id'])

    query_config = """with cd_tenders as (
            {cohort}
            )
                select cd_tenders.id_llamado, {columns}
                from cd_tenders
                left join {table} as feature_table
                on cd_tenders.id_llamado = feature_table.id_llamado
        """

    con = utils.connect_to_database()

    features_combined = pd.DataFrame()

    for feature_config in experiment['features']:

        query = query_config.format(cohort=experiment['cohort_config']['query'],
                                    columns=','.join(
                                        feature_config['columns']),
                                    table=feature_config['table'])

        features = pd.read_sql_query(query, con)

        if features_combined.empty:
            features_combined = features
        else:
            features_combined = features_combined.merge(
                features, on='id_llamado', how='inner')

    # print(features_combined.columns)

    features_combined = features_combined.dropna()

    persist_local(features_combined, args, 'features')


def generate_kfolds(experiment, args):
    """ Given a label table, it generates random stratified folds.

    Parameters
    ----------
    experiment : dict
        Paramenters of to perform the experiment
    args : dict
        Minimum set of parameters to run the pipeline

    Returns
    -------
    list of dicts
        All folds information. It carries the ids to filter the
        tables
    """

    skf = StratifiedKFold(n_splits=experiment['validation']['parameters']['number_of_folds'],
                          random_state=experiment['model_config']['random_seed'])

    labels = get_local(args, 'labels')
    X = labels['id_llamado']
    y = labels['target']

    folds = []

    for i, index in enumerate(skf.split(X, y)):

        folds.append({
            'name': i,
            'train': X[index[0]].tolist(),
            'test': X[index[1]].tolist()
        })
    
    return folds


def generate_temporal_folds(experiment, args):
    """ Given a label table and temporal parameters, it generates temporal folds.

    Parameters
    ----------
    experiment : dict
        Paramenters of to perform the experiment
    args : dict
        Minimum set of parameters to run the pipeline

    Returns
    -------
    list of dicts
        All folds information, by as of date. It carries the ids to filter the
        tables
    """

    params = experiment['validation']['parameters']
    current_aod = dt.datetime.strptime(params['as_of_date'], '%Y-%m-%d')

    labels = get_local(args, 'labels')
    X = labels[['id_llamado', 'reception_date']]
    y = labels['target']

    k = 1
    folds = []
    while True:
        test_end = current_aod + dt.timedelta(days=params['test_lag'])

        if test_end > dt.datetime.strptime(
                params['test_date_limit'], '%Y-%m-%d'):
            break

        if params['number_of_folds'] is not None:
            if k > params['number_of_folds']:
                break

        # If train_lag is 'all', train_start is set to a dummy old date
        # (2000-01-01)
        train_start = (current_aod - dt.timedelta(days=params['train_lag']) - dt.timedelta(days=params['blind_gap'])) \
            if params['train_lag'] != 'all' else dt.datetime(2000, 1, 1)
        train_end = current_aod - dt.timedelta(days=params['blind_gap'])

        train_ids = X.query(
            f"reception_date >= '{train_start}' and reception_date <= '{train_end}'")['id_llamado']
        test_ids = X.query(
            f"reception_date >= '{current_aod}' and reception_date <= '{test_end}'")['id_llamado']

        folds.append({
            'name': dt.datetime.strftime(current_aod, '%Y-%m-%d'),
            'train': train_ids.tolist(),
            'test': test_ids.tolist()
        })

        current_aod = current_aod + dt.timedelta(days=params['aod_lag'])
        k = k + 1

    persist_local(folds, args, 'folds', as_type='.p')

    return folds


def create_folds(args):
    """ Handles folder creation given the type of validation.

    It can handle k-folding and temporal folding

    Parameters
    ----------
    args : dict
        Minumum set of arguments to run the pipeline

    Returns
    -------
    list of dicts
        All folds information. It carries the ids to filter the
        tables
    """

    experiment = get_experiment(args['experiment_id'])

    if experiment['validation']['method'] == 'k-folding':

        return generate_kfolds(experiment, args)

    elif experiment['validation']['method'] == 'temporal-folding':

        return generate_temporal_folds(experiment, args)


def create_model_configuration(args):
    """
    Function wrapper that creates labels and features for modeling

    Parameters:
    ------------
    args: dictionary
        Minimum set of arguments to start functions.

    Return:
    ------------
    args: dictionary
        Minimum set of arguments to start functions.
    """

    if not check_if_local_exists(args, 'features'):
        create_features(args)

    if not check_if_local_exists(args, 'labels'):
        create_labels(args)

    #   TODO:   temporal_folds = some_function()
    args['folds'] = create_folds(args)

    return args
