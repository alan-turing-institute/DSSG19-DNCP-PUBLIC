'''
ML pipeline: run an experiment end-to-end.
    Input:
        Experiment
    Output:
        - Predictions
        - Evaluation metrics
'''

from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

import fire
import importlib
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import itertools as it
from tqdm import tqdm
import copy
import warnings
warnings.filterwarnings("ignore")

from pipeline.config_loading import load_environment, load_experiment
from pipeline.approaches import dummy_classifier
from pipeline.evaluation import evaluate, get_feature_importance
from pipeline.plotting import do_plots
from pipeline.model_data_prep import create_model_configuration
from pipeline.data_persistence import persist_errors, initialise_experiment, persist_local, get_experiment, get_approaches, persist_learner, persist_evaluation, get_local, persist_feature_importance
from pipeline.preprocessing import apply_preprocessing
from utils.utils import max_run_time
from pipeline.textprocessing import run_tfidf

logging.basicConfig(level=logging.ERROR)


def generate_hyperparameters_combinations(hyperparameters):
    """Given a dict of lists, returns a list of dicts with all unique combinations.
    It generates the Grid for the grid search.

    Parameters
    ----------
    hyperparameters : dict
        Hyperparameters sets to be used to tune the learner

    Returns
    -------
    list of dict
        Unique list of all variables combinations.
    """

    all_hyper = sorted(hyperparameters)

    combinations = it.product(*(hyperparameters[hyper] for hyper in all_hyper))

    return [dict(zip(all_hyper, comb)) for comb in list(combinations)]


def generate_folds_matrices(features, labels, fold, args):
    """Returns the data_dict for test and training given a fold

    Parameters
    ----------
    features : pd.DataFrame
        Features table for the experiment
    labels : pd.DataFrame
        Labels table for the experiment
    fold : dict
        id_llamado lists to filter features and labels

    Returns
    -------
    dict of pd.DataFrame
        train and test dict with features and labels each
    """
   
    features_train = features.loc[fold['train']]
    features_test = features.loc[fold['test']]

    #Check if textprocessing is needed
    if args:
        tfidf_features_train, tfidf_features_test = run_tfidf(fold, args)
        #Get normal features pandas dataframe

        #Combined features pandas DataFrame
        features_train = features_train.merge(tfidf_features_train, on ='id_llamado', how='inner')\
                        .set_index('id_llamado')
        features_test = features_test.merge(tfidf_features_test, on='id_llamado', how='inner')\
                        .set_index('id_llamado')


    # Ensure no NA
    features_train = features_train.dropna()
    features_test = features_test.dropna()

    train_dict = {
        'features': features_train,
        'labels': labels.loc[features_train.index]}

    test_dict =  {
        'features': features_test,
        'labels': labels.loc[features_test.index]}

    return train_dict, test_dict


def loop_the_grid(args):
    """
    Given the experiment file with experiment parameters, the list of
    temporal_folds as well as the data dictionary prepared by the
    model_data_prep function, the function loops through the various temporal folds
    and the list of approaches specified in the experiment file to calculate
    metrics specified in the experiment file.

    Parameters
    ----------
    args: dictionary
        Minimum set of arguments to start functions.
    """

    experiment = get_experiment(args['experiment_id'])
    approaches = get_approaches(args['experiment_id'])

    features = get_local(args, 'features').set_index('id_llamado')
    labels = get_local(args, 'labels').set_index('id_llamado')

    #Check if textprocessing is needed:
    if 'textprocessing' in experiment:
        args_tfidf = {}
        args_tfidf['params'] = experiment['textprocessing']['tfidf']
        args_tfidf['experiment_id'] = args['experiment_id']
    else:
        args_tfidf = {}

    print('Approaches: ', ', '.join([k['name'] for k in approaches]))

    for fold in tqdm(args['folds'], desc='Folds'):

        args['fold_name'] = fold['name']

        original_train_dict, original_test_dict = generate_folds_matrices(features, labels, fold, args_tfidf)

        for approach in tqdm(approaches, desc='Approaches'):

            args['approach_id'] = approach['approach_id']
            args['approach_name'] = approach['name']

            train_dict, test_dict = \
            apply_preprocessing(approach, original_train_dict, original_test_dict,
                                                        args)

            for hyperparameters in tqdm(generate_hyperparameters_combinations(approach['hyperparameters']), desc='Hyper'):

                
                args['hyperparameters'] = hyperparameters
                args = persist_learner(args)
                 
                try:
                    max_run_time(experiment['model_config']['max_seconds'])

                    mod = importlib.import_module(f"pipeline.approaches.{approach['python_path'][:-3]}")
                    model = mod.fit(args, train_dict=train_dict)

                    predictions = mod.predict(model, test_features=test_dict['features'])

                    evaluations = evaluate(obs=test_dict['labels'],
                                           pred=predictions,
                                           evaluation=experiment['evaluation'])

                    feature_importance = get_feature_importance(model, test_dict['features'])

                    persist_local(predictions, args, 'predictions', 
                                ['experiment_id', 'approach_id', 'learner_id', 'fold_name'])
                    persist_local(model, args, 'models', ['experiment_id', 'approach_id', 'learner_id'], '.p')
                    persist_evaluation(evaluations, args)
                    persist_feature_importance(feature_importance, args)

                except TimeoutError as error:
                    error = f'timeout < {experiment["model_config"]["max_seconds"]}'
                    persist_errors(error, args)

                    if experiment['model_config']['errors']:
                        raise 

                    continue
                
                except Exception as e:
                    persist_errors(e, args)
                    if experiment['model_config']['errors']:
                        raise 
                    continue


def run_experiment(experiment_file, testing=False):
    """
    Runs the experiment specified in the experiment yaml file to produce
    metrics for evaluation.

    It writes results on database and local system.

    Parameters
    ----------
    experiment_file: str
        Name of the experiment file inside experiment folder.
        Example: dummy_experiment.yaml
    """

    env = load_environment()

    args = initialise_experiment(experiment_file, env, testing)

    print('Experiment id', args['experiment_id'])

    args = create_model_configuration(args)

    loop_the_grid(args)


if __name__ == '__main__':
    # python src/pipeline/pipeline.py run_experiment --experiment_file='dummy_experiment.yaml'
    fire.Fire()
