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

# # Apply production function

# ## Settings

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath('annajulia-production-apply-function')).parent.parent/ 'src')
pipeline_path = str(Path(os.path.abspath('annajulia-production-apply-function')).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)
    
from pipeline.data_persistence import get_experiment, load_model, load_preprocessor_dill
from pipeline import preprocessing
from production.production import extract_minimum_information_tender, retrieve_features_from_individual, \
    apply_preprocessing, translate_features_to_human_language, predict_risk_probability, generate_prioritized_risk_list
from utils.utils import connect_to_database, retrieve_column_names_postgres, open_yaml

# Packages
import dill
import datetime as dt
import eli5
import glob
import logging
import numpy as np
import pandas as pd
import pickle
import re
from tqdm import tqdm
import yaml 
pd.options.display.max_columns = 999
# -

con = connect_to_database()

data_path = '/home/jian.wen/Documents/wen/test.csv'
data_path = '/home/anna.verdaguer/DSSG19-DNCP/test.csv'
production_path = '/data/production/'


"""
### Requirements from the ML Pipeline

1. Save the model encoder and the preprocessing encoder into the production path

    * Note that the model pickle should be in the format model_{experiment_id}_{approach_id}_{learner_id}.p
        ** Note the prefix model_
    * Note that the preprocessing picke should be in the format prepro_{experiment_id}_{approach_id}_{fold}.dill
        
2. args is a dictionary that contains the following entries:
    'features': a list of all the column names used in the experiment
    'experiment_id': experiment id
    'approach_id': approach id
    'learner_id': learner id
    'fold': aod of the fold trained on
    'k': % of list to filter for (optimised by the model)
"""

experiment_id = 9893852173
approach_id = 104801
learner_id = 3181193082
fold = '2017-12-26'
k = 0.3

#parameter in the apply function to specify how many feature importance variables
n_top_features = 3

exp = open_yaml('/home/anna.verdaguer/DSSG19-DNCP/experiments/20190820-notxt-Pablo.yaml')
col_names = []
for elem in exp['features']:
    col_names = col_names + elem['columns']

args = {'experiment_id': experiment_id,
       'approach_id': approach_id,
       'learner_id': learner_id,
       'fold': fold,
       'features': col_names,
       'k': k}

args = pickle.load(open(f"{production_path}/best_model_args.p", 'rb'))


def clean_data_for_production(df, con):
    """
    Given the minimum information of new tenders:
    - Generate date columns from reception_date
    - Adjust tender value by applying an exchange rate to dollars and an inflation rate.

    Parameters
    -----------
    df: DataFrame
        Pandas DataFrame with the minimum information needed for tenders to be predicted as risky or not.
    con: sqlalchemy.engine.base.Engine
        slalchemy connection

    Returns
    -------
    DataFrame with the needed features for the model.
    """
    
    # Define column names
    col_names = list(df.columns)
    
    # Load supporting tables
    inflation = pd.read_sql_query('select * from production.inflation', con)
    exchange_rate = pd.read_sql_query('select * from production.exchange_rate', con)
    
    # Adjust dates and add variables 
    df['reception_date']= pd.to_datetime(df['reception_date'])
    df['reception_month'] = df['reception_date'].dt.month
    df['reception_day'] = df['reception_date'].dt.day
    df['reception_year'] = df['reception_date'].dt.year
    
    # Adjust tender value
    df = df.merge(inflation, how='left', left_on='reception_year', right_on='id_year')
    df = df.merge(exchange_rate, how='left', left_on='reception_year', right_on='id_year')
    df['log_amt_planned_real_guarani'] = np.where(df['_moneda']=='PYG', np.log(df['monto_global_estimado_planificacion']/(df['inflation_index']/100)), 
                                                  np.log(df['monto_global_estimado_planificacion']*df['exchange_usd_guarani']/(df['inflation_index']/100)))
    
    # Keep only the columns that we need
    for col in ['log_amt_planned_real_guarani', 'reception_month', 'reception_day']:
        col_names.append(col)
        
    # Define final clean DataFrame
    clean_df = df[col_names].drop(['monto_global_estimado_planificacion', '_moneda', 'reception_date'], axis=1)
    
    return clean_df


def generate_features_from_new_data(new_df, args, con):
    """
    Given the minimum information of new tenders, clean it and generate the corresponding needed columns.

    Parameters
    -----------
    new_df: DataFrame
        Pandas DataFrame with the minimum information needed for tenders to be predicted as risky or not.
    args: dict
        Best model parameters. It should, at least, include a 'features' section with the corresponding features needed to run the model.
    con: sqlalchemy.engine.base.Engine
        slalchemy connection

    Returns
    -------
    DataFrame with the needed features for the model.
    """
    
    # Load and clean data
    df = new_df.copy()
    df = clean_data_for_production(df, con)
    
    # Various tables to loop through in the production schema
    tab_names = ['agencies', 'agency_type', 'procurement_type', 'service_type', 'product_type']
    
    # Generate all the features
    features_df = df.copy()
    for tab in tab_names:
        query = f"select * from production.{tab}"
        partial_features = pd.read_sql_query(query, con)
        partial_features = partial_features.drop(columns=['date', 'id_llamado'])
        features_df = pd.merge(features_df, partial_features, how='left')
    
    # Subset the variables that are included in the experiment
    features_df = features_df[args['features']]
    return features_df


def load_encoder(args, type):
    """
    Loads a pickle containing a learner.

    Parameters
    -----------
    args: dict
        Arguments dictionary containing, at least, experiment_id, approach_id and learner_id.
    type: string
        Type of encoder to be loader. Available types: 'model', 'preprocessing'.

    Returns
    -------
    Encoder. It can be:
        - Fit model (e.g. LGBMClassifer).
        - Preprocessing encoder (e.g. OneHotEncoder).
    """
    
    prefixes = {'model': 'model',
                'preprocessing': 'prepro'}
    
    experiment_id = args['experiment_id']
    approach_id = args['approach_id']
    learner_id = args['learner_id']
    fold = args['fold']
    
    if type == 'model':
        path =  f'{production_path}/{prefixes[type]}_{experiment_id}_{approach_id}_{learner_id}.p'
        encoder = pickle.load(open(path, 'rb'))['chosen_model']
    
    elif type == 'preprocessing':
        # TODO: This loads the preprocessing encoder from the last fold. Check what should happen in real production.
        path =  f'{production_path}/{prefixes[type]}_{experiment_id}_{approach_id}_{fold}.dill'
        encoder = dill.load(open(path, 'rb'))
    else:
        logging.error(f'Type {type} not defined')
        
    return encoder


def translate_features_to_human_language(feature_name):
    """
    Given a feature name, transformed it into a more understandable text.
    It can translate any feature name belonging to object features (e.g. 'tp_n_tenders', 'te_n_eff_complaints', etc.).
    For features belonging to the tender itself, it returns the same name translated to English referring to tender.

    Parameters
    -----------
    feature_name: string
        Feature name.

    Returns
    -------
    Translated text in the format "what on who" (e.g. 'tp_n_tenders' -> 'tenders on type of procurement').
    """

    object_dict = {'tp': 'type of procurement',
                   'ts': 'type of service',
                   'tc': 'product category',
                   'te': 'type of agency',
                   '': 'agency'}
    
    tenders_dict = {'log_amt_planned_real_guarani': 'value',
                    'tipo_procedimiento_codigo': 'procurement type',
                    'tipo_entidad': 'agency type',
                    '_objeto_licitacion': 'service type',
                    'reception_month': 'month',
                    'reception_day': 'day'}
    
    if feature_name.lower() == '<bias>':
        return 'no meaningful variables'
    
    else:
        object_prefixes = '|'.join([k for k in object_dict.keys()])
        what_prefixes = '|'.join(['complaints', 'tenders', 'value', 'bidders', 'products', 'high_risk'])

        regex = f"^(?P<object>({object_prefixes})*).*(?P<what>({what_prefixes})+)(_data|_bool)*_*(?P<regr>([0-9](m|y))*)$"

        x = re.search(regex, feature_name)
        try:
            who = object_dict[x.group('object')]
            what = x.group('what')
            
            if what == 'high_risk':
                what = 'high risk products'
        except:
            who = 'tender'
            what = tenders_dict[feature_name]
    return f"{what} on {who}"


def get_feature_importance(model, features, n_top_features=3):
    """
    Given a model and a features DataFrame with the different observations, calculate the top important features 
    for each individual prediction.

    Parameters
    -----------
    model: model fit
        Fit of the model wanted to analyze (e.g. LGBMClassifer).
    features: DataFrame
        Pandas DataFrame with the input matrix used for the model (rows are different observations and columns are features).
    n_top_features: int
        Number of top features wanted. By default, the top 3 features will be displayed.

    Returns
    -------
    Translated text in the format "what on who" (e.g. 'tp_n_tenders' -> 'tenders on type of procurement').
    """
    
    features_list = []
    
    for index, row in features.iterrows():
        important_features =  eli5.explain_prediction_df(model, row, 
                                    feature_names=[c for c in features.columns])['feature'][:n_top_features]
        features_list = features_list + [list(map(translate_features_to_human_language, [f for f in important_features]))]
    
    return features_list


def generate_prioritized_risk_list(data_path, n_top_features=3, save_csv=True, output_folder=''):
    """
    Given a csv file path containing tenders minimum information, prioritize them according to their risk
    and flag those that should be reviewing thoroughly (flagged as quality_review=1).

    Parameters
    -----------
    data_path: string
        Complete path of the csv file where the tenders information is. 
        The minimum information required is: 
            - id_llamado
            - convocante
            - tipo_entidad
            - tipo_procedimiento_codigo
            - categoria
            - _objeto_licitacion
            - monto_global_estimado_planificacion
            - _moneda
            - reception_date
    n_top_features: int
        Number of top features wanted. By default, the top 3 features will be displayed.
    save_csv: bool
        Save output to csv. By default, the output csv will be saved.
    output_folder: string
        Folder where to output the csv file. By default, the root path.

    Returns
    -------
    csv_df: DataFrame
        Prioritized dataframe.
    """
    
    # Load arguments of the best model
    args = pickle.load(open(f"{production_path}/best_model_args.p", 'rb'))
    
    # Connect to DB
    con = connect_to_database()
    
    # Load data
    original_df = pd.read_csv(data_path)
    
    # Read data and prepare it to start the ML pipeline
    features_df = generate_features_from_new_data(original_df, args, con)
    
    # Load encoders (model and preprocessor)
    model = load_encoder(args, type='model')
    preprocessor = load_encoder(args, type='preprocessing')
    
    # Apply preprocessing
    features, _ = preprocessing.run(preprocessors=list(preprocessor), data=features_df, preprocessing=preprocessor, fit=False)
    
    # Predict risk scores
    scores = [p[1] for p in model.predict_proba(features)]

    # Get important features for each individual tender
    features_list = get_feature_importance(model, features, n_top_features)
    
    # Define final output csv file
    csv_df = original_df.copy()
    csv_df['complaint_risk'] = scores
    csv_df = pd.concat([csv_df, pd.DataFrame(features_list).add_prefix('risk_factor_')], axis=1)
    
    # Sort by risk score
    csv_df = csv_df.sort_values(by=['complaint_risk'], ascending=False).drop(columns=['complaint_risk']).reset_index(drop=True)
    
    # Flag quality reviews, according to best model k%
    n_flags = max(1, round(args['k']*csv_df.shape[0]))
    csv_df['quality_review'] = [1]*n_flags + [0]*(csv_df.shape[0]-n_flags)
    
    if save_csv:
        csv_df.to_csv(f'{output_folder}/{dt.date.today()}_prioritized_tenders.csv', index=False, header=True)
    
    return csv_df


generate_prioritized_risk_list(data_path)


