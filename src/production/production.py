from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

import datetime as dt
import dill
import eli5
import fire
import logging
import numpy as np
import pandas as pd
import pickle
from pipeline.data_persistence import get_experiment, load_encoder, load_model, load_preprocessor_dill
from pipeline.config_loading import load_environment
from pipeline import preprocessing
import re
from tqdm import tqdm
from utils import utils
import sqlite3


production_path = load_environment()['production_path']

objects_main_column = {'tenders': 'id_llamado',
                       'tenders_agencies': 'convocante',
                       'tenders_agency_type': 'tipo_entidad',
                       'tenders_procurement_type': 'tipo_procedimiento_codigo',
                       'tenders_product_type': 'categoria',
                       'tenders_service_type': '_objeto_licitacion'}





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
    inflation = pd.read_sql_query('select * from inflation', con)
    exchange_rate = pd.read_sql_query('select * from exchange_rate', con)

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
        query = f"select * from {tab}"
        partial_features = pd.read_sql_query(query, con)
        features_df = pd.merge(features_df, partial_features, how='left')

    # Subset the variables that are included in the experiment
    features_df = features_df[args['features']]

    print(features_df[features_df.isnull().any(axis=1)])
    return features_df


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


def generate_prioritized_risk_list(data_path, n_top_features=3, return_df=True, save_csv=True, output_folder=''):
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
    return_df: bool
        Return final DataFrame as object.
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
    args = pickle.load(open(f"{production_path}best_model_args.p", 'rb'))

    # Connect to DB
    # con = utils.connect_to_database()
    con = sqlite3.connect(production_path + 'features.db')

    # Load data
    original_df = pd.read_csv(data_path)

    # Read data and prepare it to start the ML pipeline
    features_df = generate_features_from_new_data(original_df, args, con)

    # Load encoders (model and preprocessor)
    model = load_encoder(args, type='model', path=production_path)
    preprocessor = load_encoder(args, type='preprocessing', path=production_path)

    # Apply preprocessing
    features, _ = preprocessing.run(preprocessors=list(preprocessor), data=features_df, preprocessing=preprocessor, fit=False)

    print(features.head())

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
        csv_df.to_csv(f'{output_folder}{dt.date.today()}_prioritized_tenders.csv', index=False, header=True)

    if return_df:
        return csv_df


if __name__ == '__main__':
    # python src/production/production.py generate_prioritized_risk_list --data_path='test.csv' --return_df=False
    fire.Fire()
