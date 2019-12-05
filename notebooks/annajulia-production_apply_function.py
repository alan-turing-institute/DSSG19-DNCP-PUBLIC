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

# # Apply function for production

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath('annajulia-production_apply_function')).parent.parent/ 'src')
pipeline_path = str(Path(os.path.abspath('annajulia-production_apply_function')).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)
    
from pipeline.data_persistence import get_experiment, load_model, load_preprocessor_dill
from pipeline import preprocessing
from production.production import extract_minimum_information_tender, retrieve_features_from_individual, \
    apply_preprocessing, translate_features_to_human_language, predict_risk_probability, generate_prioritized_risk_list
from utils.utils import connect_to_database, retrieve_column_names_postgres

# Packages
import dill
import datetime as dt
import eli5
import glob
import logging
import pandas as pd
import pickle
import re
from tqdm import tqdm
pd.options.display.max_columns = 999
# -

# ### Start connection with db
#
#

con = connect_to_database()

# ## Example settings

# kNN (not working)
experiment_id = 1129688681
approach_id = 165975
learner_id = 9436986266

# +
# Experiment #1129688681

experiment_id = 1129688681

# Decision tree
#approach_id = 105436
#learner_id = 3063080495

# LightGBM 
approach_id = 101282
learner_id = 3242625979

# Decision tree (best learner within its experiment)
#approach_id = 105436
#learner_id = 2643795303
# -

experiment_id = 9893852173
approach_id = 104801
learner_id = 7871607723

# ## Apply function

query = """select id_llamado from semantic.tenders
    order by reception_date desc
    limit 10"""
tender_ids = [i for i in pd.read_sql_query(query, con)['id_llamado']]

# +
args={'experiment_id': experiment_id,
     'approach_id': approach_id,
     'learner_id': learner_id}

n_top_features = 3


prioritized_df, unique_risk_factors = generate_prioritized_risk_list(tenders_list=tender_ids, args=args, save_csv=False)
# -

prioritized_df.head(10)



# +
id_llamado = tender_ids[40]
minimum_info = extract_minimum_information_tender(id_llamado=id_llamado)

features = retrieve_features_from_individual(args, minimum_info_individual=minimum_info)

features, _ = apply_preprocessing(df=features, args=args)

"""
model = load_model(args)
print(model)

probability = model.predict_proba(features)[0][1]
print(probability)

important_feat =  eli5.explain_prediction(model, features,
                                             feature_names=[c for c in features.columns])
eli5.show_weights(model)
eli5.explain_prediction_df(model, features,
                                             feature_names=[c for c in features.columns])
"""

# +
#pd.DataFrame({'factor': sorted([r for r in unique_risk_factors])}).to_csv('factors.csv', index=False)
# -

# ### Read output

df = pd.read_csv(f'/data/shared_data/data/output/{dt.date.today()}_tenders_risk_{experiment_id}_{approach_id}_{learner_id}.csv')

df.head(10)

# ### Checks

ids_str = "', '".join([str(f) for f in tender_ids])
query = f"select * from semantic.labels where id_llamado in ('{ids_str}')"
labels = pd.read_sql_query(query, con)
labels.groupby('number_of_effective_complaints').count()
