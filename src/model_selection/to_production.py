from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

from pipeline.data_persistence import get_local, persist_local
from pipeline.config_loading import load_environment
from utils.utils import connect_to_database
import sqlalchemy
import sqlite3
import pandas as pd

def get_ids(final_selected_model, results, selected_learners):
    
    selected_learner_id = results[final_selected_model]['learners'].index[0]
    return selected_learners.query(f'learner_id == {selected_learner_id}').iloc[0]

def get_features(ids):

    return list(get_local({'experiment_id': ids['experiment_id']}, 'features').set_index('id_llamado').columns)

def save_args(production_path, ids, max_fold, k):
    args = {'experiment_id': ids['experiment_id'],
               'approach_id': ids['approach_id'],
               'learner_id': ids['learner_id'],
               'fold': max_fold,
               'features': get_features(ids),
               'k': k}

    persist_local(args,
                  {'name': 'best_model_args'},
                  folder=None,
                  id_keys=['name'],
                  as_type='.p',
                  save_path=production_path)

def save_preprocessor(production_path, ids, max_fold):

    content = get_local({**ids.to_dict(), **{'fold': max_fold}}, 
                     folder='preprocessing', 
                     id_keys=['experiment_id', 'approach_id', 'fold'], 
                     as_type='.dill')


    persist_local(content,
                  args={**ids.to_dict(), **{'fold': max_fold, 'preffix': 'prepro'}},
                  folder=None,
                  id_keys=['preffix', 'experiment_id', 'approach_id', 'fold'],
                  as_type='.dill',
                  save_path=production_path)


def save_model(production_path, ids):
    
    model = get_local(ids.to_dict(), 
                     folder='models', 
                     id_keys=['experiment_id', 'approach_id', 'learner_id'], 
                     as_type='.p')


    persist_local(model,
                  args={**ids.to_dict(), **{'preffix': 'model'}},
                  folder=None,
                  id_keys=['preffix', 'experiment_id', 'approach_id', 'learner_id'],
                  as_type='.p',
                  save_path=production_path)

def create_production_schema_postgresql():

    to_production_sql = Path(os.path.abspath('joaoc-experiment-checks')).parent.parent.parent / 'sql' / 'production' / 'create-production-schema.sql'

    with connect_to_database().connect() as con:
        for query in to_production_sql.open('r').read().split(';')[:-1]:
            res = con.execute(query)
            
def copy_production_schema_to_sqlite(production_path):
    
    con_sqlite = sqlite3.connect(production_path + 'features.db')
    
    with connect_to_database().connect() as con:
    
        metadata = sqlalchemy.MetaData(bind=con, schema='production')
        metadata.reflect()

    for schema, table in map(lambda x: x.split('.'), metadata.tables.keys()):

        content = pd.read_sql_table(table, schema=schema, con=connect_to_database())
        content.to_sql(table,  con=con_sqlite, if_exists='replace', index=False)
    
def prepare_to_production(final_selected_model, results, selected_learners, k, max_fold):
    
    production_path = load_environment()['production_path']
    persistance_path = load_environment()['persistance_path']

    # Erase files
    for f in Path(production_path).glob('*'):
        os.remove(f)

    ids = get_ids(final_selected_model, results, selected_learners)
    save_model(production_path, ids)
    save_preprocessor(production_path, ids, max_fold)
    save_args(production_path, ids, max_fold, k)

    create_production_schema_postgresql()
    copy_production_schema_to_sqlite(production_path)