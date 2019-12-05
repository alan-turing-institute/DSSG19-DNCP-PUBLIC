# +
'''
Given an object, save it in disk or database.
    Input:
        Object (DataFrame, model, etc.).
    Output:
        None. Object is saved.
'''
from utils import utils
import dill
import subprocess
import pickle
import pyarrow.parquet as pq
import pyarrow as pa
import datetime
import hashlib
import pandas as pd
import sqlalchemy
from glob import glob
import random
from pathlib import Path
import os
import sys
source_path = Path(os.path.abspath(__file__)).parent.parent
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, str(source_path))

from pipeline.config_loading import load_experiment, load_environment

# +
def generate_id(content, length=10):
    """
    Generate an unique id.

    Parameters
    ----------
    content: object
        Any object that can be transformed in a string

    length: int (default 10)
        Length of the id

    Return
    ------
    int
        Unique id
    """

    strg = str(content)

    m = hashlib.sha256()
    m.update(bytes(strg, 'utf8'))

    i = int.from_bytes(m.digest(), byteorder='big')

    unique_id = int(str(i)[:length])

    return unique_id


def get_data_from_db(query, as_pandas=False):
    """
    Gets query result from database

    Parameters
    ----------
    query: string
        SQL query

    as_pandas: bool
        Either if you want the result as a pd.DataFrame

    Return
    ------
    sqlalchemy.engine.result.ResultProxy or pd.DataFrame
        Content of the query
    """

    with utils.connect_to_database().connect() as con:

        result = con.execute(query)

    if as_pandas:

        return pd.DataFrame(result, columns=result.keys())

    return result


def insert_to_db(data, schema, table, how='append'):
    """
    Interts dictionary as row in the corresponding table in the database

    Parameters
    ----------
    data: dictonary
        Dict with keys being the columns of the table

    schema: string
        Schema name

    table: string
        Table name

    TODO: how: string

    Returns
    -------
    dictonary
        Status of the task
    """

    engine = utils.connect_to_database()
    con = engine.connect()

    metadata = sqlalchemy.MetaData(bind=con, schema=schema)
    metadata.reflect()

    if isinstance(data, dict):
        data = [data]



    # insertions = []
    # for datum in data:
    #     insertions.append(metadata.tables[f'{schema}.{table}'].insert().values(
    #         datum
    #     ))

    try:
        metadata.tables[f'{schema}.{table}'].insert().execute(data)

    except sqlalchemy.exc.IntegrityError:
        pass

    con.close()
    engine.dispose()

    # return {'Status': 200}


def get_experiment(experiment_id):
    """
    Get experiment dictionary from database given an experiment_id

    Parameters
    ----------
    experiment_id: integer
        Unique identifier of the experiment

    Return
    ------
    dictionary
        Experiment parameters dictionary, the yaml content column
    """

    query = f"""
        select yaml_content
        from experiments.experiments
        where experiment_id = {experiment_id}
    """

    row = get_data_from_db(query)

    experiment = dict(zip(row.keys(), list(row)[0]))['yaml_content']

    return experiment


def get_approaches(experiment_id):
    """
    Get approaches ids and hyparameters given an experiment_id

    Parameters
    ----------
    experiment_id: integer
        Unique identifier of the experiment

    Return
    ------
    list of dictionaries
        experiment_id, approach_id,  python_path, hyperparameters of each approach
    """

    query = f"""
    select distinct on (name) name, experiment_id, approach_id, python_path, hyperparameters, preprocessors
    from (
    	select
    	  experiment_id, approach_id, name, python_path, hyperparameters, preprocessors,
    	  row_number() over (partition by approach_id order by created_on desc) rown
    	from experiments.approaches
    	where experiment_id = {experiment_id}) a
    where rown = 1
    """

    table = get_data_from_db(query)

    result = [dict(zip(row.keys(), list(row))) for row in table]

    return result


def persist_local(data, args, folder, 
                    id_keys=['experiment_id'], 
                    as_type='.parquet.gz',
                    save_path='persistance'):

    save_path = build_path(args, folder, id_keys, as_type, save_path)

    # Check if path exists, add path if not
    if not save_path.parent.exists():
        save_path.parent.mkdir(mode=777, parents=True, exist_ok=True)

        subprocess.run(['chmod', '-R', '777', f'{save_path.parent}'])

    if as_type == '.parquet.gz':
        df = pa.Table.from_pandas(data)
        pq.write_table(df, save_path, compression='gzip')

    elif as_type == '.p':
        pickle.dump(data, open(save_path, 'wb'))
    elif as_type == '.dill':
        dill.dump(data, open(save_path, 'wb'))
    elif as_type == '.png':
        data.savefig(save_path)
    else:
        raise 'This file type cannot be saved '

    subprocess.run(['chmod', '777', f'{save_path}'])


def build_path(args, folder, id_keys=['experiment_id'], as_type='.parquet.gz', save_path='persistance'):

    if save_path == 'persistance':

        save_path = load_environment['persistance_path']

    path_id = '_'.join([str(args[key]) for key in id_keys])

    # If folder == Nolsne, it does not use it to build path
    return Path(os.path.join(*filter(lambda x: x is not None, 
                        [save_path, folder, path_id + as_type])))




def check_if_local_exists(args, folder, id_keys=[
                          'experiment_id'], 
                          as_type='.parquet.gz',
                          save_path='persistance'):

    return build_path(args, folder, id_keys, as_type, save_path).exists()


def get_local(args, folder, id_keys=['experiment_id'], as_type='.parquet.gz',
             save_path='persistance'):

    save_path = build_path(args, folder, id_keys, as_type, save_path)

    if as_type == '.parquet.gz':
        return pq.read_table(save_path).to_pandas()

    elif as_type == '.p':
        return pickle.load(open(save_path, 'rb'))
    
    elif as_type == '.dill':
        return dill.load(open(save_path, 'rb'))



def get_data_dict(args):

    return {'labels': get_local(args, 'labels'),
            'features': get_local(args, 'features')
            }


def persist_learner(args):
    """Persists arguments to learner table

    Parameters
    ----------
    args: dict
        General arguments

    Returns
    -------
    dict
        Updated general arguments
    """

    args['learner_id'] = generate_id(
        ''.join(map(str, [
            args['experiment_id'],
            args['approach_id'],
            args['hyperparameters']])), 10)

    learner_row = {
        'experiment_id': args['experiment_id'],
        'approach_id': args['approach_id'],
        'learner_id': args['learner_id'],
        'hyperparameters_used': args['hyperparameters']
    }

    insert_to_db(learner_row, schema='experiments', table='learners')

    return args


def persist_evaluation(evaluations, args):
    """
    Persists arguments to learner table

    Parameters
    ----------
    evaluations: dict
        Evaluations strucuted as {eval_metric: score}
    args: dict
        General arguments

    Returns
    -------
    dict
        Updated general arguments
    """

    evaluation_rows = []
    for evaluation in evaluations:

        evaluation_rows.append({
            'experiment_id': args['experiment_id'],
            'approach_id': args['approach_id'],
            'learner_id': args['learner_id'],
            'fold': args['fold_name'],
            'hyperparameters_used': args['hyperparameters'],
            'eval_metric': evaluation['name'],
            'score': evaluation['value']
        })

    insert_to_db(evaluation_rows, schema='experiments', table='evaluations')

    return args


def persist_feature_importance(features, args):
    """
    Persists arguments to learner table

    Parameters
    ----------
    features: dict
        Feature importance strucuted as {feature: importance}
    args: dict
        General arguments

    Returns
    -------
    dict
        Updated general arguments
    """

    for feature in features:

        evaluation_row = {
            'experiment_id': args['experiment_id'],
            'approach_id': args['approach_id'],
            'learner_id': args['learner_id'],
            'model_name': args['approach_name'],
            'fold': args['fold_name'],
            'hyperparameters_used': args['hyperparameters'],
            'feature': feature['feature'],
            'importance': feature['weight']
        }

        insert_to_db(
            evaluation_row,
            schema='experiments',
            table='feature_importances')

    return args


def persist_errors(error, args):
    """
    Persists arguments to learner table

    Parameters
    ----------
    features: dict
        Feature importance strucuted as {feature: importance}
    args: dict
        General arguments

    Returns
    -------
    dict
        Updated general arguments
    """

    error_row = {
        'experiment_id': args['experiment_id'],
        'approach_id': args['approach_id'],
        'learner_id': args['learner_id'],
        'model_name': args['approach_name'],
        'fold': args['fold_name'],
        'hyperparameters_used': args['hyperparameters'],
        'error': str(error),
    }

    insert_to_db(error_row, schema='experiments', table='errors')

    return args


def persist_experiment(experiment_path):
    """ Saves experiment metadata on experimets table.

    It does not duplicate entrances.

    Parameters
    ----------
    experiment : str
        Path of experiment file

    Returns
    -------
    dict
        Experiment parameters
    """
    experiment_row = {}

    experiment_row['yaml_file_name'] = Path(experiment_path).name

    experiment_row['yaml_content'] = load_experiment(experiment_path)

    experiment_row['experiment_id'] = generate_id(
        experiment_row['yaml_content'])

    features_sql_path = source_path.parent / 'sql' / 'semantic'

    experiment_row['sql_features'] = \
        {sql_file.name: sql_file.open('r').read()
            for sql_file in features_sql_path.glob('*.sql')}

    features_sql_path = source_path.parent / 'sql' / 'cleaned'

    experiment_row['sql_cleaned'] = \
        {sql_file.name: sql_file.open('r').read()
            for sql_file in features_sql_path.glob('*.sql')}

    insert_to_db(experiment_row, schema='experiments', table='experiments')

    return experiment_row


def persist_approaches(experiment_row):
    """ Saves approaches to approaches table.

    Parameters
    ----------
    experiment_row : dict
        Parameters of the experiment
    """

    approaches_rows = []

    for i, approach in enumerate(experiment_row['yaml_content']['approaches']):

        python_content = (
            source_path /
            'pipeline' /
            'approaches' /
            approach['file_path']).open('r').read()

        approaches_rows.append(
            {
                'experiment_id': experiment_row['experiment_id'],
                'approach_id': generate_id(python_content + approach['approach_name'], 6),
                'name': approach['approach_name'],
                'hyperparameters': approach['hyperparameters'],
                'python_path': approach['file_path'],
                'python_content': python_content,
                'preprocessors': approach['preprocessors']
            }
        )

    for row in approaches_rows:
        insert_to_db(row, schema='experiments', table='approaches')


def initialise_experiment(experiment_path, env, testing=False):
    """
    Saves all initial data related to experiment on database.

    It saves the yaml, features sql, experiment_id and experiment name in
    experiment.experiment.

    It saves the approaches in experiment.approaches with experiment_id, model_id,
    approach file as string and hyperparameters.

    Parameters
    ----------
    experiment_path: str
        Path to experiment
    """

    experiment_row = persist_experiment(experiment_path)

    persist_approaches(experiment_row)

    return {'experiment_id': experiment_row['experiment_id'],
            'testing': testing,
            'persist_path': env['persist_path']}


def load_model(args):
    """
    Loads a pickle containing a learner.

    Parameters
    -----------
    args: dict
        Arguments dictionary containing, at least, experiment_id, approach_id and learner_id.

    Returns
    -------
    Fit model (e.g. LGBMClassifer).
    """

    experiment_id = args['experiment_id']
    approach_id = args['approach_id']
    learner_id = args['learner_id']

    path =  f'/data/persist/models/{experiment_id}_{approach_id}_{learner_id}.p'
    return pickle.load(open(path, 'rb'))['chosen_model']


def load_preprocessor_dill(args):
    """
    Loads a dill containing a preprocessing.

    Parameters
    -----------
    args: dict
        Arguments dictionary containing, at least, experiment_id and approach_id.

    Returns
    -------
    Preprocessing dill object.
    """

    experiment_id = args['experiment_id']
    approach_id = args['approach_id']

    path =  f'/data/persist/preprocessing/{experiment_id}_{approach_id}_*.dill'

    # TODO: This loads the preprocessing encoder from the last fold. Check what should happen in real production.
    return dill.load(open(sorted([p for p in glob(path)])[-1], 'rb'))


def load_encoder(args, type, path):
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
        path =  f'{path}/{prefixes[type]}_{experiment_id}_{approach_id}_{learner_id}.p'
        encoder = pickle.load(open(path, 'rb'))['chosen_model']

    elif type == 'preprocessing':
        # TODO: This loads the preprocessing encoder from the last fold. Check what should happen in real production.
        path =  f'{path}/{prefixes[type]}_{experiment_id}_{approach_id}_{fold}.dill'
        encoder = dill.load(open(path, 'rb'))
    else:
        logging.error(f'Type {type} not defined')

    return encoder


if __name__ == '__main__':

    # experiment_name = 'dummy_experiment'
    # experiment = f'/home/joao.carabetta/Documents/dncp/experiments/{experiment_name}.yaml'
    #
    # args = initialise_experiment(experiment)

    add_learner_row(123, 23435, {'test': 'test'})
