from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
if source_path not in sys.path:
    sys.path.insert(0, source_path)

import pandas as pd
from sqlalchemy import create_engine
import os
import yaml
from os.path import expanduser
from pathlib import Path
from datetime import datetime
import logging

import signal, time, random

class TimeoutError(RuntimeError):
    pass

def handler(signum, frame):
    raise TimeoutError()

def max_run_time(seconds):
    signal.signal(signal.SIGALRM, handler)
    return signal.alarm(seconds)

def connect_to_database():
    """
    Connects to postgre database using .pgpass

    Return
    ------
    sqlalchemy.engine.base.Engine
        database engine / connection
    """

    env_dict = open_yaml(Path(os.path.abspath(__file__)).parent.parent.parent / 'env.yaml')

    host, port, database, user, password = env_dict['database'].replace('\n', '').split(':')

    database_uri = 'postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, database)

    return create_engine(database_uri)

def open_yaml(path):
    """
    Load yaml file.

    Parameters
    ----------
    path: pathlib.PosixPath
        Path to yaml file

    Return
    ------
    Dictionary
        Dictionary with yaml file content
    """
    with open(path) as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error('Error when opening YAML file.', exc_info=1)

    return yaml_dict


def path_to_shared(username,
                   directory,
                   filename,
                   filetype
                 ):
    """
    Generate path to shared folder at '/data/shared_data'.

    All files will be with the following name:

    `[username]_[filename]_[timestamp].[filetype]`

    Timestamp has the pattern '%Y%m%d%H%M%S'

    The folder is structured as:

    -- imgs
    -- notebooks
    -- data
       -- raw
       -- treated
       -- output


    Example
    -------

    ```
    import utils

    df = pd.read_csv('some_csv')
    df.to_csv(utils.path_to_shared('joaocarabetta','data_output', 'filename', 'csv'))
    ```

    Parameters
    ----------

    username: str
        your username. try not to change it

    directory: str
        where you want to save the file.
        the available options are: imgs, data_output, data_raw, data_treated, notebooks

    filename: str
        the name of the file

    filetype: str
        the type of the file, e.g. csv, png, pdf ...
        do not use a dot in the beggining.
    """

    data_path = '/data/shared_data'

    dir_to_path = {
        'imgs': 'imgs',
        'data_output': 'data/output',
        'data_raw': 'data/raw',
        'data_treated': 'data/treated',
        'notebooks': 'notebooks'
    }


    try:
        dir_path = dir_to_path[directory]

    except KeyError:

        print(f'You entered {directory} as a directory. But, the only available directories are: {", ".join(dir_to_path.keys())}')
        raise KeyError

    timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

    file_path = str(Path('/data/shared_data') / dir_path /  f'{username}_{filename}_{timestamp}.{filetype}')

    open(file_path, 'w+').write(' ')
    os.chmod(file_path, 0o774) # change permission to 774

    return file_path


def calculate_bool_stats(df, groupby_col, bool_col='bool_of_effective_complaints', count_col='id_llamado'):
    """
    Given a data frame with a group column, a boolean column and a value column,
    calculates stats by group:
        - Total value (bool0 + bool1)
        - Rate of bool1 over total for each group
        - Percentage of bool1 over complete df
        - Cumulative percentage over complete df

    Parameters
    ----------
    df: pandas.DataFrame
        Table schema
    groupby_col: string
        Column name for which to group by
    bool_col: string
        Boolean column
    count_col: string
        Numeric column over which calculate stats

    Return
    -------
    pandas.DataFrame
        Stats table
    """

    groupby_df = df.groupby([groupby_col, bool_col])[count_col].nunique().to_frame().reset_index()

    stats_df = groupby_df.pivot(index=groupby_col, columns=bool_col, values=count_col)\
    .fillna(0).reset_index().rename(index=str, columns={0: 'n_bool_0', 1: 'n_bool_1'})
    stats_df.columns.name = None

    stats_df['n_total'] = stats_df['n_bool_1'] + stats_df['n_bool_0']
    stats_df['rate'] = stats_df['n_bool_1'] / stats_df['n_total'] * 100
    stats_df['percentage'] = stats_df['n_bool_1'] / stats_df['n_bool_1'].sum() * 100

    stats_df = stats_df.sort_values('percentage', ascending=False)
    stats_df['cum_percentage'] = stats_df['percentage'].cumsum()

    return stats_df

def check_null_values(schema, table, con):
    """
    Reads through the table to count null values

    Parameters
    ----------
    schema: string
        Table schema
    table: string
        Table name
    con: sqlalchemy.engine.base.Engine
        slalchemy connection

    Return
    -------
    pandas.DataFrame
        Number of nulls per columns
    """

    query = f'select * from {schema}.{table} limit 1'

    columns = list(pd.read_sql_query(query, con).columns)

    results = []

    for column in columns:

        duplicate_query = f"""
        select count(*) - count({column})
        from {schema}.{table}"""

        results.append({
            'column': column,
            'nulls': pd.read_sql_query(duplicate_query, con).values[0][0]
        })

    return pd.DataFrame(results)


def load_pandas_to_db(df, table, schema, how='append'):
    """
    Load pandas DataFrame into database.

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame to load.
    table: string
        Name of the target table to upload data.
    schema: string
        Name of the schema where the target table is.
    how: string
        In case the table already exists, what should happen: 'fail', 'replace', 'append' (default).
    """

    con = connect_to_database()
    try:
        df.to_sql(name=table, schema=schema, con=con, if_exists=how, index=False)
        logging.info("Table loaded successfully!")
    except Exception as e:
        logging.error('Table could not be loaded', exc_info=True)


def retrieve_column_names_postgres(schema, table):
    """
    Retrieve column names of a PostgreSQL table.

    Parameters
    -----------
    schema: string
        Schema name.

    table: string
        Table name.

    Returns
    -------
    List of the column names.
    """

    con = connect_to_database()

    query = f"select column_name " \
            f" from information_schema.columns " \
            f" where table_schema = '{schema}' " \
            f" and table_name = '{table}'"
    column_names = pd.read_sql_query(query, con)

    return column_names['column_name'].tolist()


def split_yaml(yaml_path, output_folder, split_by):
    """
    Split a YAML file into multiple files by some key.
    Split files are saved into an output folder.

    Parameters
    -----------
    yaml_path: string
        Complete path of the YAML file.
    output_folder: string
        Folder name where to output the splits.
    split_by: string
        Key to use for splitting.
    """

    if type(yaml_path) == str:
        file_name = yaml_path.split("/")[-1].replace('.yaml', '')
    else:
        file_name = yaml_path.stem

    yaml_content = open_yaml(yaml_path)

    for i, approach in enumerate(yaml_content[split_by]):
        output_dict = yaml_content.copy()
        output_dict[split_by] = [approach]

        with open(f"{output_folder}/{file_name}_{i+1}.yaml", 'w') as outfile:
            yaml.dump(output_dict, outfile, default_flow_style=False)


if __name__ == '__main__':
    connect_to_database()
