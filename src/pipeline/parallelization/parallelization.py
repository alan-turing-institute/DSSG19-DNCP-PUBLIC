'''
Parallelize big experiments by splitting them into smaller pieces.
'''

from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent.parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

from pipeline.config_loading import load_environment, load_experiment
from pipeline.data_persistence import generate_id, initialise_experiment, insert_to_db
from utils.utils import open_yaml, split_yaml

import fire
import logging
import shutil
import subprocess
import yaml


parallel_experiments_folder = 'parallelization'


def initialise_parallel_experiments(experiment_file, experiment_folder='experiments', testing=False):
    '''
    Splits macro experiment into smaller pieces and initialise each of them as an independent
    experiment to be run.
    Relation macro experiment - smaller experiments is loaded into the database (experiments.parallelization).

    Parameters
    ----------
    experiment_file: string
        Name of the macro experiment ran in parallel. E.g. 'dummy_experiment.yaml'
    experiment_folder: string
        Name of the folder where the macro experiment is saved.
    testing: boolean
        Not implemented.
    '''

    env = load_environment()

    experiment_path = Path(experiment_folder) / experiment_file

    # Generate ID of the macro experiment
    macro_id = generate_id(load_experiment(experiment_file))

    # Define output folder for experiment splits
    output_folder = Path(experiment_folder) / parallel_experiments_folder / experiment_file.replace('.yaml', '')

    # Make sure folders exist. If not, create them.
    if not output_folder.parent.exists():
        output_folder.parent.mkdir(mode=777, parents=True, exist_ok=True)
        subprocess.run(['chmod', '-R', '777', f'{output_folder.parent}'])

    if not output_folder.exists():
        output_folder.mkdir(mode=777, parents=True, exist_ok=True)
        subprocess.run(['chmod', '-R', '777', f'{output_folder}'])

    # Split yaml file and save all parts
    split_yaml(yaml_path=experiment_path, output_folder=output_folder, split_by='approaches')

    # Initialize split experiments
    for experiment_split in output_folder.iterdir():
        experiment_row = {}
        experiment_row['macro_experiment_id'] = macro_id

        args = initialise_experiment(Path(*experiment_split.parts[1:]), env, testing)
        experiment_row['experiment_id'] = args['experiment_id']

        insert_to_db(experiment_row, schema='experiments', table='parallelization')


def remove_experiment_splits(experiment_file, experiment_folder='experiments'):
    '''
    Removes folder containing experiment splits, created for parallelization.

    Parameters
    ----------
    experiment_file: string
        Name of the macro experiment ran in parallel. E.g. 'dummy_experiment.yaml'
    experiment_folder: string
        Name of the folder where the macro experiment is saved.
    '''
    rm_folder = Path(experiment_folder) / parallel_experiments_folder / experiment_file.replace('.yaml', '')

    try:
        shutil.rmtree(rm_folder, ignore_errors=False)
    except FileNotFoundError as e:
        logging.error(f'Folder not found {clean_folder}')


if __name__ == '__main__':
    # python src/pipeline/parallelization.py initialise_parallel_experiments --experiment_file='dummy_experiment.yaml'
    fire.Fire()
