'''
Script aiming at loading configuration files and returning dictionaries.
    Input:
        Config file name.
    Output:
        Dictionary with configuration data.
'''

from utils.utils import open_yaml
import yaml
import logging
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
if source_path not in sys.path:
    sys.path.insert(0, source_path)


def load_environment(
        config_file=Path(os.path.abspath(__file__)).parent.parent.parent / 'env.yaml'):
    """
    Load configuration file.

    Parameters
    ----------
    config_file: string
        Environment file name

    Return
    ------
    Dictionary
        Dictionary with environment configuration
    """

    return open_yaml(config_file)


def load_experiment(experiment_file, experiment_folder='experiments'):
    """
    Given an experiment YAML file, loads and returns a dictionary.

    Parameters
    ----------
    experiment_file: string
        Experiment file name
    experiment_folder: string
        Experiments folder. By default, 'experiments' folder in the repo.

    Return
    ------
    Dictionary
        Dictionary with experiment configuration
    """

    return open_yaml(Path(experiment_folder) / experiment_file)


if __name__ == '__main__':
    load_environment()
    load_experiment('dummy_experiment.yaml')
