# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
from pipeline.data_persistence import persist_local
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from utils import utils
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
if source_path not in sys.path:
    sys.path.insert(0, source_path)
# sys.path.insert(0, '../utils')

# Packages


# -

def do_plots(experiment_id):

    # Get data on experiment results from database

    con = utils.connect_to_database()

    query = """
    select evaluation.*,approach.name
    from experiments.evaluations evaluation
    left join experiments.approaches approach
    on evaluation.approach_id = approach.approach_id
    """

    df = pd.read_sql_query(query, con)

    # Subselect data on specific experiment id
    data = df.loc[df['experiment_id'] == experiment_id]

    # Set of colors to be used in the plot
    n = len(data['learner_id'])
    color = iter(cm.rainbow(np.linspace(0, 1, n)))

    # Set font size
    plt.rcParams.update({'font.size': 14})

    # Loop to create one fig per metric and a line per learner
    for metric in data['eval_metric'].unique():

        fig, ax1 = plt.subplots(figsize=(15, 8))

        ax1.set_title(f"Metric: {metric}")
        ax1.set_ylabel('score')

        # check if it is k-fold or temporal-fold
        if '-' in data['fold'].iloc[0]:
            ax1.set_xlabel('time')
            plt.xticks(rotation=90)
        else:
            ax1.set_xlabel('fold')

        for approach in data['approach_id'].unique():

            c = next(color)

            for learner in data['learner_id'].unique():

                data_to_plot = data[(data['learner_id'] == learner) &
                                    (data['approach_id'] == approach) &
                                    (data['eval_metric'] == metric)]

                approach_name = data_to_plot['name'].unique()

                ax1.plot(data_to_plot['fold'], data_to_plot['score'], c=c)

                ax1.legend(approach_name)

        persist_local(
            data=fig,
            args={
                'experiment_id': experiment_id,
                'eval_metric': metric},
            folder='evaluation_plots',
            id_keys=[
                'experiment_id',
                'eval_metric'],
            as_type='.png')
