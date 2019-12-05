import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
if source_path not in sys.path:
    sys.path.insert(0, source_path)
# sys.path.insert(0, '../utils')
from utils import utils

# Packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
import seaborn as sns
import numpy as np
from pipeline.data_persistence import persist_local, get_local
from itertools import cycle
import pickle
import seaborn as sns
from collections import defaultdict

color = cycle(cm.get_cmap('tab20', 20).colors)

def fetch_data(experiment_id):
    
    con = utils.connect_to_database()

    query = f"""
    select a.*, b.name
    from experiments.evaluations a
    inner join experiments.approaches b
    on a.experiment_id = b.experiment_id and a.approach_id = b.approach_id 
    where a.experiment_id = {experiment_id} 
    """

    return pd.read_sql_query(query, con)[['learner_id', 'fold', 'eval_metric', 'score', 'experiment_id', 'approach_id', 'name']]


def overall_performance_per_fold(args, thresh=0.1):
    
    title = f'Experiment : {args["experiment_id"]}'
    
    def complaints_per_fold(args, data):
    
        folds = pickle.load(open(f'/data/persist/folds/{args["experiment_id"]}.p', 'rb'))
        labels = get_local(args, 'labels').set_index('id_llamado')[['target']]

        i = 0
        complaints = []
        for fold in folds:

            if fold['name'] in list(data['fold'].unique()):
                complaints.append({
                    'complaints': 100 * labels.loc[fold['test']].sum().values[0] / len(labels.loc[fold['test']]),
                    'fold': fold['name']})

        return pd.DataFrame(complaints)
    
    data = fetch_data(args['experiment_id'])

    eval_metrics = data['eval_metric'].unique()

    nrows = len(eval_metrics)  # Number of rows
    ncols = 1 # Number of columns
    fontsize = 14 # General fontsize
                                 
    grid = plt.GridSpec(nrows * 2 + 2, ncols)

    fig = plt.figure(figsize=(15, nrows * 7))
    
    fig.suptitle(title, fontsize=18, y=0.9)
                                 
    # Percentage of Complaints

    axis = plt.subplot(grid[1, 0])

    complaints = complaints_per_fold(args, data)
    axis.bar(complaints['fold'], complaints['complaints'], label='% of true labels per fold', align='edge')

    axis.get_xaxis().set_visible(False)

    axis.set_ylabel('%', 
                        fontsize=fontsize)

    axis.legend()

    for row, eval_metric in enumerate(eval_metrics):

        axis = plt.subplot(grid[2 + row * 2: 2 + (row + 1) * 2, 0])

        df = data.query(f'eval_metric == "{eval_metric}"').query(f'score > {thresh}')

        # Plot something, it can be more than one 
        axis = sns.boxplot(x='fold', y='score', hue='eval_metric', data=df, ax=axis,
                           boxprops=dict(alpha=0.3)
                          )

        if nrows - 1 > row:
            axis.get_xaxis().set_visible(False)


        # Add lines  
        # axis.axhline(y=0.03, xmin=0, xmax=1) # horizontal [xmin, xmax] range [0, 1]
        # axis.axvline(x= ,ymin=0, ymax=1) # vertical   [ymin, ymax] range [0, 1]

        # Set axis labels
        # axis.set_xlabel( 'X label', 
        #                 fontsize=fontsize)
        # axis.set_ylabel('Y label', 
        #                 fontsize=fontsize)

        # Add text somewhere inside plot
        # axis.text(0.10, 0.9, # [x, y] 
        #           'Some text',
        #           transform=axis.transAxes, fontsize=fontsize)

        # Set tick params size
        axis.tick_params(axis='both', labelsize=12)
        axis.tick_params(axis='x', rotation=45)


    fig.tight_layout()
                                                                                
    persist_local(
        data=fig, 
        args={'experiment_id': args['experiment_id'], 
              'title': title}, 
        folder='evaluation_plots', 
        id_keys=['experiment_id', 'title'], 
        as_type='.png')

def plot_metric_by_fold(selector, experiment_id):
    
    
    data = selector['data']
    selector_name = selector['name']
    
    metrics = data['eval_metric'].unique()
    
    nrows = 1  # Number of rows
    ncols = 1 # Number of columns
    fontsize = 14 # General fontsize
    
    title = """
Experiment: {experiment_id} 
Metric: {metric}
Selector: {selector_name}"""

    # Loop to create one fig per metric and a line per learner
    for metric in metrics:
        
        fig, axis = plt.subplots(figsize = (20,5))
        
        axis.set_title(title.format(experiment_id=experiment_id,
                                    selector_name=selector_name,
                                    metric=metric
                                   ))
        axis.set_ylabel('score')
        axis.legend()

        #check if it is k-fold or temporal-fold
        if '-' in  data['fold'].iloc[0]:
            axis.set_xlabel('time')
            axis.tick_params(axis='both', labelsize=12)
            axis.tick_params(axis='x', rotation=45)
        else:
            axis.set_xlabel('fold')
            
        for i, learner in enumerate(data['learner_id'].unique()):

            data_to_plot = data[(data['learner_id'] == learner) & 
                                (data['eval_metric'] == metric)]

        axis.plot(data_to_plot['fold'], data_to_plot['score'], 
                      color=next(color), label=data['name'].unique()[0] + '_' + str(learner))


        axis.legend()
        
        persist_local(
        data=fig, 
        args={'experiment_id': experiment_id, 
              'title': title, 'eval_metric': metric}, 
        folder='evaluation_plots', 
        id_keys=['experiment_id', 'title', 'eval_metric'], 
        as_type='.png')


# +
def plot_approaches_by_fold(experiment_id, best_learner = ''):
    """
    Given a experiment_id and learner_id (optional), returns a plot
    for each evaluation metric.
    The plot shows the score for the metric for each learner at each fold. 
    Learners that use the same approach have the same color.
    
    Parameters
    ----------
    experiment_id : int
        experiment_id to evaluate. 
    best_learner: int
        learner_id selected as the best learner. 

    Returns
    -------
    Plots
    """
    
    data = fetch_data(experiment_id)
    metrics = data['eval_metric'].unique()
    approaches = data['name'].unique()
    
    nrows = 1  # Number of rows
    ncols = 1 # Number of columns
    fontsize = 14 # General fontsize
    
    colors = cycle(cm.get_cmap('tab10', len(approaches)).colors)
    
    title = """
    Experiment: {experiment_id} 
    Metric: {metric}"""

    # Loop to create one fig per metric and a line per learner
    for metric in metrics:
        
        fig, axis = plt.subplots(figsize = (10,4))
        
        axis.set_title(title.format(experiment_id=experiment_id,
                                    metric=metric
                                   ))
        axis.set_ylabel('score')
        axis.set_ylim([0,1])
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.legend()

        #check if it is k-fold or temporal-fold
        if '-' in  data['fold'].iloc[0]:
            axis.set_xlabel('time')
            axis.tick_params(axis='both', labelsize=12)
            axis.tick_params(axis='x', rotation=45)
        else:
            axis.set_xlabel('fold')
            
        for approach in approaches:
            
            c = next(colors)
            label = approach
            
            for i, learner in enumerate(data['learner_id'].unique()):
                
                if learner == best_learner:
                    linestyle = 'dashed'
                    linewidth = 3
                    alpha = 1
                else:
                    linestyle = 'solid'
                    linewidth = 1
                    alpha = 0.8

                data_to_plot = data[(data['learner_id'] == learner) & 
                                    (data['eval_metric'] == metric) &
                                    (data['name'] == approach)]

                line = axis.plot(data_to_plot['fold'], data_to_plot['score'],
                            color = c , 
                            alpha = alpha, 
                            linestyle = linestyle,
                            linewidth = linewidth,
                            label=approach if i == 0 else "")
        
                
        plt.legend(loc = 'center left', bbox_to_anchor = (1,0.5), frameon = False, title = 'metric' )
        
        persist_local(
        data=fig, 
        args={'experiment_id': experiment_id, 
            'title': title, 'eval_metric': metric}, 
        folder='evaluation_plots', 
        id_keys=['experiment_id', 'title', 'eval_metric'], 
        as_type='.png')


# -

def plot_approaches_by_fold_avg_std(experiment_id):
    data = fetch_data(experiment_id)
    sns.set(rc={"figure.figsize":(10,11)})
    sns.set_style('white')
    metrics = data['eval_metric'].unique()
    ncols = 1
    nrows = len(metrics)
    fig,axes = plt.subplots(ncols = ncols, nrows = nrows) 

    for metric,ax in zip(metrics,axes.flat):
        sns.lineplot(ax = ax,
                     x='fold', y = 'score', 
                     hue = 'name',
                     alpha = 0.8,
                     data = data.loc[data['eval_metric'] == metric])
        ax.set_title(f'Experiment_id: {experiment_id} \n Metric: {metric}')
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc = 'center left', bbox_to_anchor = (1,0.5), frameon = False, title = 'Approaches')
    fig.tight_layout()



def get_index_of_line(line, df, axis, forbiden='forest'):
    line = [i for i in df.unique() if ((line in i) & (forbiden not in i))][0]
    return [i.get_text() for i in axis.lines[0].axes.get_legend().texts].index(line) - 1

def plot_metrics(res, data, max_fold, baselines, metric='recall', fontsize=20):

    for selector in res['selectors']:

        fig, axis = plt.subplots(figsize=(15,  10),
                                 ncols=1, nrows=1)

        df = data[data['learner_id'].isin(selector['data']['learner_id'].unique())]\
            .query(f'fold == "{max_fold}"')
        df = pd.concat([df, data[data['name'].isin([base['database_name'] for base in baselines])]\
                .query(f'fold == "{max_fold}"')])

        translate = defaultdict(lambda: 'Selected Models')
        translate.update({base['database_name']: base['plot_name']  for base in baselines})

        df['Models'] = df['name'].apply(lambda x: translate[x])

        axis = sns.lineplot(x="at", 
                            y="score",
                            hue="Models",
                            hue_order=[base['plot_name'] for base in baselines] + ['Selected Models'],
                            palette=[base['color'] for base in baselines] + ['#f3b32a'],
                            ci=100,
                            data=df.query('type_procurement == "overall"')\
                                    .query(f'metric == "{metric}"'),
                            ax=axis)

        leg = axis.legend(prop={'size': fontsize})

        for base in baselines:

            idx = get_index_of_line(base['plot_name'], df['Models'], axis)
            axis.lines[idx].set_linestyle(":")
            axis.lines[idx].set_linewidth(5)
            leg.get_lines()[idx+1].set_linestyle(":")
            leg.get_lines()[idx+1].set_linewidth(5)


        axis.set_ylabel('Recall', fontsize=fontsize)
        axis.set_xlabel('% of quality reviews', fontsize=fontsize)
        axis.tick_params(axis='both', labelsize=fontsize)
        axis.set_title(f"""
        Best models for {selector['name']}""".format(selector['name']), fontdict={'fontsize': fontsize})

        fig.tight_layout()


# Plot Bias

def prepare_to_plot(df, name):

    df['perc_tenders'] = df['pp'].divide(df['k']).multiply(100).round(0)
    df = df[['attribute_name', 'attribute_value', 'perc_tenders']]
    df['approach'] = name
    df = df.rename(columns={'attribute_name':'attribute_group', 'attribute_value':'attribute'})
    
    return df

def get_plot_data(results, baselines, selected_data):

    for result in results:
    
        pre_concat = [prepare_to_plot([s for s in selected_data 
                                        if s['learner_id'] == result['learners'].index[0]][0]['aequitas'],
                                    'Best Model')]

        for base in baselines:
            pre_concat.append(prepare_to_plot([s for s in selected_data 
                                                if s['learner_id'] == base['id']['learner_id'].iloc[0]][0]['aequitas'],
                                                base['plot_name']))


        result['plot_data'] = pd.concat(pre_concat)

    return results

def force_order(df, keys):
    
    return df.set_index('attribute').ix[keys].reset_index()

def plot_bars(result, baselines):

    df = result['plot_data']
    
    attributes = df['attribute_group'].unique()

    # Set quantity and position of axis
    nrows = round(len(attributes) / 2)
    ncols = 2
    fontsize = 20 # General fontsize

    # Create a subplot grid
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(20, 12)) # (width, heigth)

    fig.suptitle(result['name'], fontsize=fontsize)
    
    # Iterate through plots
    i=0
    for ax, name in zip(axis.flatten(), df['attribute_group'].unique()):
        
        # Define subset of data to plot
        df_to_plot = df.loc[(df['attribute_group'] == name)]
           
        
        if name == 'Tender Value':
            df_to_plot = force_order(df_to_plot, ['Low Value', 'Medium Value', 'High Value'])
            
        if name == 'Human Development Index':
            keys = ['Low', 'Medium Low', 'Medium High', 'High']
            df_to_plot = force_order(df_to_plot, keys)
        
        
        ax = sns.barplot(x='attribute', y='perc_tenders', hue='approach', data=df_to_plot, ax=ax,
                        hue_order=[base['plot_name'] for base in baselines] + ['Best Model'],
                        palette=[base['color'] for base in baselines] + ['#f3b32a'])
        
        # Set title
        ax.set_title(name, fontsize=fontsize)

        ax.set_ylabel('% of procurements to review', 
                        fontsize=fontsize)
        
        ax.set_xlabel('')
        # Set axis limits

        ax.set_ylim(0, 100)

        # Set tick params size
        ax.tick_params(axis='both', labelsize=fontsize)

        # Set legend
        ax.legend(fontsize=fontsize)
 
        i = i + 1

def plot_bias(results, baselines, selected_data):

    results = get_plot_data(results, baselines, selected_data)

    for result in results:

        plot_bars(result, baselines)