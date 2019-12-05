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

# # Welcome to Model Selection Notebook
#
# >This notebook has the goal of **selecting a model from an experiment** and **ship it to production**.
#
# This process is made in two steps. The first selection ensures a high performance. All models goes through the first one and just hundreds are selected. Then, the second selection comes in to select from those top performing models. The second selection is based in Bias and Fairness metrics. With that, DNCP can ensures more equity or, give more importance to a certain type of tender. But, all of that without losing performance.
#

# # Setup

# ## Imports

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath('joaoc-experiment-checks')).parent.parent.parent / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)
    
import yaml

from model_selection import plotting, select_model, select_bias
from model_selection.to_production import prepare_to_production


from ipywidgets import widgets

# %matplotlib inline
# -

# ## >> Insert Macro Experiment ID
#
# <div class="alert alert-block alert-info">
# This is the same id that is printed on the terminal when you run the pipeline.
#
# But, you can also find it on the database. Just look for it on <b>experiments.parellelization</b> schema.
# </div>

print('Macro Experiment id:')
text = widgets.Text()
display(text)

macro_experiment_id = int(text.value) # Add here your parameters
args = dict(experiment_id=macro_experiment_id)

# ## >> Set Baselines
#
# <div class="alert alert-block alert-info">
# It is improtant to compare your model with a baseline. The standard implementation carries a Random and a Ordered by Value Baseline.
# But, in the future, old models can be used as baseline. 
#     
# As long as the baseline data is in your experiment evaluation dataset, you can add it by filling up the dictonary below.
#
# The <b>database_name</b> is the approach name.
# </div>
#
# <div class="alert alert-block alert-success">
# <b>TIP:</b> you can use any approach name in the database_name. So, you can compare a family of models, like xgb, with the selected models.
# </div>

baselines = [{'database_name': 'random',
              'plot_name': 'Random Baseline',
              'color': 'grey'
             },
              {'database_name': 'higher_than_x',
               'plot_name': 'Ordered By Value',
              'color': '#1f497d'}]

# ## Count Errors

select_model.count_errors(macro_experiment_id)

# # First Data Selection: Performance Metrics
#
# It selects models according to overall performance metrics such as `recall` and `precision`.

# ## How to Use
#
# List of rules to filter the table experiments.evaluations.
#
# One rule orders a list of learners according to an statistic that is 
# applied to all folds of a given experiment. Then, it orders it given
# the higher_is_better paramenter. Finally, the top is used to cut the list.
#
# **Rules**
#
# `metric`: 
#       It accepts any implemented metric, i.e., precision, recall, acurracy.
#       See pipeline/evaluation.py.
#
# `filter`: It filters the statistic of the metrix by threshold or top. 
#       `type` 
#          can be `threshold` or `top`.
#       `value` 
#           is a number to filter by.  
#
# `statistic`: 
#       Any statistic accepted by pd.Groupby.agg function.
#       
#       Some examples are: mean, median, std, sum, min, max, skew, kurt, sem, var
#
# `higher_is_better`: 
#       Teels the algorithm how to order the list given the statistic of the metric.
#       If the metric is good with high values, then, this parameter should be true
#
# **Selector**
#
# Selector is a set of rules to be applied to an experiment result.         
# One can build multiple selectors with different rules. 
#
# To build a selector, you need to declare the order that you want the rules
# above to be applied. For instance, a selector can use the rules in the following order:
#  1 -> 3 -> 2. Or it can be as simple as one rule: 1
#
# This chains are defined by a list, [1, 3, 2] or [1], that has to be added to the order key
# for each selector.

selector = yaml.load("""
rules:
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 5
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 10
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 30
        statistic: mean
        higher_is_better: true
    - 
        metric: 'recall@30@overall'
        filter: 
            type: top
            value: 10
        statistic: std
        higher_is_better: false
        

selectors:
    -
        order: [1] 
    -
        order: [2] 
    -
        order: [3, 4]
""")

metric_selected, data, max_fold = select_model.run(macro_experiment_id, selector)


# ## Recall by Quality Reviews
#
# <div class="alert alert-block alert-info">
# The <b>Selected Models</b> line has an interval. The intervall boundaries are the minimum and maximum of all selected models.
# </div>

plotting.plot_metrics(metric_selected, data, max_fold, baselines, metric='recall')

# ## Precision by Quality Reviews

plotting.plot_metrics(metric_selected, data, max_fold, baselines, metric='precision')

# # Second Data Selection: Bias and Fairness Metrics

# ## Set Configuration

# This selector applies Bias and Fairness rules to variables that you want to control for. For instance, DNCP may want a models that equally selects tenders according to type of procedure. With that in mind, the data scientist can select a model that has a high recall, but ensures better distribution of tenders for type of procedure.
#
# The selector is built on top of [aequitas](https://github.com/dssg/aequitas). This is a open source project that calculates Bias and Fairness metrics for any given dataset. The available variables to filter from are:
#
# 'fdr', 'fdr_disparity', 'fnr', 'fnr_disparity', 'for',
# 'for_disparity', 'fpr', 'fpr_disparity', 'npv', 'npv_disparity',
# 'ppr', 'ppr_disparity', 'pprev', 'pprev_disparity', 'precision',
# 'precision_disparity', 'prev', 'tnr', 'tnr_disparity', 'tpr',
# 'tpr_disparity'
#
# The most important metrics for DNCP are `ppr` and `pprev`, which ensures equity given the population distribution.
#
# You can know more about them in the [official documentation](https://dssg.github.io/aequitas/metrics.html).
#
#
# ### How to Use
#
# #### Query
#
# Here you select the variables that you want to control for. Those variables have to be somewhere in the database, in order for you to be able to query them. Make sure that you query `all tenders` available. 
#
# <div class="alert alert-block alert-info">
# Remember that they have to be categorical
# </div>
#
# #### Groups
#
# This allows you to tweak the controled variable categories. This ensures that you are controlling for bias using labels that make sense. Also, you can change the names to get prettier and more explainable plots. 
#
# <div class="alert alert-block alert-info">
# Notice that you can colapse different labels into one.
# </div>
#
# `name`: name of the variable to tweak. It is the same name that it had in the query.
#
# `new_name`: you can give it a prettier name, if you want.
#
# `remove`: a list of labels that you want to exclude. Be careful with that, because you may add bias (even more).
#
# `replace`: a relation of labels and their new names. `old_name:new_name`
#
# #### Selection
#
# Where the magic begin. The selection part has two main constructs: `rules` and `selectors`. The `rules` are basic filters to select the model that you want based on aequitas metrics. The `selectors` are a set of rules that you may want to use to select the model. It can be as simple as one rule. Or, it allows you to chain rules to get more complex selections.
#
# Given a variable that you are controlling for, the selector calculates a metric for each label of the value. Then, it applies an statistic over it. With that results, different models can be orderes by this metric statitic, and thus selected by filters.
#
# **rules**
#
# `attribute_name`: variable that you want to control for
#
# `metric`: an aequitas metric
#
# `filter`: there are two ways to filter. `top` gets a slice of the list of size set by `value`. `threshold` selects all models that are has the metric statistics higher or lower than a `value`
#
# `statistic`: how you want to compute the metric. It can use any pandas aggregate function, but you can also set a custom function. It comes with `sum_of_abs_difference` which subtracts all results by 1, gets the absolute value and adds it up. 
#
# `higher_is_better`: boolean. if true, higher metric statistics are better than lower. This is used to sort the models.
#
# **selectors**
#
# A list of rules that are suposed to filter a set of models. Make sure that the last rule selects just 1 model. This can be done using the filter as `top` with `value` 1.
#
# To build a selector, you need to declare the order that you want the rules
# above to be applied. For instance, a selector can use the rules in the following order:
#  1 -> 3 -> 2. Or it can be as simple as one rule: 1
#
# This chains are defined by a list, [1, 3, 2] or [1], that has to be added to the order key
# for each selector.
#

bias_selector = yaml.load("""
query: |
    select 
            tenders.id_llamado, 
            tipo_entidad,
            _objeto_licitacion,
            tipo_procedimiento_codigo,
            idh
    from semantic.tenders as tenders
    left join (
    select 
        case when idh::float <= 0.703 then 'Low'
             when idh::float > 0.744 then 'High'
             when (idh::float <= 0.735 and idh::float > 0.703) then 'Medium Low'
             when (idh::float > 0.735 and idh::float <= 0.744) then 'Medium High'
        end idh,
        id_llamado
    from semantic.tenders_demographics) as idh 
    on tenders.id_llamado = idh.id_llamado
    

groups:
    -
        name: 'tipo_entidad'
        new_name: 'Agency Type'
        remove: ['NO CLASIFICADO']
        replace:
            Organismos de la Administración Central: Central Adm.
            Municipalidades: Cities
            Entidades Descentralizadas: Decentralised Entities
    -
        name: 'tipo_procedimiento_codigo'
        new_name: 'Tender Value'
        replace:         
            CD: Low Value
            CO: Medium Value
            LPN: High Value
            LPI: High Value
            LC: High Value
            CE: High Value
    -
        name: '_objeto_licitacion'
        new_name: 'Service Category'
        remove: [LOCINM, LOCMUE, TIERRAS, CON]
        replace:
            SER: Services
            BIEN: Goods
            OBRAS: Infrastructure
            CON: Consultancy
    -
        name: 'idh'
        new_name: 'Human Development Index'
        
selection:
    rules:
        - 
            attribute_name: 'Agency Type'
            metric: 'ppr'
            filter:
                type: top
                value: 1
            statistic: sum_of_abs_difference
            higher_is_better: false 
            
        - 
            attribute_name: 'Agency Type'
            metric: 'pprev'
            filter: 
                type: top
                value: 10
            statistic: sum_of_abs_difference
            higher_is_better: false    

    selectors:
        -
            order: [1] # 1st
        -
            order: [2] # 2nd
""")

# ## >> Choose a metric selector
#
# <div class="alert alert-block alert-info">
# You have to choose which metric selector to apply the Bias and Fairness selection. 
#
# The numbers go from 1 to the number of selectors that you have built
# </div>

print('Metric Selector:')
text = widgets.Text()
display(text)

selector_index = int(text.value)

# %%capture
selected_learners = metric_selected['selectors'][selector_index - 1]['learner_stats'].index.to_frame(index=False)
results, baselines, selected_data = select_bias.run(bias_selector, data, selected_learners, baselines)

# ## Plot Bias and Fairness Results

plotting.plot_bias(results, baselines, selected_data)

# # Ship Model to Production

# ## >> Choose Bias Selector
#

print('Bias Selector [0, n]:')
text = widgets.Text()
display(text)

final_selected_model = int(text.value)

# ##  >> What is the percentage of good quality reviews?

print('Percentage of quality reviews [0, 100]:')
text = widgets.Text()
display(text)

quality_reviews_perc = float(text.value) / 100 # from 0 to 1

# ## Run

prepare_to_production(final_selected_model, results, 
                      selected_learners, quality_reviews_perc,
                      data['fold'].sort_values().unique()[-2]) # Last fold trained

# # Feature Importance
#
# <div class="alert alert-block alert-danger">
# <b>IN DEVELOPMENT</b>
# </div>

import pandas as pd
from utils.utils import connect_to_database
con = connect_to_database()


def get_features(learner_ids):
       
    query = """
    select learner_id, feature, importance
    from experiments.feature_importances
    where learner_id in ({learner_ids})
    """.format(learner_ids=','.join(map(lambda x: str(x), learner_ids)))
    
    return pd.read_sql_query(query, con)


def get_feature_importance_distribution_for_learner(features, learner_id, top=20):
    
    f = features[features['learner_id'] == learner_id].groupby(['learner_id', 'feature']).mean().reset_index()
    f.nlargest(10, 'importance')

    return f.nlargest(top, 'importance').plot(y='importance', x='feature', kind='barh', figsize=(8, top*0.5))


learner_ids = [results[final_selected_model]['learners'].index[0]]
features = get_features(learner_ids)

# ## Feature distribution for given learner

get_feature_importance_distribution_for_learner(features, list(learner_ids)[0], top=10)


# ## Useless Features

useless = features.groupby(['learner_id', 'feature']).mean().query('importance == 0')\
    .reset_index().groupby('feature').count()['importance'].sort_values(ascending=False).to_frame()

print('Number of Useless Features: ', len(useless))

useless


