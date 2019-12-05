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

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
from pathlib import Path
import os
source_path = str(Path(os.path.abspath('joaoc-experiment-checks')).parent.parent / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)


# Packages
import pandas as pd
pd.options.display.max_columns = 999

import warnings
warnings.filterwarnings('ignore')
# -

from model_selection import plotting, select_model
from utils.utils import connect_to_database

con = connect_to_database()

# Add here your parameters:
experiment_id = 1129688681
model_selector = 'good_selector'

args = dict(experiment_id=experiment_id)

plotting.overall_performance_per_fold(args, thresh=0)

res = select_model.run(args['experiment_id'], Path().cwd().parent / 'model_selectors' / (model_selector + '.yaml'))


for df in res['selectors']:
    print(df['learner_stats'])

good_learners = res['selectors'][3]['learner_ids']

# ## Feature Importance

query = """
select feature, avg(abs(importance)) importance
from experiments.feature_importances
where learner_id = {learner_id}
and importance != 0
group by feature
order by importance desc
"""

good_learners.index[1]

df = pd.read_sql_query(query.format(learner_id=good_learners.index[0]), con)

# %matplotlib inline

import numpy as np

df['importance_log'] = df['importance'].apply(lambda x: np.log10(x) + 1)

df

df.head(20).plot(x='feature', y='importance_log', kind='barh', )

df.head(10)


# ### Features

from pipeline.data_persistence import get_local
features = get_local(args, 'features').set_index('id_llamado')

features

test('a')


