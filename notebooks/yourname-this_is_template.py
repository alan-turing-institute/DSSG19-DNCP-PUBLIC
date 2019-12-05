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
sys.path.insert(0, '../src/utils')
import utils 

# Packages
import pandas as pd
pd.options.display.max_columns = 999
# -

# ### Start connection with db
#
#

con = utils.connect_to_database()

# ### Querying with pandas

query = """
select *
from raw.proceso
limit 1000
"""

df = pd.read_sql_query(query, con).head(1)

df.head()

# ### Creating path to save data in shared folder

path = utils.path_to_shared('wen', 'data_outpt', 'test', 'csv')

df.to_csv(path)

# ### Creating Plot

# +
nrows = 1 # Number of rows
ncols = 3 # Number of columns
fontsize = 14 # General fontsize

# Create a subplot grid
# ax is a list with the dimensions [nrows][ncols]
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(10, 4)) # (width, heigth)

# You can loop though your data or manually set the subplot ax
for r in range(nrows):
    for c in range(ncols):
        
        # Set the subplot
        axis = ax[r][c]
        
        # Plot something, it can be more than one 
        axis.scatter(df['X'], df['Y'], '--', c='red')
        
        # Add lines  
        axis.axhline(y=, xmin=0, xmax=1) # horizontal [xmin, xmax] range [0, 1]
        axis.axvline(x= ,ymin=0, ymax=1) # vertical   [ymin, ymax] range [0, 1]
        
        # Set axis labels
        axis.set_xlabel( 'X label', 
                        fontsize=fontsize)
        axis.set_ylabel('Y label', 
                        fontsize=fontsize)
        
        # Add text somewhere inside plot
        axis.text(0.10, 0.9, # [x, y] 
                  'Some text',
                  transform=axis.transAxes, fontsize=fontsize)
        
        # Set tick params size
        axis.tick_params(axis='both', labelsize=12)



fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'processo_fechas_diff', 'png')
# fig.savefig(path) # save it somewhere
# -


