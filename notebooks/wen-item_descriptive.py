# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
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
import matplotlib.pyplot as plt
import numpy as np
pd.options.display.max_columns = 999
# -

# ### Start connection with db
#
#

con = utils.connect_to_database()

# ### Understanding the items data

# Find the number of items in the database
query = """
select count(*) 
from raw.item_solicitado
"""

df = pd.read_sql_query(query, con).head(1)

df.head()

# Number of items procured over the year
query = """
select count(*), cast(extract(year from fecha_publicacion) as int) as year
from raw.item_solicitado
group by extract(year from fecha_publicacion)
order by year;
"""

df = pd.read_sql_query(query, con)

plt.bar('year', 'count', data = df)

# Number of categories in producto_nombre_convocante
query = """
with convocante as 
	(select distinct producto_nombre_convocante
	from raw.item_solicitado)
select count(*)	from convocante;
"""
pd.read_sql_query(query, con)

# Number of categories in producto_nombre_catalogo
query = """
with convocante as 
	(select distinct producto_nombre_catalogo
	from raw.item_solicitado)
select count(*)	from convocante;
"""
pd.read_sql_query(query, con)

# What are the most frequent items procured by producto_nombre_catalogo
query = """
select distinct producto_nombre_catalogo, count(producto_nombre_catalogo) as freq
from raw.item_solicitado
group by producto_nombre_catalogo
order by freq desc;
"""
df = pd.read_sql_query(query, con)
df[:10]

# Cumulative graph of product categories
df['cumsum'] = np.cumsum(df['freq'])
df['perc'] = df['cumsum']/np.sum(df.freq)
df['index'] = np.arange(len(df))
fig, ax = plt.subplots()
ax.fill_between(df.index, 0, df.perc,)
plt.axvline(x=df['index'].loc[df['perc']>=0.8].iloc[0], color='red')
plt.xlabel('Items ordered by frequency in procurement')
plt.ylabel('Cumulative percentage % of total procurements')

# Number of types of presentation of items
query = """
with presentation as 
(select distinct presentacion
from raw.item_solicitado)
select count(*) from presentation;
"""
pd.read_sql_query(query, con)

# Most frequent types of presentation
query = """
select distinct presentacion, count(presentacion) as freq
from raw.item_solicitado
group by presentacion
order by freq desc
limit 10;
"""
pd.read_sql_query(query,con)

# Number of units of measure
query = """
with units as 
(select distinct unidad_medida
from raw.item_solicitado)
select count(*) from units;
"""
pd.read_sql_query(query, con)

# Distribution of value of items 
query = """
select monto * precio_unitario_estimado as total_value
from raw.item_solicitado;
"""
df = pd.read_sql_query(query, con)
df['total_value'].describe()

# ### Creating path to save data in shared folder

path = utils.path_to_shared('wen', 'data_outpt', 'test', 'csv')

df.to_csv(path)
