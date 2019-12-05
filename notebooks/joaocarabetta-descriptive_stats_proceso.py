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
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
# Para legenda com linhas:
from matplotlib import lines
# Para legenda com barras:
from matplotlib.patches import Patch
# -

con = utils.connect_to_database()

# # Questions

# Sanity check
# * Check for duplicates in tender ID
# * Check null values
# * EVery tender has documents? 
#
# Counts
# * Who are the biggest public bodies by tender counts
#
# Time differences
# * Number of tenders over time for each type of time type
# * Check differences between planning amount (42) tender amount (43) and final award amount (44)
# * Total time it takes from planning to tender to award 
#
# Distributions
# * Distribution of categories of agencies (13), categories of items (14), type of procurement (17)
# * Distribution of time of contract execution (50)
# * Distribution of type of awarding system (39)
#
# Values General
# * Guaranies and Dolar identification
# * Inflation
#
# Values
# * Who are the biggest public bodies by tender values 
# * Top tenders over time
# * How many processes are currently "open"
#
#
#
#



# # Sanity Check

# ## Duplicates

query = """
select id_llamado, count(*) as counta
from raw.proceso
group by id_llamado
order by counta desc
"""

pd.read_sql_query(query, con)

# #### There are no duplicates

# ## Number of Nulls

df = utils.check_null_values('raw', 'proceso', con)

df

path = path_to_shared('joaocarabetta', 'data_output', 'proceso_nulls_count', 'csv')

df = df.query('nulls > 0').sort_values(by='nulls', ascending=False)
df.to_csv(path, index=False)

df

# ## Every tender has a document?

query = """
select count(*)
from raw.proceso
"""

print('Number of tender: ', pd.read_sql_query(query, con).values[0][0])

query = """
select count(distinct t1.id_llamado)
from raw.proceso t1
join raw.pbc_adenda t2
on t1.id_llamado = t2.id_llamado
"""

print('Number of tenders with documents: ', pd.read_sql_query(query, con).values[0][0])

print('Number of tenders without documents: ', 126238 - 77761)

# ### Complaints over time

query = """
select *
from raw.proceso_juridico
"""

complaints = pd.read_sql_query(query, con)

complaints.head(1)

complaints.info()

complaints.query('_tipo_resultado == "a_favor"')\
        .set_index('fecha_resolucion_proceso')\
        .resample('M').count()['id_llamado']

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 7))


# First plot
axis = ax

data = complaints.query('_tipo_resultado == "a_favor"')\
        .set_index('fecha_resolucion_proceso')\
        .resample('M').count()

for year in data.index.year.unique():
    d = data[data.index.year == year]

    axis.plot(d.index.month, d['id_llamado'], label=year)
# axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of effective complaints', fontsize=fontsize)
axis.set_xlabel('Fecha resolucion proceso', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
# path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_planificacion', 'png')
# fig.savefig(path)

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 7))


# First plot
axis = ax

data = complaints.query('_tipo_resultado == "a_favor"')\
        .set_index('fecha_resolucion_proceso')\
        .resample('M').count()

axis.plot(data.index, data['id_llamado'], label='Total', alpha=0.2)
axis.plot(data.index, data['id_llamado'].rolling(6).mean(), label='6 Months')
axis.plot(data.index, data['id_llamado'].rolling(12).mean(), label='12 Months')
# axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of effective complaints per Month', fontsize=fontsize)
axis.set_xlabel('Fecha resolucion proceso', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
# path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_planificacion', 'png')
# fig.savefig(path)
# + {}
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 7))


# First plot
axis = ax

data = complaints.query('_tipo_resultado == "a_favor"')\
        .set_index('fecha_resolucion_proceso')\
        .resample('d').count()

axis.bar(data.index, data['id_llamado'], label='Total')
# axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of effective complaints per Day', fontsize=fontsize)
axis.set_xlabel('Fecha resolucion proceso', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
# path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_planificacion', 'png')
# fig.savefig(path)
# -


query = """
select bool_of_effective_complaints, reception_date
from semantic.labels labels
"""

tenders = pd.read_sql_query(query, con)

tenders.head(1)

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 7))


# First plot
axis = ax

data = tenders.query('bool_of_effective_complaints == 1')\
        .set_index('reception_date')\
        .resample('M').count()

axis.plot(data.index, data['bool_of_effective_complaints'], label='Total', alpha=0.2)
axis.plot(data.index, data['bool_of_effective_complaints'].rolling(6).mean(), label='6 Months')
axis.plot(data.index, data['bool_of_effective_complaints'].rolling(12).mean(), label='12 Months')
# axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of effective complaints per Month', fontsize=fontsize)
axis.set_xlabel('reception_date', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'Number_of_effective_Compaints_per_month_overtime', 'png')
fig.savefig(path)

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 7))


# First plot
axis = ax

data = tenders.query('bool_of_effective_complaints == 1')\
        .set_index('reception_date')\
        .resample('d').count()

axis.bar(data.index, data['bool_of_effective_complaints'], label='Total')
# axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of effective complaints per Day', fontsize=fontsize)
axis.set_xlabel('reception_date', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
# path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_planificacion', 'png')
# fig.savefig(path)

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 7))


# First plot
axis = ax

data = tenders.query('bool_of_effective_complaints == 1')\
        .set_index('reception_date')\
        .resample('M').count()

for year in data.index.year.unique():
    d = data[data.index.year == year]

    axis.plot(d.index.month, d['bool_of_effective_complaints'], label=year)
# axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of effective complaints', fontsize=fontsize)
axis.set_xlabel('reception date', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
# path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_planificacion', 'png')
# fig.savefig(path)
# -

data['month'] = data.index.month

# +
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 7))

ax = sns.boxplot(x="month", y="bool_of_effective_complaints",
                  data=data, palette="Set3", ax=ax)

ax.set_ylabel("bool_of_effective_complaints", fontsize=fontsize)
ax.set_xlabel("month", fontsize=fontsize)

ax.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'tenders_with_effective_complaints_per_reception_month', 'png')
fig.savefig(path)
# -



# ## Is there any temporal explanation?

# ### Fecha planificacion

query = """
select id_llamado, bool_of_effective_complaints
from raw_labeled.proceso
"""

proceso = pd.read_sql_query(query, con)

# +
query = """
select 
    extract(year from t1.fecha_publicacion_planificacion) as year,
    extract(month from t1.fecha_publicacion_planificacion) as month,
    count(distinct t1.id_llamado) as number_of_tenders_with_documents
from raw.proceso t1
join raw.pbc_adenda t2
on t1.id_llamado = t2.id_llamado
group by extract(year from t1.fecha_publicacion_planificacion), 
extract(month from t1.fecha_publicacion_planificacion)
"""

tender_with_docs = pd.read_sql_query(query, con)

query = """
select 
    extract(year from t1.fecha_publicacion_planificacion) as year,
    extract(month from t1.fecha_publicacion_planificacion) as month,
    count(distinct t1.id_llamado) as number_of_tenders
from raw.proceso t1
group by extract(year from t1.fecha_publicacion_planificacion), 
extract(month from t1.fecha_publicacion_planificacion)
"""

tenders = pd.read_sql_query(query, con)
# -

proceso.head(1)

tenders = tenders.merge(tender_with_docs, on=['year', 'month'])
tenders['difference'] = tenders['number_of_tenders'] - tenders['number_of_tenders_with_documents']
tenders['%_difference'] = tenders['difference'] / tenders['number_of_tenders'] * 100
tenders['timestamp'] = tenders.apply(lambda x: datetime.strptime('{}-{}'.format(int(x['year']), int(x['month'])),
                                                            '%Y-%m'), 1)
tenders = tenders.set_index('timestamp')

# +
nrows = 2
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 10))


# First plot
axis = ax[0]

axis.plot(tenders.index, tenders['number_of_tenders'], label='Total')
axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# Second plot
axis = ax[1]

axis.plot(tenders.index, tenders['%_difference'])

axis.set_ylabel('% of tenders with documents', fontsize=fontsize)
axis.set_xlabel('Fecha Planificacion', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_planificacion', 'png')
fig.savefig(path)
# -

# The 'fecha planificacion' is the date that a tender is first planned. All governm. agencies have to submit a plan of expenditures until the end of Feburary of the year. However, if an agency has to buy a new product, the planning stage has to happen again. 
#
# The upper plot shows peaks of tenders being submited to the planning phase in the begginnig of the year. One particular year, 2013, has a huge peak. This may be explained by a change in internal procedures to organise tenders which may be a effect of presidential elections.
#
# The lower plot shows that only about 40% of the tenders have documents related to them. There is no clear explanation why this happens.

# ### Fecha apertura oferta

# +
query = """
select 
    extract(year from t1.fecha_apertura_oferta) as year,
    extract(month from t1.fecha_apertura_oferta) as month,
    count(distinct t1.id_llamado) as number_of_tenders_with_documents
from raw.proceso t1
join raw.pbc_adenda t2
on t1.id_llamado = t2.id_llamado
group by extract(year from t1.fecha_apertura_oferta), 
extract(month from t1.fecha_apertura_oferta)
"""

tender_with_docs = pd.read_sql_query(query, con)

query = """
select 
    extract(year from t1.fecha_apertura_oferta) as year,
    extract(month from t1.fecha_apertura_oferta) as month,
    count(distinct t1.id_llamado) as number_of_tenders
from raw.proceso t1
group by extract(year from t1.fecha_apertura_oferta), 
extract(month from t1.fecha_apertura_oferta)
"""

tenders = pd.read_sql_query(query, con)
# -

tenders = tenders.merge(tender_with_docs, on=['year', 'month'])
tenders['difference'] = tenders['number_of_tenders'] - tenders['number_of_tenders_with_documents']
tenders['%_difference'] = tenders['difference'] / tenders['number_of_tenders'] * 100
tenders = tenders.dropna(subset=['year', 'month'])
tenders['timestamp'] = tenders.apply(lambda x: datetime.strptime('{}-{}'.format(int(x['year']), int(x['month'])),
                                                            '%Y-%m'), 1)
tenders = tenders.set_index('timestamp')

# +
nrows = 2
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 10))


        
axis = ax[0]

axis.plot(tenders.index, tenders['number_of_tenders'], label='Total')
axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

axis = ax[1]

axis.plot(tenders.index, tenders['%_difference'])

axis.set_ylabel('% of tenders with documents', fontsize=fontsize)
axis.set_xlabel('Fecha Apertura Oferta', fontsize=fontsize)
        
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_apertura_oferta', 'png')
fig.savefig(path)
# -

# The 'fecha apertura oferta' is the date when the tender goes to bidding. Until 2012, there were two clear peaks of tender volume. One in the beggining of ther year and other in the last trimester. After 2013, the pattern changes. There are no more peak in the beggining of the year, just in October, November and December.
#
# The lower plot shows a decreasing proportion of tenders with documents attached.

# ### Fecha Publicacion Adjudicacion

# +
query = """
select 
    extract(year from t1.fecha_publicacion_adjudicacion) as year,
    extract(month from t1.fecha_publicacion_adjudicacion) as month,
    count(distinct t1.id_llamado) as number_of_tenders_with_documents
from raw.proceso t1
join raw.pbc_adenda t2
on t1.id_llamado = t2.id_llamado
group by extract(year from t1.fecha_publicacion_adjudicacion), 
extract(month from t1.fecha_publicacion_adjudicacion)
"""

tender_with_docs = pd.read_sql_query(query, con)

query = """
select 
    extract(year from t1.fecha_publicacion_adjudicacion) as year,
    extract(month from t1.fecha_publicacion_adjudicacion) as month,
    count(distinct t1.id_llamado) as number_of_tenders
from raw.proceso t1
group by extract(year from t1.fecha_publicacion_adjudicacion), 
extract(month from t1.fecha_publicacion_adjudicacion)
"""

tenders = pd.read_sql_query(query, con)
# -

tenders = tenders.merge(tender_with_docs, on=['year', 'month'])
tenders['difference'] = tenders['number_of_tenders'] - tenders['number_of_tenders_with_documents']
tenders['%_difference'] = tenders['difference'] / tenders['number_of_tenders'] * 100
tenders = tenders.dropna(subset=['year', 'month'])
tenders['timestamp'] = tenders.apply(lambda x: datetime.strptime('{}-{}'.format(int(x['year']), int(x['month'])),
                                                            '%Y-%m'), 1)
tenders = tenders.set_index('timestamp')

# +
nrows = 2
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(15, 10))


# First plot
axis = ax[0]

axis.plot(tenders.index, tenders['number_of_tenders'], label='Total')
axis.plot(tenders.index, tenders['number_of_tenders_with_documents'], label='With Documents')

axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# Second plot
axis = ax[1]

axis.plot(tenders.index, tenders['%_difference'])

axis.set_ylabel('% of tenders with documents', fontsize=fontsize)
axis.set_xlabel('Fecha Planificacion', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'tenders_and_documents_fecha_adjucacion', 'png')
fig.savefig(path)
# -

# ## Quantity of tenders in each phase per year

query = """
select
    id_llamado,
    fecha_publicacion_planificacion,
    fecha_publicacion_convocatoria,
    fecha_junta_aclaracion ,
    fecha_apertura_oferta,
    fecha_entrega_oferta,
    fecha_publicacion_adjudicacion
from raw.proceso 
"""

tenders = pd.read_sql_query(query, con)

tenders = tenders.set_index('id_llamado')

# +
fechas = tenders.columns

df = pd.DataFrame()

for fecha in fechas:
    df = pd.concat([df, tenders.groupby(tenders[fecha].dt.year).count()[[fecha]]], 1)
# -

df.head()

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(12, 7))


# First plot
axis = ax

axis.plot(df.index, df['fecha_publicacion_planificacion'], label='Planning')
axis.plot(df.index, df['fecha_publicacion_convocatoria'], label='Tender')
axis.plot(df.index, df['fecha_apertura_oferta'], label='Bidding')
axis.plot(df.index, df['fecha_publicacion_adjudicacion'], label='Award')

axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'number_of_tenders_per_process_phase', 'png')
fig.savefig(path)
# -

# ## Amount per year per phase

query = """
select
	id_llamado,
	fecha_publicacion_convocatoria,
	monto_global_estimado_planificacion,
	monto_estimado_convocatoria,
	monto_total_adjudicado,
    complaint
from raw_labeled.proceso 
"""

df = pd.read_sql_query(query, con)

df.head()

# ### Amount per year

import numpy as np

# +

montos = [c for c in df.columns if 'monto' in c] 


df_year = df.groupby(df['fecha_publicacion_convocatoria'].dt.year).sum()[montos]

df_year = df_year.applymap(np.log10)
# -

df_year.head()

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(12, 7))


# First plot
axis = ax

for m in montos:
    axis.plot(df_year.index, df_year[m], label=m)


axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'amount_of_tenders_per_process_phase', 'png')
fig.savefig(path)
# -

# ### Distributions of value 

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(12, 7))


# First plot
axis = ax


data = df.query('complaint == 0')[montos[2]].apply(lambda x: np.log10(1 + x))
axis.hist(data, 
          density=True,
          alpha=0.5, label=f'No complaint. Mean {data.mean().round(2)}'
         )
axis.axvline(data.mean())

data = df.query('complaint == 1')[montos[2]].apply(lambda x: np.log10(1 + x))
axis.hist(data, 
          density=True, alpha=0.5, label=f'With complaint. Mean {data.mean().round(2)}')
axis.axvline(data.mean(), color='orange')


axis.set_ylabel('Normalized Number of Tenders', fontsize=fontsize)
axis.set_xlabel('Value of Award tender in log (Guaranies)', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'histogram_amount_complaint_comp', 'png')
fig.savefig(path)
# -

# ### Probability of complaint by value

df.head()

df['value'] = round(df['monto_total_adjudicado'] / 100000000, 0)
df['prob'] = df.groupby('value').agg(['count', 'sum'])['complaint'].apply(lambda x: int(x['sum']) / int(x['count']), 1)

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(12, 7))


# First plot
axis = ax


data = df[['value', 'prob']]
axis.scatter(df['value'].apply(np.log10), 
          df['prob'],
          label=f'No complaint. Mean {data.mean().round(2)}'
         )



axis.set_ylabel('Normalized Number of Tenders', fontsize=fontsize)
axis.set_xlabel('Value of Award tender in log (Guaranies)', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
# path = utils.path_to_shared('joaoc', 'imgs', 'histogram_amount_complaint_comp', 'png')
# fig.savefig(path)
# -

# # Time Differences

# ## Total time it takes from planning to tender to award 

query = """
select
    id_llamado,
    fecha_publicacion_planificacion,
    fecha_publicacion_convocatoria,
    fecha_junta_aclaracion ,
    fecha_apertura_oferta,
    fecha_entrega_oferta,
    fecha_publicacion_adjudicacion
from raw.proceso 
"""

timestamps = pd.read_sql_query(query, con)

timestamps.info()

# +
nrows = 1
ncols = 3
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(20, 7))

# First plot
axis = ax[0]

var = (timestamps['fecha_publicacion_convocatoria'] - timestamps['fecha_publicacion_planificacion']).apply(lambda x: x.days)
axis.hist(var, bins='fd')

axis.set_xlabel('Convocatoria - Planificacion (days)', fontsize=fontsize)
axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.set_yscale('log')

axis.text(0.10, 0.9, 'Median: {}\nMean: {} '.format(round(var.median(), 0), round(var.mean(), 2)),
                  transform=axis.transAxes, fontsize=fontsize)

# Second plot
axis = ax[1]

var = (timestamps['fecha_apertura_oferta'] - timestamps['fecha_publicacion_planificacion']).apply(lambda x: x.days)
axis.hist(var, bins='fd')

axis.set_xlabel('Apertura Oferta - Planificacion (days)', fontsize=fontsize)
axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.set_yscale('log')

axis.text(0.10, 0.9, 'Median: {}\nMean: {} '.format(round(var.median(), 0), round(var.mean(), 2)),
                  transform=axis.transAxes, fontsize=fontsize)

# Third plot
axis = ax[2]

var = (timestamps['fecha_publicacion_adjudicacion'] - timestamps['fecha_publicacion_planificacion']).apply(lambda x: x.days)
axis.hist(var, bins='fd')

axis.set_xlabel('Adjucacion - Planificacion (days)', fontsize=fontsize)
axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.set_yscale('log')

axis.text(0.10, 0.9, 'Median: {}\nMean: {} '.format(round(var.median(), 0), round(var.mean(), 2)),
                  transform=axis.transAxes, fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'processo_planificacion_fecha_diff', 'png')
fig.savefig(path)
# -

# ## Where does this negative values come from?



negative = timestamps[(
    timestamps['fecha_publicacion_convocatoria'] - 
    timestamps['fecha_publicacion_planificacion']).apply(lambda x: x.days) < 0]

negative.head()

# ### Year related?

from copy import deepcopy

# +
df = deepcopy(timestamps)

df['diff'] = df['fecha_publicacion_convocatoria'] - df['fecha_publicacion_planificacion']

df = df.set_index('fecha_publicacion_planificacion')

df = df['diff'].apply(lambda x: x.days)

# +
nrows = 1
ncols = 1
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(10, 5))

# First plot
axis = ax

gr = df.groupby(pd.Grouper(freq="M")).describe()
axis.plot(gr.index, gr['min'], label='Minimum')
axis.plot(gr.index, gr['50%'], label='Median')
axis.plot(gr.index, gr['mean'], label='Mean')
axis.plot(gr.index, gr['max'], label='Maximum')

axis.set_xlabel('Timestamp', fontsize=fontsize)
axis.set_ylabel('Convocatoria - Planificacion (days)', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'proceso_convocatoria_planificacion_diff_hist', 'png')
fig.savefig(path)

# +
nrows = 1
ncols = 3
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(20, 7))

# First plot
axis = ax[0]

var = (timestamps['fecha_apertura_oferta'] - timestamps['fecha_publicacion_convocatoria']).apply(lambda x: x.days)
axis.hist(var, bins='fd')

axis.set_xlabel('Apertura Oferta - Convocatoria (days)', fontsize=fontsize)
axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.set_yscale('log')

axis.text(0.10, 0.9, 'Median: {}\nMean: {} '.format(round(var.median(), 0), round(var.mean(), 2)),
                  transform=axis.transAxes, fontsize=fontsize)

# Second plot
axis = ax[1]

var = (timestamps['fecha_publicacion_adjudicacion'] - timestamps['fecha_publicacion_convocatoria']).apply(lambda x: x.days)
axis.hist(var, bins='fd')

axis.set_xlabel('Adjucacion - Convocatoria (days)', fontsize=fontsize)
axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.set_yscale('log')

axis.text(0.10, 0.9, 'Median: {}\nMean: {} '.format(round(var.median(), 0), round(var.mean(), 2)),
                  transform=axis.transAxes, fontsize=fontsize)

# Third plot
axis = ax[2]

var = (timestamps['fecha_publicacion_adjudicacion'] - timestamps['fecha_apertura_oferta']).apply(lambda x: x.days)
axis.hist(var, bins='fd')

axis.set_xlabel('Adjucacion - Apertura Oferta (days)', fontsize=fontsize)
axis.set_ylabel('Number of tenders', fontsize=fontsize)

axis.set_yscale('log')

axis.text(0.10, 0.9, 'Median: {}\nMean: {} '.format(round(var.median(), 0), round(var.mean(), 2)),
                  transform=axis.transAxes, fontsize=fontsize)

# General
axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'processo_fechas_diff', 'png')
fig.savefig(path)
# -

# # Categories 
#
# * Distribution of categories of agencies (13), categories of items (14), type of procurement (17)
# * Distribution of time of contract execution (50)
# * Distribution of type of awarding system (39)

query = """select
id_llamado,
fecha_publicacion_convocatoria,
categoria,
complaint
from raw_labeled.proceso"""

categorias = pd.read_sql_query(query, con)

categorias.info()

print('Number of categories: ', categorias['categoria'].nunique())

# +
gr = categorias.groupby('categoria').agg(['count', 'sum'])['complaint']
gr['complaint rate'] = gr['sum'] / gr['count'] * 100
gr = gr.rename(columns={'count': 'number of tenders', 'sum': 'number of complaints'})
gr['perc of complaints'] = gr['number of complaints'] / gr['number of complaints'].sum() * 100

gr.sort_values(by='number of tenders', ascending=False)
# .count()[['id_llamado']]\
# .rename(columns={'id_llamado':'number of tenders'})\
# .sort_values(by='number of tenders', ascending=False)
# -

gr.sort_values(by='complaint rate', ascending=False)

gr = gr.sort_values(by='number of complaints', ascending=False)
gr['cum perc of complaints'] = gr['perc of complaints'].cumsum()
gr

gr_year = categorias.groupby(['categoria', categorias['fecha_publicacion_convocatoria'].dt.year]).agg(['count', 'sum'])['complaint']
gr_year['complaint rate'] = gr_year['sum'] / gr_year['count'] * 100
gr_year = gr_year.rename(columns={'count': 'number of tenders', 'sum': 'number of complaints'})
gr_year['perc of complaints'] = gr_year['number of complaints'] / gr_year['number of complaints'].sum() * 100
gr_year = gr_year.reset_index()

# +
number_of_categories = 10

main_categories = gr.index[:number_of_categories]

nrows = number_of_categories
ncols = 2
fontsize = 14

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(12, 5 * nrows))
# First plot
for r in range(nrows):
    for c in range(ncols):
        
        axis = ax[r][c]
        
        plt_data = gr_year.query(f"categoria == '{main_categories[r]}'")\
                          .query('fecha_publicacion_convocatoria > 2010')\
                          .query('fecha_publicacion_convocatoria < 2019')

        if c == 0:
            
            axis.plot(plt_data['fecha_publicacion_convocatoria'], 
                plt_data['number of tenders'], )
            
            axis.set_ylabel('Number of tenders', fontsize=fontsize)
            
            axis.set_title(main_categories[r][:30],fontsize=fontsize)
        else:
            
            axis.plot(plt_data['fecha_publicacion_convocatoria'], 
                plt_data['number of complaints'], color='orange')
            axis.set_ylabel('Number of complaints', fontsize=fontsize)
            
        axis.set_ylim([0, None])
         

axis.tick_params(axis='both', labelsize=12)
fig.tight_layout()
path = utils.path_to_shared('joaoc', 'imgs', 'processo_fechas_diff', 'png')
# fig.savefig(path)

# +

plt.plot(plt_data['fecha_publicacion_convocatoria'], 
        plt_data['number of tenders'], 
       )


# -

plt_data

# ## Correlation between amendments and complaints

import seaborn as sns

query = """
select *
from raw_labeled.tender_additional_info
"""

df = pd.read_sql_query(query, con)

df.columns

columns = ['number_of_total_complaints',
           'number_of_effective_complaints',
           'number_of_amendments_0',
           'number_of_amendments_1',
           'number_of_amendments_2',
           'bool_of_effective_complaints'
          ]

df = df.replace(0, np.nan).dropna(subset=columns, how='all').fillna(0)

len(df)

ax = sns.pairplot(df[columns], hue='bool_of_effective_complaints', diag_kind="kde")
path = utils.path_to_shared('joaoc', 'imgs', 'pairplot_complaints_amend', 'png')
# fig.savefig(path) # save it somewhere

ax.savefig(path)

# ### Complaints and ammendments per type of procurement 

query = """
select
    id_llamado,
    tipo_procedimiento,
    number_of_total_complaints,
    number_of_effective_complaints,
    number_of_amendments,
    number_of_amendments_1,
    number_of_amendments_2,
    case when number_of_total_complaints > 0 then 1 else 0 end bool_of_total_complaints,
    bool_of_effective_complaints,
    bool_of_amendments,
    bool_of_amendments_1,
    bool_of_amendments_2
from raw_labeled.proceso
"""

df = pd.read_sql_query(query, con).set_index('id_llamado')

number_of_complaints = df.groupby('tipo_procedimiento').sum().sort_values(by='number_of_effective_complaints', ascending=False)
number_of_complaints['number_of_cases'] = df.groupby('tipo_procedimiento').count()['number_of_total_complaints']

number_of_complaints['complaint_rate'] = number_of_complaints['bool_of_effective_complaints'] / number_of_complaints['number_of_cases'] * 100

number_of_complaints[['number_of_cases',  'bool_of_effective_complaints', 'complaint_rate']]

path = utils.path_to_shared('joaoc', 'data_output', 'complaint_rate_by_procurement_type', 'csv')
number_of_complaints.to_csv(path)

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










