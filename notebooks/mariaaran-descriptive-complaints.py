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
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
# -

# ### Start connection with db
#
#

con = utils.connect_to_database()

# ### Querying with pandas

# +
# SQL queries to extract data from database


query_complaints = """
select
count(distinct tenders.id_llamado) quantity_tenders,
sum(case when _tipo_resultado = 'a_favor' then 1 else 0 end ) quantity_complaints
from 
raw.proceso tenders 
left join
raw.proceso_juridico complaints
on tenders.id_llamado = complaints.id_llamado;
"""

# Complaints by '_tipo_resultado'
query_a_favor = """
select
resultado,
count(*) quantity
from 
raw.proceso_juridico
 where _tipo_resultado = 'a_favor'
group by  resultado
order by quantity asc;
"""

query_en_contra = """
select
resultado,
count(*) quantity
from 
raw.proceso_juridico
 where _tipo_resultado = 'en_contra'
group by  resultado
order by quantity asc;
"""

# Complaints by year/month
query_time = """
select
extract(year from fecha_resolucion_proceso) * 100 + extract (month from fecha_resolucion_proceso) yearmonth,
sum(case when _tipo_resultado = 'a_favor' then 1 else 0 end) favorable,
sum(case when _tipo_resultado = 'en_contra' then 1 else 0 end) not_favorable,
sum(case when _tipo_resultado is null then 1 else 0 end) result_null,
count(*) total_complaints
from
raw.proceso_juridico
group by yearmonth
order by yearmonth desc;
"""

query_amendments = """

with proceso_juridico_flatten as (
select
id_llamado,
count(distinct _tipo_resultado) q_distinct_complaints_results,
sum(case when _tipo_resultado = 'a_favor' then 1 else 0 end) q_effective_complaints,
sum(case when _tipo_resultado = 'en_contra' then 1 else 0 end) q_noneffective_complaints,
min(fecha_resolucion_proceso) min_fecha_resolucion_proceso,
max(fecha_resolucion_proceso) max_fecha_resolucion_proceso
from
raw.proceso_juridico
group by 
id_llamado)

, data_to_plot as (
select
tenders.id_llamado,
tenders.monto_total_adjudicado,
tenders.tipo_procedimiento,
	max(review.audit_date)::date - min(review.reception_date)::date Reviewing_time_in_days,
count(distinct review.id) Reviewing_quantity,
sum(case when (amendments.tipo_documento = 'Pliego de bases y Condiciones' and tenders.fecha_publicacion_convocatoria > amendments.fecha_archivo) then 1 else 0 end) Q_PBC_before_publication_date,	sum(case when (amendments.tipo_documento = 'Adenda' and tenders.fecha_publicacion_convocatoria > amendments.fecha_archivo) then 1 else 0 end) Q_Adendas_before_publication_date,
tenders.fecha_publicacion_convocatoria,
sum(case when (amendments.tipo_documento = 'Pliego de bases y Condiciones' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo < complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_PBC_after_publication_date,
sum(case when (amendments.tipo_documento = 'Adenda' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo < complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_Adendas_after_publication_date_before_complaint,
complaints.min_fecha_resolucion_proceso, 
coalesce(complaints.q_distinct_complaints_results,0) q_distinct_complaints_results,
complaints.q_effective_complaints,
complaints.q_noneffective_complaints,
sum(case when (amendments.tipo_documento = 'Pliego de bases y Condiciones' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo > complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_PBC_after_publication_date_after_complaint,
sum(case when (amendments.tipo_documento = 'Adenda' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo > complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_Adendas_after_publication_date_after_complaint
from
raw.proceso tenders
left join
raw.tender_record review
on tenders.id_llamado = review.llamado_id
left join
raw.pbc_adenda amendments
on tenders.id_llamado = amendments.id_llamado
left join
proceso_juridico_flatten complaints
on tenders.id_llamado = complaints.id_llamado
group by
tenders.id_llamado,
tenders.monto_total_adjudicado,
tenders.tipo_procedimiento,
tenders.fecha_publicacion_convocatoria,
complaints.min_fecha_resolucion_proceso, 
complaints.q_distinct_complaints_results,
complaints.q_effective_complaints,
complaints.q_noneffective_complaints)

select * from data_to_plot;
"""

# -


# Data to pandas Dataframes
complaints = pd.read_sql_query(query_complaints, con)
a_favor = pd.read_sql_query(query_a_favor, con)
en_contra = pd.read_sql_query(query_en_contra, con)
time = pd.read_sql_query(query_time, con)
adendas = pd.read_sql_query(query_amendments, con)

# Change data type
time = time.dropna(subset=['yearmonth'])
time['yearmonth'] = time['yearmonth'].apply(lambda x: datetime.strptime(str(x), '%Y%m.0'))

# # Plots

# ## Complaints

complaints.head()

# Effective complaints
g = sns.FacetGrid(a_favor)
svm = g.map(plt.barh, 'resultado', 'quantity' ,  color = 'green')
plt.title('Effective complaints')
plt.xlabel('Quantity of cases')
plt.ylabel('Categories')

# Non-effective complaints
g = sns.FacetGrid(en_contra)
noneffective = g.map(plt.barh, 'resultado', 'quantity' ,  color = 'red')
plt.title('Non effective complaints')
plt.xlabel('Quantity of cases')
plt.ylabel('Categories')


# +
# Complaints in time

fig, ax = plt.subplots(figsize = (12,5))
sns.lineplot(data = time[['yearmonth','favorable']], x = 'yearmonth', y = 'favorable', ax=ax)
sns.lineplot(data = time[['yearmonth','not_favorable']], x = 'yearmonth', y = 'not_favorable', ax=ax)
ax.set_title('Complaints in time')
ax.set_xlabel('Years')
ax.set_ylabel('Cases')
ax.legend(loc = 'best',  labels = ['Favorable','Non favorable'])

# -

# ## Amendments 1: Related to tender process (before complaints)

# Total of tenders = 126.238
#
# Total tenders with at least 1 amendment = 1.279
#
# Mean of adenda documents per tender = 20. Median of adenda documents per tender = 20.

# General descriptive stats
adendas['q_adendas_after_publication_date_before_complaint'].describe()

# descriptive stats of the tenders that have adendas
tenders_w_amendment_1 = adendas.loc[adendas['q_adendas_after_publication_date_before_complaint'] > 0]
tenders_w_amendment_1['q_adendas_after_publication_date_before_complaint'].describe()

# +
## Amendments 1
# Distribution of quantity and money

nrows = 1 # Number of rows
ncols = 2 # Number of columns
fontsize = 14 # General fontsize
num_bins = 200
x = tenders_w_amendment_1['q_adendas_after_publication_date_before_complaint'].loc[tenders_w_amendment_1['q_adendas_after_publication_date_before_complaint'] < 1886]


# Create a subplot grid
fig, (ax1,ax2) = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 4)) # (width, heigth)

# Plot 1
#Hist
n,bins,patches = ax1.hist(x, bins = num_bins, density = 1)

#Set Xlim
ax1.set_xlim(0,30)

# Set axis labels
ax1.set_title('Amendments 1: After tender', fontsize=fontsize)
ax1.set_xlabel( 'Quantity of adenda documents' , fontsize=fontsize)
ax1.set_ylabel('%', fontsize=fontsize)

#Plot 2
ax2.boxplot(x)


fig.tight_layout()
plt.show()
# -

# ### Amendments 1: Tipo procedimiento

# Total of tipo procedimiento = 34
#
# Choose the ones they mentioned that are the tenders that concentrate the most quantity of tenders.

adendas['tipo_procedimiento'].nunique()

adendas.groupby('tipo_procedimiento').count().sort_values(by = 'id_llamado', ascending = False)

# +
## Amendments 1
# Distribution of quantity per tipo contrato

nrows = 2 # Number of rows
ncols = 2 # Number of columns
fontsize = 14 # General fontsize
num_bins = 200
adendas_licitacion_publica = tenders_w_amendment_1['q_adendas_after_publication_date_before_complaint'].loc[tenders_w_amendment_1['tipo_procedimiento'] == 'Licitación Pública Nacional']
adendas_contratacion_directa = tenders_w_amendment_1['q_adendas_after_publication_date_before_complaint'].loc[tenders_w_amendment_1['tipo_procedimiento'] == 'Contratación Directa']
adendas_concurso_ofertas = tenders_w_amendment_1['q_adendas_after_publication_date_before_complaint'].loc[tenders_w_amendment_1['tipo_procedimiento'] == 'Concurso de Ofertas']
adendas_excepcion = tenders_w_amendment_1['q_adendas_after_publication_date_before_complaint'].loc[tenders_w_amendment_1['tipo_procedimiento'] == 'Contratación por Excepción']

# Create a subplot grid
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=nrows, ncols=ncols,figsize=(12, 6)) # (width, heigth)

# Plot 1
#Hist
n,bins,patches = ax1.hist(adendas_licitacion_publica, bins = num_bins, density = 1)

#Set Xlim
ax1.set_xlim(0,20)

# Set axis labels
ax1.set_title('Amendments 1: Licitacion publica', fontsize=fontsize)
ax1.set_xlabel( 'Quantity of adenda documents' , fontsize=fontsize)
ax1.set_ylabel('%', fontsize=fontsize)

#Plot 2
#Hist
n,bins,patches = ax2.hist(adendas_contratacion_directa, bins = num_bins, density = 1)

#Set Xlim
ax2.set_xlim(0,20)

# Set axis labels
ax2.set_title('Amendments 1: contratacion directa', fontsize=fontsize)
ax2.set_xlabel( 'Quantity of adenda documents' , fontsize=fontsize)
ax2.set_ylabel('%', fontsize=fontsize)

#Plot 3
#Hist
n,bins,patches = ax3.hist(adendas_concurso_ofertas, bins = num_bins, density = 1)

#Set Xlim
ax3.set_xlim(0,20)

# Set axis labels
ax3.set_title('Amendments 1: Concurso ofertas', fontsize=fontsize)
ax3.set_xlabel( 'Quantity of adenda documents' , fontsize=fontsize)
ax3.set_ylabel('%', fontsize=fontsize)

#Plot 4
#Hist
n,bins,patches = ax4.hist(adendas_excepcion, bins = num_bins, density = 1)

#Set Xlim
ax4.set_xlim(0,30)

# Set axis labels
ax4.set_title('Amendments 1: Excepcion', fontsize=fontsize)
ax4.set_xlabel( 'Quantity of adenda documents' , fontsize=fontsize)
ax4.set_ylabel('%', fontsize=fontsize)

fig.tight_layout()
plt.show()
# -

# ### Amendments 1: Reviewing time

adendas['reviewing_time_in_days'].describe()

adendas.head()

# +
## Reviwing time
# Distribution of quantity and money

nrows = 1 # Number of rows
ncols = 1 # Number of columns
fontsize = 14 # General fontsize
num_bins = 300
x = adendas['reviewing_time_in_days']

# Create a subplot grid
fig, ax1 = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 4)) # (width, heigth)

# Plot 1
#Hist
n,bins,patches = ax1.hist(x, bins = num_bins, density = 1)

#Set Xlim
ax1.set_xlim(0,40)

# Set axis labels
ax1.set_title('Reviewing time', fontsize=fontsize)
ax1.set_xlabel( 'Days' , fontsize=fontsize)
ax1.set_ylabel('%', fontsize=fontsize)

fig.tight_layout()
plt.show()
# -

adendas['q_distinct_complaints_results'].describe()

# +
## Reviwing time
# Distribution of quantity and money

nrows = 2 # Number of rows
ncols = 2 # Number of columns
fontsize = 14 # General fontsize
num_bins = 300
x_w_adendas = adendas['reviewing_time_in_days'].loc[adendas['q_adendas_after_publication_date_before_complaint'] >=1]
x_wo_adendas = adendas['reviewing_time_in_days'].loc[adendas['q_adendas_after_publication_date_before_complaint'] == 0]
x_w_complaints = adendas['reviewing_time_in_days'].loc[adendas['q_distinct_complaints_results'] >=1]
x_wo_complaints = adendas['reviewing_time_in_days'].loc[adendas['q_distinct_complaints_results'] == 0]


# Create a subplot grid
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=nrows, ncols=ncols,figsize=(12, 6)) # (width, heigth)

# Plot 1
#Hist
n,bins,patches = ax1.hist(x_w_adendas, bins = num_bins, density = 1, color = 'blue')

#Set Xlim
ax1.set_xlim(0,40)

# Set axis labels
ax1.set_title('Reviewing time - tenders w adendas', fontsize=fontsize)
ax1.set_xlabel( 'Days' , fontsize=fontsize)
ax1.set_ylabel('%', fontsize=fontsize)

# Plot 2
#Hist
n,bins,patches = ax2.hist(x_wo_adendas, bins = num_bins, density = 1,  color = 'lightblue')

#Set Xlim
ax2.set_xlim(0,40)

# Set axis labels
ax2.set_title('Reviewing time - tenders w/o adendas', fontsize=fontsize)
ax2.set_xlabel( 'Days' , fontsize=fontsize)
ax2.set_ylabel('%', fontsize=fontsize)

# Plot 3
#Hist
n,bins,patches = ax3.hist(x_w_complaints, bins = num_bins, density = 1, color = 'red')

#Set Xlim
ax3.set_xlim(0,40)

# Set axis labels
ax3.set_title('Reviewing time - tenders w complaints', fontsize=fontsize)
ax3.set_xlabel( 'Days' , fontsize=fontsize)
ax3.set_ylabel('%', fontsize=fontsize)

# Plot 4
#Hist
n,bins,patches = ax4.hist(x_wo_complaints, bins = num_bins, density = 1,  color = 'pink')

#Set Xlim
ax4.set_xlim(0,40)

# Set axis labels
ax4.set_title('Reviewing time - tenders w/o complaints', fontsize=fontsize)
ax4.set_xlabel( 'Days' , fontsize=fontsize)
ax4.set_ylabel('%', fontsize=fontsize)

fig.tight_layout()
plt.show()

# -

# ### Amendments 2: After complaints

# +
## Amendments 2
# Distribution of quantity and money

nrows = 1 # Number of rows
ncols = 2 # Number of columns
fontsize = 14 # General fontsize
num_bins = 200
x = adendas['q_adendas_after_publication_date_after_complaint']

# Create a subplot grid
fig, (ax1,ax2) = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 4)) # (width, heigth)

# Plot 1
#Hist
n,bins,patches = ax1.hist(x, bins = num_bins, density = 1)

#Set Xlim
ax1.set_xlim(0,5)

# Set axis labels
ax1.set_title('Amendments 2: After Complaint', fontsize=fontsize)
ax1.set_xlabel( 'Quantity of adenda documents' , fontsize=fontsize)
ax1.set_ylabel('%', fontsize=fontsize)


#Plot 2
ax2.boxplot(x)

fig.tight_layout()
plt.show()
# -

# ## Plot template

fig.tight_layout()
path = utils.path_to_shared('mariaaran', 'imgs', 'complaints', 'png')
# fig.savefig(path) # save it somewhere

# ### Creating path to save data in shared folder

# +
#path = utils.path_to_shared('mariaaran', 'imgs', 'effective_complaints', 'png')
#path = utils.path_to_shared('mariaaran', 'imgs', 'non_effective_complaints', 'png')
#path = utils.path_to_shared('mariaaran', 'imgs', 'complaints_time', 'png')

# +
#path = utils.path_to_shared('mariaaran', 'imgs', 'effective_complaints', 'png')
#svm.savefig(path)

# +
#path = utils.path_to_shared('mariaaran', 'imgs', 'non_effective_complaints', 'png')
#noneffective.savefig(path)

# +
#path = utils.path_to_shared('mariaaran', 'imgs', 'complaints_time', 'png')
#fig.savefig(path)
