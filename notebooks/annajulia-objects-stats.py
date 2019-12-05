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

# # Objects stats and concentration of complaints

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0, '../src/utils')
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import utils 

# Packages
import pandas as pd
pd.options.display.max_columns = 999
# -

con = utils.connect_to_database()

tab_path = 'data_output'
user_name = 'annajulia'

# - Concentration table of procurement type
# - Concentration of agencies
# - Top20 agencies
# - Concentration of product categories
# - Top20 products
# - Concentration of items
# - Top20 items
# - Concentration of type of agencies
# - Top20 type of agencies
# - Concentration of municipalities
# - Top20 municipalities

# ## Tenders

query = """select id_llamado, convocante, tipo_entidad, categoria, tipo_procedimiento, etapa_licitacion, 
fecha_publicacion_convocatoria, monto_estimado_convocatoria, monto_total_adjudicado, moneda, vigencia_contrato, 
bool_of_effective_complaints, bool_of_amendments, number_of_amendments, 
bool_of_amendments_0, number_of_amendments_0, bool_of_amendments_1, number_of_amendments_1,
bool_of_amendments_2, number_of_amendments_2
from raw_labeled.proceso"""

tenders = pd.read_sql_query(query, con)

tenders.head()

# ### Basic checks

check_fields = ['convocante', 'categoria', 'tipo_procedimiento', 'tipo_entidad']

# Define number of top elements we're interested in
n_top = 20

for field in check_fields:
    unique_list = tenders[field].unique()
    grouped_df = tenders.groupby([field])['id_llamado'].agg(['count']).rename(columns={'count': 'n_tenders'})\
    .sort_values('n_tenders', ascending=False)
    grouped_df['perc_tenders'] = grouped_df['n_tenders'] / grouped_df['n_tenders'].sum() * 100
    
    print(f"-----------------\nNumber of unique {field}: {len(unique_list)}")
    print(f"Number of documents by {field}")
    display(HTML(grouped_df.head(10).to_html()))

# ### Amount of tenders by agency

# Basic facts and figures about distribution of tenders among agencies.

# +
agency_distr = tenders.groupby(['convocante'])['id_llamado'].nunique().to_frame().reset_index()\
.sort_values('id_llamado', ascending=False).rename(columns={'id_llamado': 'n_tenders'})

agency_distr['perc_tenders'] = agency_distr['n_tenders'] / agency_distr['n_tenders'].sum() * 100

agency_distr.head(10)

# +
nrows = 1 # Number of rows
ncols = 1 # Number of columns
fontsize = 14 # General fontsize

plot_df = agency_distr.head(n_top)
plot_df['convocante'] = plot_df['convocante'].str.slice(stop=30)

# Create a subplot grid
# ax is a list with the dimensions [nrows][ncols]
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(10, 4)) # (width, heigth)
ax.barh(plot_df['convocante'], plot_df['perc_tenders'], color='orange')
ax.set_xlabel( '% tenders', 
                fontsize=fontsize)
ax.set_ylabel('Public Agency', 
                fontsize=fontsize)
ax.set_title(f'Top {n_top} Public Agencies in total amount of tenders', fontsize=fontsize)

#path = utils.path_to_shared('anna', 'imgs', 'top_20_agencies_tenders', 'png')
# fig.savefig(path)
# -

accum_col_names = ['Number of tenders without complaints',
                  'Number of tenders with complaints',
                  'Total number of tenders',
                  'Complaint rate',
                  'Percentage of complaints (overall)',
                  'Cumulative percentage']

# ### Accumulation of complaints in certain public agencies

# Are complaints accumulated in certain agencies?

groupby_col = 'convocante'
agency_compl = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                          bool_col='bool_of_effective_complaints', count_col='id_llamado')

# Export table

# +
n = 10
col_names = [groupby_col.capitalize()] + accum_col_names

#img_path = utils.path_to_shared(user_name, tab_path, f'top_{n}_agencies_concentration_complaints', 'csv')

output_df = agency_compl.head(n)
output_df.columns = col_names
#output_df.to_csv(img_path, index=False)
output_df
# -

# **Insight**: 9 public agencies accumulate 25% of the total amount of complaints.

# +
plot_col = 'percentage'

plot_df = agency_compl.sort_values(plot_col, ascending=False).head(n_top)
plot_df[groupby_col] = plot_df[groupby_col].str.slice(stop=30)

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(10, 4)) 
ax.barh(plot_df[groupby_col], plot_df[plot_col])
ax.set_xlabel('% complaints', 
                fontsize=fontsize)
ax.set_ylabel('Public Agency', 
                fontsize=fontsize)
ax.set_title(f'Top {n_top} Public Agencies in {plot_col} of complaints', fontsize=fontsize)

#path = utils.path_to_shared('anna', 'imgs', f'top_{n_top}_agencies_complaints', 'png')
#fig.savefig(path)

# +
plot_col = 'n_bool_1'

plot_df = agency_compl.sort_values(plot_col, ascending=False).head(n_top)
plot_df[groupby_col] = plot_df[groupby_col].str.slice(stop=30)

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(10, 4)) 
ax.barh(plot_df[groupby_col], plot_df[plot_col])
ax.set_xlabel( 'Percentage of tenders with complaints', 
                fontsize=fontsize)
ax.set_ylabel('Public agency', 
                fontsize=fontsize)
ax.set_title(f'Top {n_top} public agencies in total of complaints', fontsize=fontsize)

#path = utils.path_to_shared('anna', 'imgs', f'top_{n_top}_agencies_total_complaints', 'png')
#fig.savefig(path)
# -

# ## Accumulation of complaints in certain tender categories

groupby_col = 'categoria'
category_compl = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                            bool_col='bool_of_effective_complaints', 
                                            count_col='id_llamado')
category_compl.head()

# +
n = 10
col_names = [groupby_col.capitalize()] + accum_col_names

#img_path = utils.path_to_shared(user_name, tab_path, f'top_{n}_tender_categories_concentration_complaints', 'csv')

output_df = category_compl.head(n)
output_df.columns = col_names
#output_df.to_csv(img_path, index=False)
output_df

# +
plot_col = 'percentage'

plot_df = category_compl.sort_values(plot_col, ascending=False).head(20)
plot_df[groupby_col] = plot_df[groupby_col].str.slice(stop=30)

fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(10, 4)) 
ax.barh(plot_df['categoria'], plot_df[plot_col])
ax.set_xlabel( 'Percentage of tenders with complaints', 
                fontsize=fontsize)
ax.set_ylabel('Tender category', 
                fontsize=fontsize)
ax.set_title('Top 20 tender categories in percentage of tender with complaints', fontsize=fontsize)

#path = utils.path_to_shared('anna', 'imgs', f'top_{n_top}_categories_complaints', 'png')
#fig.savefig(path)
# -

# ## Complaints by type of process

groupby_col = 'tipo_procedimiento'
type_tender = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                         bool_col='bool_of_effective_complaints', count_col='id_llamado')
type_tender.head()

# +
n = 10
col_names = [groupby_col.capitalize().replace('_', ' ')] + accum_col_names

output_df = type_tender.head(n)
output_df.columns = col_names
#img_path = utils.path_to_shared(user_name, tab_path, f'top_{n}_tender_process_type_concentration_complaints', 'csv')
#output_df.to_csv(img_path, index=False)
output_df
# -

# More than 50% of complaints refer to Licitacion Publica.

# ### Accumulation of complaints by type of agency

groupby_col = 'tipo_entidad'
agency_type = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                         bool_col='bool_of_effective_complaints', count_col='id_llamado')

# +
col_names = [groupby_col.capitalize().replace('_', ' ')] + accum_col_names

output_df = agency_type
output_df.columns = col_names
#img_path = utils.path_to_shared(user_name, tab_path, f'agency_type_concentration_complaints', 'csv')
#output_df.to_csv(img_path, index=False)
output_df
# -

# ### Municipalities

groupby_col = 'convocante'
municipality_tenders = tenders.query("tipo_entidad == 'Municipalidades'")
munic_df = utils.calculate_bool_stats(df=municipality_tenders, groupby_col=groupby_col,
                                      bool_col='bool_of_effective_complaints', count_col='id_llamado')

# +
n = 10
col_names = [groupby_col.capitalize().replace('_', ' ')] + accum_col_names

output_df = munic_df.head(n)
output_df.columns = col_names
#img_path = utils.path_to_shared(user_name, tab_path, f'top_{n}_municipalities_concentration_complaints', 'csv')
#output_df.to_csv(img_path, index=False)
output_df
# -

years = range(2009, 2020, 1)
for year in years:
    limit_date = f'{year}-01-01'
    aux_df = tenders.query(f"fecha_publicacion_convocatoria > '{limit_date}'")
    n_types = aux_df.tipo_procedimiento.nunique()
    print(f'For year {year}, number of unique types of procurement processes: {n_types}')

# ## Amendments by public agency

bool_col = 'bool_of_amendments'
count_col = 'id_llamado'

groupby_col = 'convocante'
stats_df = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                      bool_col=bool_col, 
                                      count_col=count_col)
stats_df.head(n_top)

# ## By category of process

groupby_col = 'categoria'
stats_df = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                      bool_col=bool_col, 
                                      count_col=count_col)
top_categories_0 = stats_df[groupby_col][:5]
stats_df.head(5)

# The first 5 categories of processes accumulate 50% of total amendments.

groupby_col = 'categoria'
stats_df = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                      bool_col='bool_of_amendments_1', 
                                      count_col=count_col)
top_categories_1 = stats_df[groupby_col][:5]
stats_df.head(5)

# +
groupby_col = 'categoria'
stats_df = utils.calculate_bool_stats(df=tenders, groupby_col=groupby_col, 
                                      bool_col='bool_of_amendments_2', 
                                      count_col=count_col)
top_categories_2 = stats_df[groupby_col][:5]

stats_df.head(5)
# -

len(set(top_categories_0.tolist() + top_categories_1.tolist() + top_categories_2.tolist()))

# ## Items

query = """select id_llamado, producto_nombre_catalogo, precio_unitario_estimado, cantidad, unidad_medida, presentacion,
fecha_publicacion, bool_of_effective_complaints, bool_of_amendments, bool_of_amendments_0, bool_of_amendments_1,
bool_of_amendments_2
from raw_labeled.item_solicitado"""

items_df = pd.read_sql_query(query, con)

items_df.head()

print(f"Number of unique products: {items_df.producto_nombre_catalogo.nunique()}")

count_col = 'id_llamado'
groupby_col = 'producto_nombre_catalogo'
stats_df = utils.calculate_bool_stats(df=items_df, groupby_col=groupby_col, 
                                      bool_col='bool_of_effective_complaints', 
                                      count_col=count_col)
stats_df.producto_nombre_catalogo.nunique()

# +
n = 10
col_names = [groupby_col.capitalize().replace('_', ' ')] + accum_col_names

output_df = stats_df.head(n)
output_df.columns = col_names
#img_path = utils.path_to_shared(user_name, tab_path, f'top_{n}_product_concentration_complaints', 'csv')
#output_df.to_csv(img_path, index=False)
output_df
# -

stats_df.head(10)

groupby_col = 'presentacion'
stats_df = utils.calculate_bool_stats(df=items_df, groupby_col=groupby_col, 
                                      bool_col='bool_of_effective_complaints', 
                                      count_col=count_col)

stats_df.head(10)


