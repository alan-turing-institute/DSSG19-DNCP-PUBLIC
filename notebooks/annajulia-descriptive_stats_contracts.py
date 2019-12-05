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

# # Descriptive statistics: contracts

# ## Settings

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
import sys
sys.path.insert(0, '../src/utils')
import utils 

# Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.max_columns = 999
# -

con = utils.connect_to_database()

schema = 'raw'
table_name = 'contrato'

# ## Basic 

# **1. Total number of contracts**

query = f"select count(*) as n_contracts from {schema}.{table_name}"

n_contracts = pd.read_sql_query(query, con)

# + {"variables": {" n_contracts.iloc[0]['n_contracts'] ": "<p><strong>NameError</strong>: name &#39;n_contracts&#39; is not defined</p>\n"}, "cell_type": "markdown"}
# Total number of contracts: {{ n_contracts.iloc[0]['n_contracts'] }}
# -

query = f"select count(distinct id_llamado) as n_tenders from {schema}.{table_name}"

distinct_tenders = pd.read_sql_query(query, con)

# + {"variables": {" distinct_tenders.iloc[0]['n_tenders'] ": "<p><strong>NameError</strong>: name &#39;distinct_tenders&#39; is not defined</p>\n"}, "cell_type": "markdown"}
# Total number of contracts: {{ distinct_tenders.iloc[0]['n_tenders'] }}
# -

# **2. Null values**
#
# Check for Null values across the table.

utils.check_null_values(schema=schema, table=table_name, con=con).query('nulls > 0').sort_values('nulls', ascending=False)

# **3. Make sure all tender processes already awarded have a corresponding contract**

query = f"with tender_contract as (" \
        f" select a.id_llamado, a.etapa_licitacion, b.adjudicacion_slug, b.contrato_slug from raw.proceso as a" \
        f" left join {schema}.{table_name} as b" \
        f" on a.id_llamado = b.id_llamado" \
        f" where lower(a._etapa_licitacion) = 'adj'" \
    f")" \
    f" select * from tender_contract" \
    f" where contrato_slug is null"

no_contracts = pd.read_sql_query(query, con)

# Number of tenders already awarded without a corresponding contract: {{ no_contracts.shape[0] }}

# **4. Check how many contracts per tender and whether type is by lots or items (sistema_adjudicacion).**

query = """with total_contracts as (
    select id_llamado, count(*) as n_contracts from raw.contrato
    group by id_llamado
    order by n_contracts desc
    ), awarded_processes as (
    select * from raw.proceso
    where lower(_etapa_licitacion) = \'adj\'
    )
    select awarded_processes.id_llamado, awarded_processes.sistema_adjudicacion, 
    total_contracts.n_contracts
    from awarded_processes
    left join total_contracts
    on awarded_processes.id_llamado = total_contracts.id_llamado;"""

award_system = pd.read_sql_query(query, con)

award_system.head(10)

award_system['n_contracts'].hist(by=award_system['sistema_adjudicacion'], figsize=(12, 8), bins=50)

award_system[['sistema_adjudicacion', 'n_contracts']].groupby('sistema_adjudicacion')\
.agg('sum').sort_values('n_contracts')

award_system.query('sistema_adjudicacion == \'Por Total\' & n_contracts > 1').head()

# **5. Quantity of contracts and corresponding money awarded by supplier (historical data)**

query = """with supplier as (
    select proveedor, proveedor_slug, sum(monto_adjudicado) as total_awarded from raw.contrato
    group by proveedor, proveedor_slug
    ), total_contracts as (
    select proveedor_slug, count(*) as n_contracts from raw.contrato
    group by proveedor_slug
    )
    select proveedor, supplier.proveedor_slug, total_awarded, n_contracts, total_awarded/n_contracts as avg_contract_award from supplier
    left join total_contracts 
    on supplier.proveedor_slug = total_contracts.proveedor_slug
    --order by total_awarded desc;
    order by avg_contract_award desc;"""

pd.read_sql_query(query, con)

# **6. Most awarded supplier by year and category**

query = """
with summary as (
    select categoria, extract(year from fecha_contrato) as cyear, proveedor, count(*) as n_contracts, 
    sum(monto_adjudicado) as amount_awarded, moneda
    from raw.contrato
    where fecha_contrato is not null
    group by categoria, cyear, moneda, proveedor
), ranking as (
    select categoria, cyear, moneda, proveedor, n_contracts, amount_awarded, row_number() over(partition by categoria, cyear, moneda 
    order by n_contracts desc) as rk from summary
)
select categoria, cyear, moneda, proveedor, n_contracts, amount_awarded, amount_awarded/n_contracts as avg_award from ranking
where rk = 1;
"""

most_awarded = pd.read_sql_query(query, con)

most_awarded.head()

most_awarded.groupby(['categoria', 'moneda'])['proveedor'].value_counts()

most_awarded.head()

query = """select id_llamado, id_contrato, proveedor, categoria, fecha_contrato, extract(year from fecha_contrato) as cyear,
    monto_adjudicado, moneda
from raw.contrato"""

contracts = pd.read_sql_query(query, con)

contracts = contracts[contracts.cyear <= 2019 & contracts.cyear >= 2009]
contracts['cyear'] = pd.to_numeric(contracts['cyear'], downcast='integer')
contracts.head()

# **Evolution of number of contracts**

# +
nrow = 1
ncol = 1
fontsize = 14

gr = contracts.groupby('cyear').count()[['id_llamado']].reset_index()

fig, ax = plt.subplots(nrow, ncol, figsize=(12, 5))

ax.bar(gr['cyear'], gr['id_llamado'])
ax.set_xlabel('Year')
ax.set_ylabel('Number of contracts')
ax.set_title('Evolution of number of contracts', fontsize=fontsize)
ax.set_xticks(np.arange(min(gr['cyear']), max(gr['cyear']) + 1, 1))
# -

# **Average money awarded by contract along the years**

# +
groupby = contracts.groupby(['categoria', 'moneda'])

for gr, df in groupby:
    aux_df = df.groupby('cyear')['monto_adjudicado'].agg('mean')  
    aux_df.plot('bar', figsize=(12, 8))
    plt.title(f'{gr[0]} - {gr[1]}')
    plt.xlabel('Year')
    plt.ylabel('Average awarded money')
    plt.show()    
# -

contracts.groupby(['cyear'])[['id_contrato']].nunique().plot(kind='bar', figsize=(12, 8))
plt.title('Evolution of number of contracts')
plt.xlabel('Year')
plt.ylabel('Number of contracts')
plt.show()


