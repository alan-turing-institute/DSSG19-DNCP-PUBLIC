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

# # Check PDFs documents

# ## Settings

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0, '../src/utils')
import utils 

# Packages
import pandas as pd
import os
pd.options.display.max_columns = 999
# -

con = utils.connect_to_database()

data_path = '/data/original_data/docs/'
new_data_path = '/data/original_data/docs_0207/'
pbc_first = '/data/original_data/first_pbc/'

# ## List PDF files

# +
pdf_files = os.listdir(data_path)
new_files = os.listdir(new_data_path)

files_list = [f for f in set(pdf_files + new_files)]
# -

print(f"Number of files availables: {len(files_list)}")

# Number of first version of PBCs:

pbc_first_files = os.listdir(pbc_first)
len(pbc_first_files)

# ## List files available in the DB

query = """select tenders.id_llamado as id, documents.nombre_archivo as file_name, documents.tipo_documento as doc_type
from raw.proceso as tenders
left join raw.pbc_adenda as documents
on tenders.id_llamado = documents.id_llamado
"""

db_files = pd.read_sql_query(query, con)

print(f"Number of unique tender ids: {db_files.id.nunique()}")
print(f"Number of unique file names: {db_files.file_name.nunique()}")

print(f"Number of tenders without a corresponding file: {db_files[db_files['file_name'].isnull()].shape[0]}")

db_files.head()

reshaped_files = db_files.groupby(['id', 'doc_type']).agg('count').reset_index().rename(columns={'file_name': 'n_docs'})

reshaped_files = reshaped_files.pivot(index='id', columns='doc_type', values='n_docs')
reshaped_files.head()

reshaped_files.describe()

reshaped_files['has_pbc'] = reshaped_files['Pliego de bases y Condiciones'] >= 1
reshaped_files['has_inv'] = reshaped_files['Carta de Invitación'] >= 1

reshaped_files.has_pbc.value_counts()

print(f"Number of tenders without a corresponding PBC file: "
      f"{reshaped_files[reshaped_files.has_pbc == False].shape[0]}")

strange_cases = reshaped_files[(reshaped_files.has_pbc == False) & (reshaped_files.has_inv == False)]

print(f"Number of tenders without neither a corresponding PBC file nor a Carta de Invitacion: "
      f"{strange_cases.shape[0]}")

# Double check that tables _proceso_ and pbc_adenda have common tenders identifiers.

query = """select id_llamado from raw.proceso"""

id_proceso = pd.read_sql_query(query, con)

query = """select id_llamado from raw.pbc_adenda""" 

id_doc = pd.read_sql_query(query, con)

print(f"Number of tender IDs without file in the docs table: {len(set(id_proceso.id_llamado) - set(id_doc.id_llamado))}")

print(f"Number of docs IDs without linked tender process: {len(set(id_doc.id_llamado) - set(id_proceso.id_llamado))}")

# **Documents for each tender process**

db_files.groupby('id')['file_name'].agg({'count'}).rename(columns={'count': 'n_docs'})\
.sort_values('n_docs', ascending=False).head(10)

# ## Check differences

doc_not_db = set(files_list) - set(db_files.file_name)
print(f"Number of docs in the folder but not in the DB: {len(doc_not_db)}")

db_not_doc = set(db_files.file_name) - set(files_list)
print(f"Number of files as DB rows not in docs folder: {len(db_not_doc)}")

[doc for doc in db_not_doc][:10]

print(f"Percentage of missing docs: {round(len(db_not_doc) / len(db_files.file_name) * 100, 2)} %")


