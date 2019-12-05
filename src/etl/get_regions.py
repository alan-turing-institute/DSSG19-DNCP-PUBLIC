# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Script to macth municipalities and regions

# We need to map each tender that was submited from a municipality with its region so that we can compute the bias of the models at regional level.
#
# To do this we import data from the national statistics office in Paraguay and performed some data transformation so that we can use it for our problem. Also get data about IDH at region level. 
#
# The last step of this script is to upload the dataframe into database.
#
# Plus there is a function to add gender population at municipality level, not used but leave it here just in case it is useful for the future.

# ### Packages

import sys
import os
import unidecode
import nltk
from nltk.corpus import stopwords

import pandas as pd
pd.options.display.float_format = '{:20,.2f}'.format
pd.set_option('precision',0)

# %reload_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0, '../src/utils')
import utils 

# ### Map departamentos and distritos

# Import Department - Distrito map from Paraguay Statistics Institute
path = '/data/demographics_department_level/Distritos_Paraguay_Codigos_DGEEC.csv'
departamentos_distritos = pd.read_csv(path, encoding = 'latin')

# Subset data
departamentos_distritos = departamentos_distritos[['Descripción de Departamento','Descripción de Distrito']]

# +
# Lower case
departamentos_distritos = departamentos_distritos.apply(lambda x: x.astype(str).str.lower())

# Remove accents
departamentos_distritos['Descripción de Departamento'] = departamentos_distritos['Descripción de Departamento'].apply(unidecode.unidecode)
departamentos_distritos['Descripción de Distrito'] = departamentos_distritos['Descripción de Distrito'].apply(unidecode.unidecode)

# Remove stopwords
stop = set(stopwords.words('spanish'))
stop.remove('a')
stop.remove('e')
stop.remove('o')

departamentos_distritos['Descripción de Distrito'].apply(lambda x: [item for item in x if item not in stop])
departamentos_distritos['Descripción de Distrito'] = departamentos_distritos['Descripción de Distrito'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# -

departamentos_distritos.head()

# ### IDH Index

# IDH is a Human development Index developed by PNUD. It is a statistic composite index of life expectancy, education and per capita income indicar which are used to rank regions of Paraguay

# Table url
url ='https://es.wikipedia.org/wiki/Anexo:Departamentos_del_Paraguay_por_IDH'

table = pd.read_html(url, header = 0, decimal = ',', thousands = '.')
idh = table[2]

# +
# Lower case
idh = idh.apply(lambda x: x.astype(str).str.lower())

# Remove accents
idh['Departamento'] = idh['Departamento'].apply(unidecode.unidecode)
idh['Capital'] = idh['Capital'].apply(unidecode.unidecode)
# -

# ### Concat

# Concatenates department, municipality and idh

df = pd.merge(departamentos_distritos,idh, left_on = 'Descripción de Departamento', right_on ='Departamento')

df = df[['Departamento','Descripción de Distrito','IDH (2018)']]
df = df.rename(columns={'Departamento':'departament','Descripción de Distrito':'municipality','IDH (2018)':'idh'})
df.head()

# ### Insert to DB

df.head()

con = utils.connect_to_database()


# +
#df.to_sql('demographics',con,'semantic')

# +
# After this point sql/semantic/create_tenders_demographics.sql  should be run to match municipality in tenders
# -

# #### Get Data of Gender and age by municipality

# We didnt use this part in the bias analysis, leave it here just in case we need to pick it up later. 

def demographics(file):
    
    # Read file
    path = os.path.join('/data/demographics/',file)
    df = pd.read_csv(path, encoding='latin' )
    df_one_year = df[['#','2018']]
    
    # Discard age data
    df_gender = df_one_year[['#','2018']][df_one_year['#'].str.contains('0|5|10|15|20|25|35|45|55|65|75') == False]
    
    # Split column
    new = df_gender['#'].str.split('-', n=1,expand=True)
    df_gender['distrito'] = new[0]
    df_gender['genero'] = new[1]
    
    # subset columns
    df_gender[['distrito','genero','2018']]
    
    # reshufle data
    df_gender = df_gender.pivot_table(values = '2018', index= 'distrito',columns = 'genero')
    
    # table flatten
    df_gender.reset_index(level = 0, inplace=True)
    
    # append data to dataframe
    return (df_gender)


csv_files = list(os.listdir('/data/demographics/'))
data = pd.DataFrame()
for file in csv_files:
    data = data.append(demographics(file))
