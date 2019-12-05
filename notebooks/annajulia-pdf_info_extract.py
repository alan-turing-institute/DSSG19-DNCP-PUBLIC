# # Extract information from PDF documents

# ## Imports

# +
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath('annajulia-pdf_info_extract')).parent.parent / 'src')
pipeline_path = str(Path(os.path.abspath('annajulia-pdf_info_extract')).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys

from utils.utils import connect_to_database
from documents.extract_information import check_or_create_output_file, load_documents_to_database

# Packages
import csv
import datetime as dt
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import re
from tika import parser
from tqdm import tqdm
pd.options.display.max_columns = 999
# -

import PyPDF2
from PyPDF2 import PdfFileReader

# ### PDFminer imports

# +
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
# Import this to raise exception whenever text extraction from PDF is not allowed
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator
# -

# ## Settings

docs_path = '/data/original_data/first_pbc'
txt_folder = '/data/shared_data/data/raw/txt_extracted/'
output_file = "{date}_documents_extractability_{method}.csv"
header = "filename,is_extractable,number_of_pages\n"
method = "tika"
all_pbc_path = '/data/original_data/docs'
all_pbc_path2 = '/data/original_data/docs_0207'

# ### Start connection with db

con = connect_to_database()


# ## Functions for text extraction

def extract_information(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfFileReader(f)
        info = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()
    return {'info': info, 'n_pages': number_of_pages}


# ## Extraction of documents information

# List of files to extract information from.

list_files = sorted([file for file in os.listdir(docs_path) if file.lower().endswith('.pdf')])[:2]

# Unique file extensions:

# Check files that have not been checked, yet.

files_missing = sorted([file for file in set(list_files) - set(check_or_create_output_file(output_file))])

output_file = 'test_output.csv'
txt_folder = '/data/shared_data/data/raw/test/'

# Iterate over all missing files.

# +
f = open(output_file, 'a')

for i in tqdm(range(len(files_missing))):
    file = files_missing[i]
    try:
        is_extractable, n_pages = extract_text_from_pdf(file_path=f"{docs_path}/{file}",
                                                              txt_filename=file.lower().replace('.pdf', ''),
                                                              txt_folder=txt_folder, method=method)
    except PDFSyntaxError:
        logging.error(f"PDF {file} could not be processed", exc_info=True)
        continue
    except Exception as e:
        logging.error(f"PDF {file} could not be processed", exc_info=True)
        continue
    break 
    f.write(f"{file},{is_extractable},{n_pages}\n")
    
    if not i%100:
        print(f"Files checked: {i}")
        f.flush()

f.close()
# -
# ## Persist into DB

# +
#method = 'tika'
#file = f"/data/shared_data/data/output/{output_file.format(date='2019-08-06', method=method)}"
# load_documents_to_database(file_path=file, method=method, table='documents', schema='semantic')
# -

# ## Documents stats

output_file = '2019-07-31_documents_extractability.csv'

docs_df = pd.read_csv(output_file, delimiter=',')
print(f"Number of documents processed {docs_df.shape[0]}")

# Aggregate by whether documents are extractable or not:

# +
agg_df = docs_df[['filename', 'is_extractable']].groupby('is_extractable').agg(['count']).\
rename(index=str, columns={'count': 'n'})

agg_df['perc'] = agg_df['filename']['n'] / sum(agg_df['filename']['n']) * 100
agg_df
# -

# Number of documents with errors (= cannot be opened and processed):

docs_df.query("not is_extractable and number_of_pages == 0").shape[0]

docs_df.describe()

# Plotting the distribution of documents by number of pages.

# +
series_plot = docs_df['number_of_pages']

nrows = 1 # Number of rows
ncols = 1 # Number of columns
fontsize = 14 # General fontsize

# Create a subplot grid
# ax is a list with the dimensions [nrows][ncols]
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(10, 6)) # (width, heigth)
ax.hist(series_plot, bins=50, alpha=0.5)

ax.axvline(x=series_plot.mean(), ymin=0, color='orange')
ax.axvline(x=series_plot.median(), ymin=0, color='red')

ax.set_xlabel( 'Number of pages', 
                fontsize=fontsize)
ax.set_ylabel('Frequency', 
                fontsize=fontsize)

ax.text(0.60, 0.9, # [x, y] 
          f"Mean number of pages: {round(series_plot.mean(), 2)}",
          transform=ax.transAxes, fontsize=fontsize)
ax.text(0.60, 0.8, # [x, y] 
          f"Median number of pages: {round(series_plot.median(), 2)}",
          transform=ax.transAxes, fontsize=fontsize)

ax.tick_params(axis='both', labelsize=12)

ax.set_title("Distribution of documents by number of pages")

fig.tight_layout()
# -

# ### Crossing documents with complaints

docs_list = [f for f in docs_df.filename]
docs_list = "', '".join(docs_list)

query = f"select id_llamado, nombre_archivo as filename, complaints._tipo_resultado as result, complaints.fecha_resolucion_proceso as resolution_date" \
    f" from raw.pbc_adenda" \
    f" left join semantic.complaints as complaints" \
    f" using (id_llamado) " \
    f" where nombre_archivo in ('{docs_list}')"

complaints = pd.read_sql_query(query, con)

complaints.shape[0]

df = docs_df.merge(complaints, on='filename')

df.shape[0]

df.loc[df['result'].isnull(), 'result'] = 'No complaint'

df = df[df['is_extractable']]
df.head()

# Plotting the distribution of documents by number of documents, by whether they had an effective complaint or not.

# +
nrows = 1
ncols = 1
fontsize = 14

fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 7))


data = df.query('result in ("No complaint", "en_contra")')['number_of_pages']
axis.hist(data, 
          density=True,
          alpha=0.5, label=f'No complaint or uneffective. Mean {round(data.mean(), 2)}. Median {round(data.median(), 2)}. Count {data.shape[0]}'
         )
axis.axvline(data.mean())

data = df.query('result == "a_favor"')['number_of_pages']
axis.hist(data, 
          density=True, 
          alpha=0.5, label=f'Effective complaint. Mean {round(data.mean(), 2)}. Median {round(data.median(), 2)}. Count {data.shape[0]}'
         )
axis.axvline(data.mean(), color='orange')


axis.set_ylabel('Normalized number of documents', fontsize=fontsize)
axis.set_xlabel('Number of pages', fontsize=fontsize)

axis.set_title('Distribution of documents by number of pages, separated by complained or not', fontsize=fontsize)

axis.legend(loc='best', fontsize=fontsize)
# -


