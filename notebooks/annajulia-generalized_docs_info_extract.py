# # Extract information from PDF and zip files

# ## Imports

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0, '../src/utils')
import utils 

# Packages
import csv
import datetime as dt
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import re
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
zip_folder = '/data/original_data/unzip_first_pbc'
output_file = f"{dt.date.today()}_documents_extractability.csv"
header = "filename,is_extractable,number_of_pages\n"

# ### Start connection with db

con = utils.connect_to_database()


# ## Functions for text extraction

def list_files_size(path, file_ext=None, descending=True):
    for dirpath, dirnames, files in os.walk(path):
        if dirpath == path:
            continue
        file_names = [f for f in files]
        print("----")
        print(dirpath)
        print(file_names)
        
        if file_ext is not None:
            file_names = [f for f in files if any(f.lower().endswith(x) for x in file_ext)]
    print(file_names)
"""
    file_sizes = []
    for file in files:
        file_sizes.append({'file': file, 'size': os.path.getsize(os.path.join(path, file))})

    file_sizes.sort(reverse=descending, key=lambda s: s['size'])
    return file_sizes"""

list_files_size(path='/data/original_data/unzip_first_pbc/')#, file_ext=['.pdf', '.docx'])


def get_pdf_max_size(path):
    pdf_files = list_pdf_files_size(path)
    
    if len(pdf_files) > 0:
        return pdf_files[0]['file']
    else:
        logging.error(f"No PDF found in {path}")
        return ""


def find_pdf_folder_and_file(zip_file, zip_folder):
    docs_folder = f"{zip_folder}/{zip_file.replace('.zip', '')}"
    return docs_folder, get_pdf_max_size(docs_folder)


# ## Extraction of documents information

# List of files to extract information from.

list_files = sorted([file for file in os.listdir(docs_path)])

# Unique file extensions:

set([f[-4:].lower() for f in list_files])

# Check files that have not been checked, yet.

files_missing = sorted([file for file in set(list_files) - set(check_or_create_output_file(output_file))])

# Iterate over all missing files.

# +
f = open(output_file, 'a')

for i, file in enumerate(missing_files):
    is_extractable, n_pages = False, 0  # Default
    
    if file.lower().endswith('.zip'):
        pdf_folder, pdf_file = find_pdf_folder_and_file(zip_file=file, zip_folder=zip_folder)
    else:
        pdf_folder, pdf_file = docs_path, file

    try:
        is_extractable, n_pages = extract_pdf_characteristics(f"{pdf_folder}/{pdf_file}")
    except PDFSyntaxError:
        logging.error(f"PDF {file} could not be processed", exc_info=True)
    except Exception as e:
        logging.error(f"PDF {file} could not be processed", exc_info=True)
    
    f.write(f"{file},{is_extractable},{n_pages}\n")
    
    if not i%100:
        print(f"Files checked: {i}")
        f.flush()

f.close()
# -

# ## Zip files with complaints

list_files = sorted([file for file in os.listdir(docs_path) if file.lower().endswith('.zip')])
list_files = sorted([file for file in os.listdir(docs_path)])

len(list_files)

docs_list = "', '".join(list_files)

query = f"select id_llamado, nombre_archivo as filename, complaints._tipo_resultado as result, complaints.fecha_resolucion_proceso as resolution_date" \
    f" from raw.pbc_adenda" \
    f" left join semantic.complaints as complaints" \
    f" using (id_llamado) " \
    f" where nombre_archivo in ('{docs_list}')"

complaints = pd.read_sql_query(query, con)

complaints.head()

complaints['file_ext'] = complaints['filename'].apply(lambda x: x[-4:].lower())

complaints.groupby('file_ext').size()

complaints[complaints['result'] == 'a_favor'].groupby('file_ext').size()

zip_with_complaints = complaints[complaints['result'].notnull()]
zip_with_complaints.head()

print(zip_with_complaints.shape[0])
zip_with_complaints.result.unique()

eff_complaints = complaints[complaints['result'] == 'a_favor']

eff_complaints.shape[0]
