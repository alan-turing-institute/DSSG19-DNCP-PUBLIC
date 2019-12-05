# # Unzip files to different folder

# ## Imports

# +
import sys
sys.path.insert(0, '../src/documents')
from manage_documents import unzip_to_folder

import logging
import os
import zipfile
# -

# ## Settings

docs_path = '/data/original_data/first_pbc/'
zip_folder = '/data/original_data/unzip_first_pbc'

list_files = sorted([file for file in os.listdir(docs_path) if file.lower().endswith('.zip')])

# ## Unzip process

for i, file in enumerate(list_files):
    if not i%100:
        print(f"File {i} out of {len(list_files)}")
    file_path = f"{docs_path}/{file}"
    
    if file.lower().endswith('zip'):
        unzip_to_folder(file_path=file_path, extract_folder=zip_folder)  
    else:
        print(f"{file} is not a zip file")


