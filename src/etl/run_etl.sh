#! /bin/bash

# Create Database from sqldumpfile
echo Creating database and empty schemas...
source 01_settingup_database.sh

# Create tables in support_docs schema
echo Creating tables inn support_docs schema...
source 02_create_tables_support_docs_schema.sh

# Create tables in raw labeled schema
echo Creating tables inn raw_labeled schema...
source 03_create_raw_labeled_schema.sh

# Create tables in cleaned schema
echo Creating tables inn cleaned schema...
source 04_create_tables_cleaned_schema.sh

# Create tables in semantic schema
echo Creating tables inn semantic schema...
source 05_create_tables_semantic_schema.sh

# Create tables in experiments schema
echo Creating tables inn experiments schema...
source 06_create_tables_experiments_schema.sh

echo Process finished
