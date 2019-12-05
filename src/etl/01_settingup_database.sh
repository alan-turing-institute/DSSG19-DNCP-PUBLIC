#! /bin/bash

# Set original data path
cd /data/original_data

# Input name of sql dump file
echo 'Hi! Insert name of original dump file that contains database(without .sql):'
read sqldumpfile

echo 'Insert name of original dump file that contains tender_record data(without .sql):'
read tenderrecordfile

echo 'Insert name of original dump file that contains bidders data(without .sql):'
read bidderstable

echo 'Insert name of original dump file that contains first_version_pbc_adenda data(without .sql):'
read pbcadenda

echo 'Insert name of original dump file that contains first_version_pbc_adenda data(without .sql):'
read denunciastable

echo 'Insert the name you want to give to the database: '
read databasename

# Create name for copy file
#DB
copy='_copy'
sql='.sql'
sqldumpfile_copy=$sqldumpfile$copy
#Tender_record
tenderrecordfile_copy=$tenderrecordfile$copy
#Bidders
bidderstable_copy=$bidderstable$copy
#Pbc
pbcadenda_copy=$pbcadenda$copy
# Denuncias
denunciastable_copy = $denunciastable$copy

# Create copy of dump file just in case
echo Copying files...
cd /data/original_data
#DB
cp $sqldumpfile$sql $sqldumpfile_copy$sql
#Tender record
cp $tenderrecordfile$sql $tenderrecordfile_copy$sql
#Bidders table
cp $bidderstable$sql $bidderstable_copy$sql
#Pbc_adenda
cp $pbcadenda$sql $pbcadenda_copy$sql
#denuncias
cp $denunciastable$sql $denunciastable_copy$sql

# Change owner name to dbadmin to meet our requirements
echo Changing database owner ...
mod='_owner'
sed 's/OWNER TO postgres;/OWNER TO dbadmin;/g' $sqldumpfile_copy$sql > $sqldumpfile_copy$mod$sql
sed 's/OWNER TO postgres;/OWNER TO dbadmin;/g' $tenderrecordfile_copy$sql > $tenderrecordfile_copy$mod$sql
sed 's/OWNER TO postgres;/OWNER TO dbadmin;/g' $bidderstable_copy$sql > $bidderstable_copy$mod$sql
sed 's/OWNER TO sistema;/OWNER TO dbadmin;/g' $pbcadenda_copy$sql > $pbcadenda_copy$mod$sql
sed 's/sistema;/dbadmin;/g' $pbcadenda_copy$mod$sql > $pbcadenda_copy$mod$sql

# Load SQL dump
echo Getting into Postgres ...
psql service=postgres << EOF

-- Create Empty DATABASE
CREATE DATABASE $databasename OWNER dbadmin;

-- Create database and data from dump file
\c $databasename
\i /data/original_data/$sqldumpfile_copy$mod$sql

-- Add tender_record data
\i /data/original_data/$tenderrecordfile_copy$mod$sql

-- Add bidders data
\i /data/original_data/$bidderstable_copy$mod$sql

-- Add bidders data
\i /data/original_data/$pbcadenda_copy$mod$sql

-- Add denuncias data
\i /data/original/$denunciastable_copy$sql

-- Change schema name to raw, create cleaned and semantic schemas
ALTER SCHEMA dsfsg RENAME TO raw;
CREATE SCHEMA semantic;
CREATE SCHEMA experiments;
CREATE SCHEMA features;
CREATE SCHEMA cohorts;
CREATE SCHEMA labels;
CREATE SCHEMA results;

EOF

cd
cd DSSG19-DNCP/src/etl/
pwd
echo 'Setting up database process finished!'
