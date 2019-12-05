#! /bin/bash

# Import password from env.yaml
source yaml_functions.sh
create_variables ../../env.yaml
password=${database##*:}

# Get current date to name the file
date=$(date +"%d_%m_%Y_%H%M")

# Create a dump file of the database and store it in /data/database_backups
pg_dump --dbname=postgresql://dbadmin:$password@localhost:5432/dncp > /data/database_backups/dncp_bckup_$date.sql

