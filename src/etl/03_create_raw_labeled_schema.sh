#! /bin/bash

# Connecting to Postgres
psql service=postgres << EOF

-- Run SQL script that builds tables for raw_labeled schema
\c dncp
\ir ../../sql/dataprep/create_schema_raw_labeled.sql

EOF
