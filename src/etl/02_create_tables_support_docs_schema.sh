#! /bin/bash

# Connecting to Postgres
psql service=postgres << EOF

-- Run SQL script that builds tables for support docs
\c dncp
\ir ../../sql/dataprep/wen-create_exchange_inflation.sql

EOF
