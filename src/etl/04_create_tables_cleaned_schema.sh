#! /bin/bash

# Connecting to Postgres
psql service=postgres << EOF

-- Run SQL script that builds tables for cleaned schema
\c dncp

-- Tenders table
\i ../../sql/cleaned/create-cleaned-proceso.sql

\i ../../sql/cleaned/create-cleaned-tender-records.sql

EOF
