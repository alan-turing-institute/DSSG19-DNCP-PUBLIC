#! /bin/bash

# Connecting to Postgres
psql service=postgres << EOF

-- Run SQL script that builds tables for semantic schema
\c dncp

-- Tenders table
\i ../../sql/semantic/create_tenders.sql

-- Labels tables
\i ../../sql/semantic/create_labels.sql

-- Agencies
\i ../../sql/semantic/create_agencies.sql

-- Complaints
\i ../../sql/semantic/create_complaints.sql
EOF
