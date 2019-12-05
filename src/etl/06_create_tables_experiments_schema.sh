#! /bin/bash

# Connecting to Postgres
psql service=postgres << EOF

-- Run SQL script that builds tables for semantic schema
\c dncp

-- Experiments table
\i ../../sql/experiments/create_table_experiments.sql

-- Evaluations table
\i ../../sql/experiments/create_table_evaluations.sql

-- Approaches table
\i ../../sql/experiments/create_table_approaches.sql

-- Learners table
\i ../../sql/experiments/create_table_learners.sql

-- Feature Importances table
\i ../../sql/experiments/create_table_feature_importances.sql

p-- Errors table
\i ../../sql/experiments/create_table_errors.sql

EOF
s
