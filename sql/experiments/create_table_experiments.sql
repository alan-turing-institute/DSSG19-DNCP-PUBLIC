-- Create table to store experiments
CREATE TABLE IF NOT EXISTS experiments.experiments (
  experiment_id bigint PRIMARY KEY,
  yaml_file_name varchar(50) NOT NULL ,
  yaml_content json NOT NULL,
  sql_features json NOT NULL,
  sql_cleaned json NOT NULL,
  created_on TIMESTAMP DEFAULT now()
);
