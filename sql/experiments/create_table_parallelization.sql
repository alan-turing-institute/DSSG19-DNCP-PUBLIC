-- Create table to store parallelized experiments
CREATE TABLE IF NOT EXISTS experiments.parallelization (
  macro_experiment_id bigint NOT NULL,
  experiment_id bigint NOT NULL,
  created_on TIMESTAMP DEFAULT now(),
  PRIMARY KEY (macro_experiment_id, experiment_id)
);
