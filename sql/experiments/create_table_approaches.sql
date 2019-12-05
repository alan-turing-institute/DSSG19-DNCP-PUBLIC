CREATE TABLE IF NOT EXISTS experiments.approaches(
  experiment_id bigint NOT NULL,
  approach_id bigint NOT NULL,
  name varchar(255) NOT NULL,
  hyperparameters json NOT NULL,
  python_path varchar(255),
  python_content text,
  created_on TIMESTAMP DEFAULT now(),
  PRIMARY KEY (experiment_id,approach_id)
);

ALTER TABLE experiments.approaches
ADD COLUMN preprocessors json;