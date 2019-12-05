CREATE TABLE IF NOT EXISTS experiments.learners(
  experiment_id bigint NOT NULL,
  approach_id bigint NOT NULL,
  learner_id bigint NOT NULL,
  hyperparameters_used json NOT NULL,
  pickle_path varchar(255),
  created_on TIMESTAMP DEFAULT now(),
  PRIMARY KEY (experiment_id,approach_id,learner_id)
);
