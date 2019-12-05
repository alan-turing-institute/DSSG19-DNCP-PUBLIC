CREATE TABLE IF NOT EXISTS experiments.feature_importances (
  experiment_id bigint NOT NULL,
  approach_id bigint NOT NULL,
  learner_id bigint NOT NULL,
  model_name varchar(255) NOT NULL,
  hyperparameters_used json NOT NULL,
  fold varchar(255) NOT NULL,
  feature text NOT NULL,
  importance float NOT NULL,
  created_on TIMESTAMP DEFAULT now(),
  PRIMARY KEY (experiment_id,approach_id,learner_id,feature,fold));

