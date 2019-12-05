CREATE TABLE IF NOT EXISTS experiments.evaluations (
  experiment_id bigint NOT NULL,
  approach_id bigint NOT NULL,
  learner_id bigint NOT NULL,
  hyperparameters_used json NOT NULL,
  fold varchar(255) NOT NULL,
  eval_metric varchar(30) NOT NULL,
  score float NOT NULL,
  created_on TIMESTAMP DEFAULT now(),
  PRIMARY KEY (experiment_id,approach_id,learner_id,eval_metric, fold));
