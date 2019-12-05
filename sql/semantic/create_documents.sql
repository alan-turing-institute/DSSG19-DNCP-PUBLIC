CREATE TABLE IF NOT EXISTS semantic.documents(
  id_llamado bigint NOT NULL,
  filename varchar(255) NOT NULL,
  is_extractable boolean NOT NULL,
  number_of_pages integer NOT NULL,
  method varchar(20) NOT NULL,
  PRIMARY KEY (id_llamado, filename, method)
);
