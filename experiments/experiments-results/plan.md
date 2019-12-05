# Experiments Plan and Overall Results

# Best Classifiers

|   learner_id |   experiment_id | selector                                                                   |   mean_f1 |   mean_precision |   mean_recall |
|-------------:|----------------:|:---------------------------------------------------------------------------|----------:|-----------------:|--------------:|
|   2643795303 |      1129688681 | mean_recall_threshold_0.2->min_recall_threshold_0.05->mean_precision_top_5 | 0.0479592 |        0.0273453 |      0.325111 |
|   3063080495 |      1129688681 | mean_recall_threshold_0.2->min_recall_threshold_0.05->mean_precision_top_5 | 0.0383318 |        0.0210071 |      0.310476 |
|   1006657992 |      1129688681 | mean_recall_threshold_0.2->std_recall_threshold_1->mean_precision_top_5    | 0.0448942 |        0.0281337 |      0.226553 |
|   5959360530 |      1129688681 | mean_recall_threshold_0.2->std_recall_threshold_1->mean_precision_top_5    | 0.0413761 |        0.0236126 |      0.29711  |
|   2643795303 |      1129688681 | mean_f1_threshold_0.01->min_recall_threshold_0.05->mean_precision_top_5    | 0.0479592 |        0.0273453 |      0.325111 |
|   3063080495 |      1129688681 | mean_f1_threshold_0.01->min_recall_threshold_0.05->mean_precision_top_5    | 0.0383318 |        0.0210071 |      0.310476 |


## 08-12 

### Summary: 
See if TFIDF features improves the baseline model run on 06-08

### Null Hypothesis ($H_0$):
TFIDF does not improve the performance of the baseline model

### Methodology: 

#### Cohort

Tenders of the important procurement types filtered for processes with tender documents extracted

#### Features
Three sets of features will be selected to the experiments:

[all current features] + [TFIDF @5000 features]

1. all current features + TFIDF features [8857201145]
2. all current features [8289102229]
3. TFIDF features [6452858095]

#### Classifiers

[all classifiers] + Coarse hyperparameters

### Evaluation
Current model selectors 

### Actions
Select features that are important

------------

### Results


## 06-08 

### Summary: 
Generate baseline with current features

### Null Hypothesis ($H_0$):
No hypothesis

### Methodology: 

#### Cohort

Tenders of the important procurement types

#### Features

[all current features] 

#### Classifiers

1. Very grained hyperparameters [8269072727]
2. Coarse hyperparameters [1129688681]

### Evaluation
Current model selectors 

### Actions
Select features that are importante

------------

### Results

1. Experiment 8269072727  with very grained hyperparameters took too much to run. 
So, it was stoped.
2. The best learners of experiment 1129688681 are listed below. Given the selectors, 
the classifiers had `recall ~ 30%`, `precision ~ 2%` and `f1 ~ 4%`.
3. Some uselss features show that `tipo_procedimiento` column is not very well
normalised. One easy turnaround is to use `codigo_procedimiento`.

|   learner_id |   experiment_id | selector                                                                   |   mean_f1 |   mean_precision |   mean_recall |
|-------------:|----------------:|:---------------------------------------------------------------------------|----------:|-----------------:|--------------:|
|   2643795303 |      1129688681 | mean_recall_threshold_0.2->min_recall_threshold_0.05->mean_precision_top_5 | 0.0479592 |        0.0273453 |      0.325111 |
|   3063080495 |      1129688681 | mean_recall_threshold_0.2->min_recall_threshold_0.05->mean_precision_top_5 | 0.0383318 |        0.0210071 |      0.310476 |
|   1006657992 |      1129688681 | mean_recall_threshold_0.2->std_recall_threshold_1->mean_precision_top_5    | 0.0448942 |        0.0281337 |      0.226553 |
|   2643795303 |      1129688681 | mean_recall_threshold_0.2->std_recall_threshold_1->mean_precision_top_5    | 0.0479592 |        0.0273453 |      0.325111 |
|   3063080495 |      1129688681 | mean_recall_threshold_0.2->std_recall_threshold_1->mean_precision_top_5    | 0.0383318 |        0.0210071 |      0.310476 |
|   5959360530 |      1129688681 | mean_recall_threshold_0.2->std_recall_threshold_1->mean_precision_top_5    | 0.0413761 |        0.0236126 |      0.29711  |
|   8518578992 |      1129688681 | mean_recall_threshold_0.2->std_recall_threshold_1->mean_precision_top_5    | 0.0395237 |        0.0228857 |      0.254652 |
|   2643795303 |      1129688681 | mean_f1_threshold_0.01->min_recall_threshold_0.05->mean_precision_top_5    | 0.0479592 |        0.0273453 |      0.325111 |
|   3063080495 |      1129688681 | mean_f1_threshold_0.01->min_recall_threshold_0.05->mean_precision_top_5    | 0.0383318 |        0.0210071 |      0.310476 |


## 05-08 


### Summary:
Test whether documents metadata improve results

### Null Hypothesis ($H_0$):
[all current features] + [document metadata] = [all current features]

### Methodology:

#### Cohort

It is important to notice that this test will happen in a subset of the total data. About 30k tenders have easily accessible documents, the rest of the documents are zipped.

#### Features
Three sets of features will be selected to the experiments:

1. [all current features]
2. [all current features] + [document metadata]
3. [document metadata]

Where [document metadata] features have `number of pages` and `text extractable pdf` values.

#### Classifiers

All available classifers with medium hyperparameter grid

### Evaluation
Higher average recall (?) of the selected models.

### Actions
- If $H_0$ is the case, then there is no reason to unzip and process the rest of the documents
- If $H_0$ is **not** the case, then we should consider unziping the remaining files given the time constrains.

------------

### Results

> H_0 is true

|Features sets | Experiment id | Avg Recall |
|--------------| --------------:| ---------: |
[all current features] | 5084808129 | |
[all current features] + [document metadata] |9361830894| |
[document metadata]|2625569271 |
