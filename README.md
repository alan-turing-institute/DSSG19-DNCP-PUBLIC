# Reducing corruption through automatic risk detection in public procurement in Paraguay

This project has been developed in partnership with the National Agency for Public Procurement in Paraguay (DNCP) during the 2019 Data Science for Social Good Fellowship, hosted jointly by The Alan Turing Institute and the University of Warwick.

<p align="center">
    <img src=/images/dncp_logo.png width="180">
</p>

---

## Table of contents

1. [Technical Set-up](#tech-setup)
    * [Prerequisites](#prerequisites)
    * [Initializing the pipeline](#pipeline-init)
    * [Running an experiment](#run-experiment)
2. [The Policy Problem: background and social impact](#policy-problem)
3. [The Machine Learning Problem](#ml-problem)
    * [Objective](#objective)
    * [ML pipeline](#ml-pipeline)
    * [Temporal Cross-Validation](#temporal-cv)
    * [Feature Generation](#features)
4. [Experiments Configuration](#experiment-config)
5. [Results](#results)
6. [Impact](#impact)
7. [Next Steps](#next-steps)
8. [Contributors](#contributors)

---

<a name="tech-setup"></a>
## Technical Set-up

<a name="prerequisites"></a>
### Prerequisites
To use the code of this project, you first need:
- Python 3.5+
- A PostgreSQL database with your source data (procurement processes, complaints, etc.) loaded.
- Ample space on an available disk to store predicted results as well as models.

<a name="pipeline-init"></a>
### Initializing the pipeline

Follow the steps below to initialize your Machine Learning pipeline:

1. Clone the project repository with latest changes in the _master_ branch.

2. Install Python requirements following one of these two options:
    * Using pip:
    ```
    pip install -r requirements.txt
    ```
    * Using conda:
    ```
    conda env create -f conda_env.yml
    ```

3. Create an enviroment configuration file `env.yaml` with, at least, the following entry:
    - `database=0.0.0.0:5432:<database_name>:<user>:<pwd>`: Database connection configuration, replacing with the corresponding values the database name, the user and the password.
    - `persistance_path='<some_path>'`: Directory where all objects are persisted, written as a plain string.
    - `production_path='<some_other_path>'`: Directory where all the data from the best model selected is stored. The path needs to be written as a plain string.

<a name="run-experiment"></a>
### Running an experiment

#### Experiment definition
Experiments are the core of this project in order to:

- Predict on a subset of the data
- Try new **approaches**, which are a combination of algorithms and hyperparameters
- Choose **features** included
- Define model **validation**
- Define model **evaluation**

And some other configuration items.

You can get some guidance in how to define an experiment configuration file (e.g. _yyyymmdd-fake-experiment.yaml_) from the experiments found in the `experiments` folder.

#### Experiment launching
We use the package [Fire](https://github.com/google/python-fire) to easily run experiments using the `run_experiment` function.

To launch a new experiment:

1. Place yourself in the root of the repository.

2. Run the following command, specifying the YAML experiment you want to test:

```
python src/pipeline/pipeline.py run_experiment --experiment_file='yyyymmdd-fake-experiment.yaml'
```

#### Parallel experiment launching
If your experiment is too big and it is taking too long to run, you may want to run it in parallel. How parallelization works here is that the big experiment YAML is split by _approach_

To launch a new experiment in parallel:

1. Place yourself in the root of the repository.

2. Run the following command in bash, specifying the YAML experiment you want to test:

```
source run_parallelization 'yyyymmdd-fake-experiment.yaml'
```

This will create some temporary files and folder:
* Separate bash commands: `list_of_commands.sh`
* Experiment splits: `experiments/parallelization/yyyymmdd-fake-experiment`

They will be automatically removed when all the processes finish.

Once the parallelization starts, your terminal will show a similar message:

```
tmux -S <session_name> attach
```

To check the progress of the parallel launching:
* Run the command above in a new terminal
* Once done, type `Prefix + w` to check the progress of the different parallelized processes (`Prefix` is `Ctrl + b`, or `Cmd + b` if using Mac)

**Note**: A guide to further commands in tmux can be found [here](https://www.linode.com/docs/networking/ssh/persistent-terminal-sessions-with-tmux/).

<a name="policy-problem"></a>
## The Policy Problem: background and social impact
Each year, the National Agency for Public Procurement in Paraguay (Dirección Nacional de Contraciones Públicas, DNCP) receives an average of 13,000 procurement processes that have to be reviewed before being published.

The reviewing process is performed by 30 officials inside DNCP and relies heavily on the experience and heuristic judgement of the own reviewer to flag problems. Besides, they only have 3 days to review each tender before publishing it. The allocation of these officials to review documents is a challenge, since there is no clear procedure on how to prioritize them to better catch potential errors or irregularities.

If left unregulated or failed to perform high-quality reviews, public procurement processes are prone to corruption and represents a huge loss in taxpayer’s money.

In the case of DNCP, corruption is not labelled itself. However, we can use effective complaints as a proxy. _Complaints_ are a legal process by which a potential bidder points out irregularities in a procurement process, which may be limiting the participation and transparency of it. By _effective_, we mean that there has been a judge behind certifying the arguments of the complaint.

<a name="ml-problem"></a>
## The Machine Learning Problem

<a name="objective"></a>
### Objective
The goal of this project is to build a system that helps DNCP prioritise effort and optimise allocation of labour in order to reduce likelihood of corruption in the procurement process.

In order to achieve our goal, we train a model that predicts a probability of a given tender in resulting in an effective complaint. By getting a risk score between 0 and 1, we are able to prioritise procurements, so that the most risky ones get more attention when reviewing. Besides, our model also provides a list with the top risk factors influencing each individual process.

<a name="ml-pipeline"></a>
### ML pipeline
We built an end-to-end machine learning pipeline, generalised enough to be easily adapted to different projects on public procurement.

The following diagram shows an outline of the pipeline generated.

<p align="center">
    <img src=/images/ml_pipeline.png width="500">
</p>


<a name="temporal-cv"></a>
### Temporal Cross-Validation
We split our dataset on a temporal basis as we found that the number of effective complaints seems to follow a time trend - growing in numbers with each year.

<p align="center">
    <img src=/images/temporal_folding.png width="400">
</p>

The training set increases over time and is cumulative. This mirrors live deployment as the model can learn with a bigger train set as new data is added each year. We also implemented a blind gap between training set and test set also to simulate live deployment where the model would not have learnt about the latest outcomes of most recent procurements.

The function `generate_temporal_folds` performs the temporal folding described above given some parameters specified in the experiment setup file. These parameters that can be tuned are:

| Parameter name | Description | Example |
|:------------- |:------------- | :-----: |
| as_of_date | First reference date where to start building the first fold. | '2013-01-01' |
| train_lag | Length of the training set in number of days. If ‘all’ specified, the training set will be a growing window from the most recent data for each fold. | 365 |
| test_lag | Length of the test set in number of days. | 365 |
| aod_lag | Span in number of days between one reference date (as of date) and the following one. | 182 |
| blind_gap | Gap in days between training set and test set to simulate live deployment. | 90 |
| number_of_folds | Number of folds wanted. If null specified, the function will generate all possible complete folds between the as_of_date and the test_date_limit. | 10 |
| test_date_limit | Threshold date to generate folds. The function will only generate complete folds. | '2019-01-31' |

<a name="features"></a>
### Feature generation
Our feature engineering process included two sources of information:

 #### Historical data from metadata
 At the very first stage of a process, the **information available** is very limited:
 - Agency
 - Planned amount
 - Type of agency
 - Type of procurement
 - Type/category of product
 - Type of service

From these variables above regarding what we know about the tender, we create **historical features** for each of the items. The complete list of feature groups is listed below:

- Boolean whether the agency is a municipality or not
- Number of tenders received up to a certain date
- Planned values of tenders
- Number of complaints received (effective and uneffective)
- Number of effective complaints received
- Whether the agency ever received an effective complaint
- Number of cancelled complaints
- Number of tenders with complaints
- Number of tenders with effective complaints
- Number of complaints in the last 6 months
- Number of effective complaints in the last 6 months
- Number of bidders
- Distribution of number of products for the range of tenders
- Whether the agency submitted a tender of the top 10 high risk product category in the past [2 years]

#### Text features from documents
Documents in a procurement process are a rich source of information, since they contain details such as specifications, legal implications, and more, that could limit competition.

The main text feature that we are creating for the text documents uses an algorithm called TF-IDF (term frequency-inverse document frequency). It is an ideal algorithm to help pick out crucial words that are highly utilised in the documents that do not appear in other documents.

<a name="experiment-config"></a>
## Experiments Configuration

#### Generating a new
Refer to the [Technical Setup-up](#tech-setup) for generating and launching a new experiment. Check the `experiments` folder for examples. The most recent ones contain the latest structure needed for the pipeline to be run.

For experiment evaluation and selection, we developed a notebook that is well detailed and can be easily run (`Example Experiment Selection.py`). It can be found under the `experiments/experiments-results` folder.


<a name="results"></a>
## Results
We selected the best model out of all the experiments we tested balancing two important criteria: **Performance** and **Bias**.

#### Model performance
The first criteria was to provide good quality performance of our models, measured with recall. For each day of any given year:
- We select the tenders received on that date
- Order them by risk score
- Label the top k% of tenders as risky

Once performed for each day, we append all the data and calculate how many of all effective complaints these quality reviews capture.

#### Equity
The second criteria was to ensure that our model takes into account the bias in our data and fights against perpetuating it. More specifically we used the equity parity metric that ensures all protected groups (e.g. region or type of agency) have equal representation in the selected set.


#### Model selection
In the model selection process, we compared all our models against two other approaches:
- **Random baseline**: This baseline emulates the current first-come-first-served practise in DNCP, where we label k% of daily incoming procurements.
- **Heuristic baseline (order by value)**: This approach uses tender value as a proxy for risk score. It is an easy-to-implement alternative to current DNCP practise.

Figure 1 below shows the performance of our selected best model at each k% of selected tenders to receive quality reviews, compared to the two baselines.

**Figure 1: Comparison of recall curves optimised at 30% quality review**
<p align="center">
    <img src=/images/recall_comparison.png width="500">
</p>

For our final model selection, we understood that around 30% of reviewers are considered very experienced, thereby able to perform high-quality reviews. At 30% of these quality reviews, both the heuristics baseline and our model respond with an **80% of recall**.

Recall here means proportion of all actual effective complaints are caught by the quality  for closer review by experienced reviewers.  

However, we also want to include our second perspective: equity. In Figures 2 and 3 below we analyse the distribution of procurements flagged by the different approaches into different categories of some selected bias variables. Here we only show two of them: value and type of agency, but we could also be analysing other variables such as the region or the type of product.

**Figure 2: Distribution of procurements to review by value**
<p align="center">
    <img src=/images/tender_value.png width="500">
</p>

Given the inherent skewness in effective complaints mainly in medium and high value tenders, low value tenders are hence left out in the review if we simply order tenders by value for review. Our model gives about 30% more weight to the low value tenders compared to the heuristic baseline. It does this without suffering much dilution in total recall.

**Figure 3: Distribution of procurements to review by type of agency**
<p align="center">
    <img src=/images/agency_type.png width="500">
</p>

We could optimise the model for any bias variable that we consider important. We experimented with controlling for equity among agency types. Given that the results are more or less equitable, the model selection did not make a huge difference.


#### Summary
Our selected best model is using the Extra Trees algorithm. We see that the machine learning model that we chose that maximises both recall and equity performs as well the baseline heuristic with a bias for high value tenders.

<a name="impact"></a>
## Impact of the project
We built a machine learning model to help DNCP prioritise which tenders should be investigated based on their risk of finding irregularities in the procurement process.

By doing so, we expect the project to have positive impact in different areas.

1. **Fraud discovery**: using lessons learnt from high-value corruption cases for smaller procurements.
2. **Proactive regulation**: anticipating corrupt acts before they become a tragedy.
3. **Increased accountability**: ensuring the quality of goods and services provided to citizens.

<a name="next-steps"></a>
## Next Steps
We obtained really promising results in this project. To further improve the predictions, we would recommend three main steps.

#### Improve text features
Our text features are generated using the TF-IDF algorithm. Given the length of the texts, it would be ideal to test the `max_features` parameters of the algorithm above 10,000. Besides, in processing the text features, we performed a basic cleaning of the symbols, punctuation and stop words. More cleaning can be done such as stemming or lemmatization to improve meaning in the word features.

#### Include more metadata features
There are many red flag projects around the world implementing a risk scoring model for procurements. This includes the [EU ARACHNE project](https://ec.europa.eu/social/main.jsp?catId=325&intPageId=3587&langId=en#navItem-4), the [EU Red Flag](https://www.redflags.eu/files/redflags-summary-en.pdf), or [World Bank indicators](https://openknowledge.worldbank.org/bitstream/handle/10986/3731/WPS5243.pdf?sequence=1). These are rich resources to brainstorm new features to be added into the current feature engineering set. A thorough study of the red flags could also help DNCP come up with ideas about existing data gaps in the system.

#### Explore network features
Using network theory to deeper analyse the relationships between contracting agencies and the bidders and the outcome of the bid could help identify clusters of agency-bidder relationship that would be otherwise difficult to quantify.


<a name="contributors"></a>
## Contributors
- **Data Science for Social Good Fellows (Summer 2019)**:
    - [Maria Ines Aran](https://github.com/nakanotokyo)
    - [João Carabetta](https://github.com/JoaoCarabetta)
    - [Wen Jian](https://github.com/wjivan)
    - [Anna Julià-Verdaguer](https://github.com/ajuliaverdaguer)
- **Project Manager**: Josh Sidgwick
- **Technical Mentor**: Pablo Rosado
