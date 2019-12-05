# # Logistic regression
#

# Imports
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def fit(args, train_dict):
    """
    Given train data, fit a logistic regression. The default options are:
    criterion=’gini’,
    splitter=’best’,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    class_weight=None,
    presort=False
    See: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    Parameters
    ----------
    args: dictionary
        Approach configuration coming from the experiment yaml

    train_dict: dictionary
        Dictionary containing the train features DataFrame and the train labels DataFrame.


    Return
    ------
    Class
        Model fitted using the train data.

    """
    dt_args = args['hyperparameters']
    dt = DecisionTreeClassifier(**dt_args)

    model = dt.fit(X = train_dict['features'], y = train_dict['labels'])

    return model

def predict(model, test_features):
    """
    predict test set.

    Parameters
    ----------
    model: Class
        Fitted model: logistic regression.
    test_features: DataFrame
        Pandas DataFrame containing features to be used for predicting.

    Return
    ------
    List
        DataFrame of predictions (0s or 1s).
    """
    prediction_index = list(test_features.index.values)
    prediction_values = model.predict(test_features)
    predictions = pd.DataFrame(data = {'prediction': prediction_values, 'id_llamado' : prediction_index})
    predictions = predictions.set_index('id_llamado')

    return predictions
