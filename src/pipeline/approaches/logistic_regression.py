# # Logistic regression
#

# Imports
from sklearn.linear_model import LogisticRegression
import pandas as pd


def fit(args, train_dict):
    """
    Given train data, fit a logistic regression
      
    Parameters
    ----------
    approach: dictionary
        Approach configuration coming from the experiment yaml
    train_dict: dictionary
        Dictionary containing the train features DataFrame and the train labels DataFrame.

    Return
    ------
    Class
        Model fitted using the train data.
    
    """
    logreg = LogisticRegression(random_state = 42, 
                                C = args['hyperparameters']['C'], 
                                penalty = args['hyperparameters']['penalty'],
                                class_weight= args['hyperparameters']['class_weight'])
    
    model = logreg.fit(X = train_dict['features'], y = train_dict['labels'])
    
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
