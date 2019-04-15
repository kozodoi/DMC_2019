########### FUNCTION FOR PROFIT MEASURE

# self-defined metric
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# costs according to the Prudsys task
# feval (callable or None, optional (default=None)) – Customized evaluation function. Should accept two parameters: preds, train_data, and return (eval_name, eval_result, is_higher_better) or list of such tuples. https://lightgbm.readthedocs.io/en/latest/Python-API.html

from sklearn.metrics import confusion_matrix
def prediction_reward(y_true, y_preds):
        preds_labels = y_preds > 0.5
        tn, fp, fn, tp = confusion_matrix(y_true = y_true, y_pred = preds_labels).ravel()
        costs = tn*0.0 + fp*(-25.0) + fn*(-5.0) + tp*(5.0)
        return 'profit', costs, True
    
    
    
########### FUNCTION FOR PROFIT MEASURE WITH CUTOFF PARAMETER

# can use after modeling for threshold optimization
# computation same as prediction_reward()

from sklearn.metrics import confusion_matrix
def recompute_reward(y_true, y_preds, cutoff = 0.5):
        preds_labels = y_preds > cutoff
        tn, fp, fn, tp = confusion_matrix(y_true = y_true, y_pred = preds_labels).ravel()
        costs = tn*0.0 + fp*(-25.0) + fn*(-5.0) + tp*(5.0)
        return costs
    
    
    
########### FUNCTION FOR WEIGHTED LOG-LOSS OBJECTIVE

# false positive is 5 times worse than false negative
# logsloss for LGB that penalizes false positives more severly
    
def weighted_logloss_train(y_true, y_preds):
    beta = 5
    p = 1. / (1. + np.exp(-y_preds))
    y = y_true
    grad = p * (beta + y - beta*y) - y
    hess = p * (1 - p) * (beta + y - beta*y)
    return grad, hess

def weighted_logloss_eval(y_true, y_preds): 
    beta = 5
    p = 1. / (1. + np.exp(-y_preds))
    y = y_true
    val = -y * np.log(p) - beta * (1 - y) * np.log(1 - p)
    return 'weighted logloss', np.sum(val), False
    

    
########### FUNCTION FOR COUNTING MISSINGS

# computes missings per variable (count, %)
# displays variables with most missings

import pandas as pd
def count_missings(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending = False)
    table = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
    table = table[table["Total"] > 0]
    return table