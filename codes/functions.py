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


########### FUNCTION FOR TEST TIME AUGMENTATION

# creates multiple versionf of test data (with noise)
# averages predictions over the created samples

def predict_proba_with_tta(X_test, model, num_iteration, alpha = 0.01, n = 4, seed = 0):
    
    # set random seed
    np.random.seed(seed = seed)
    
    # original prediction
    preds = model.predict_proba(X_test, num_iteration = num_iteration)[:, 1] / (n + 1)
     
    # select numeric features
    num_vars = [var for var in X_test.columns if X_test[var].dtype != "object"]
    
    # synthetic predictions
    for i in range(n):
        
        # copy data
        X_new = X_test.copy()
                  
        # introduce noise
        for var in num_vars:
            X_new[var] = X_new[var] + alpha * np.random.normal(0, 1, size = len(X_new)) * X_new[var].std()
            
        # predict probss
        preds_new = model.predict_proba(X_new, num_iteration = num_iteration)[:, 1]
        preds += preds_new / (n + 1)
    
    # return probs
    return preds




##### FUNCTION FOR MEAN TARGET ENCODING 

# replaces factors with mean target values per value
# training data: encoding using internal CV
# validation and test data: encoding using  training data

def mean_target_encoding(train, valid, test, features, target, folds = 5):

    ##### TRAINING

    # cross-validation
    skf = StratifiedKFold(n_splits = folds, random_state = 777, shuffle = True)
    for n_fold, (trn_idx, val_idx) in enumerate(skf.split(train, train[target])):

        # partition folds
        trn_x, trn_y = train.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train.iloc[val_idx], y.iloc[val_idx]

        # loop for facrtors
        for var in features:

            # feature name
            name = "_".join(["mean_target_per", str(var)])

            # compute means
            means = val_x[var].map(trn_x.groupby(var)[target].mean())
            val_x[name] = means

            # impute means
            if n_fold == 0:
                train[name] = np.nan
                train.iloc[val_idx] = val_x
            else:
                train.iloc[val_idx] = val_x


    ##### VALIDATION

    # loop for factors
    for var in features:
        means = valid[var].map(train.groupby(var)[target].mean())
        valid[name] = means


    ##### TEST
    
    # copy data
    tmp_test = test.copy()

    # loop for factors
    for var in features:
        means = tmp_test[var].map(train.groupby(var)[target].mean())
        tmp_test[name] = means
        
        
    ##### CORRECTIONS

    # remove target
    del train[target], valid[target]
    
    # remove factors
    for var in features:
        del train[var], valid[var], tmp_test[var]

    # return data
    return train, valid, tmp_test



########### CLASS FOR CUSTOM METRIC FOR CATBOOSTCLASSIFIER

# set catboost.CatBoostClassifier(...,eval_metrics = CustomMetric(),...)
import numpy as np
class CustomMetric(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes - list of list-like objects (one object per approx dimension)
        # target - list-like object
        # weight - list-like object, can be None
        
        y_true = np.array([int(i) for i in target])
        y_preds = np.array([i for i in approxes[0]])
        res = prediction_reward(y_true, y_preds)[1]
        return res, 0
