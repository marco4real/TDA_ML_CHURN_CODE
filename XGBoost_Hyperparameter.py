# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:35:30 2021

@author: msagming
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe # import packages for hyperparameters tuning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Importing the dataset
dataset = pd.read_csv('Dataset_Statistics_10_percent_random.txt', header = None)
dataset.columns = 'Column' + dataset.columns.astype(str)

x = dataset.iloc[:, :24].values
y = dataset.iloc[:, 24].values


#Split data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Initialize the hyperparameter space with a range of values 
hyperparameter_space={
                      'max_depth': hp.quniform('max_depth', 3, 18, 1),
                      'gamma': hp.uniform('gamma', 1,9),
                      'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
                      'reg_lambda' : hp.uniform('reg_lambda', 0,1),
                      'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
                      'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
                      'n_estimators': hp.uniformint('n_estimators', 0, 200), # Number of trees
                      'seed': hp.quniform('seed',0,10,1)
                     }
#Define the objective function
def objective(hyperparameter_space):
    
    clf=xgb.XGBClassifier(n_estimators =hyperparameter_space['n_estimators'], max_depth = int(hyperparameter_space['max_depth']), gamma = hyperparameter_space['gamma'],
                    reg_alpha = int(hyperparameter_space['reg_alpha']),min_child_weight=int(hyperparameter_space['min_child_weight']),
                    colsample_bytree=int(hyperparameter_space['colsample_bytree']), seed=int(hyperparameter_space['seed']), use_label_encoder=False)
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,eval_set=evaluation, eval_metric="auc", early_stopping_rounds=10,verbose=2)
    
    y_pred = clf.predict(X_test)
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Compute the accuracy
    accuracy = cm.trace()/cm.sum()
    print("Accuracy: %.2f%%" % (accuracy*100.00))
    
    # Compute the precision
    precision = precision_score(y_test, y_pred)
    print("Precision: %.2f%%" % (precision*100.00))
    
    # Compute the recall
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall: %.2f%%" % (recall*100.00))
    
    # Compute the precision
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F-Measure: %.2f%%" % (f1*100.00))
    
    return {'loss': -accuracy, 'status': STATUS_OK }

#Trials is an object that stores all the relevant information such as hyperparameter, loss-functions for each set of parameters that the model has been trained.
trials = Trials()

#Best_parameter gives us the optimal parameters that best fit model and better loss function value.
#fmin is an optimization function that minimizes the loss function
best_hyperparams = fmin(fn = objective,space = hyperparameter_space, 
                        algo = tpe.suggest, max_evals = 500, trials = trials)

#Print the best parameters
print("The best hyperparameters are : ","\n")
print(best_hyperparams)