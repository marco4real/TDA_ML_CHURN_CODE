# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:25:08 2021

@author: msagming
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing 

#Import the dataset
dataset = pd.read_csv('Dataset_Statistics_15_percent_random.txt', header = None)
dataset.columns = 'Column' + dataset.columns.astype(str)

#Create KNN Object.
knn = KNeighborsClassifier()

#Read input and output features
x = dataset.iloc[:,0:24]
y = dataset.iloc[:,24]

#Split data into training and testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Training the model.
knn.fit(x_train, y_train)

#Predict test data set.
y_pred = knn.predict(x_test)

#Checking performance our model with classification report.
print(classification_report(y_test, y_pred))

#Checking performance our model with ROC Score.
roc_auc_score(y_test, y_pred)

#List Hyperparameters that we want to tune.
leaf_size = list(range(1,100))
n_neighbors = list(range(1,50))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
classifier = GridSearchCV(knn_2, hyperparameters, cv=2,verbose=2)
#Fit the model
classifier.fit(x_train, y_train) 

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = cm.trace()/cm.sum()

#Print The value of best Hyperparameters
print(cm)
print('Best leaf_size:', classifier.best_estimator_.get_params()['leaf_size'])
print('Best p:', classifier.best_estimator_.get_params()['p'])
print('Best n_neighbors:', classifier.best_estimator_.get_params()['n_neighbors'])
print("Accuracy: %.2f%%" % (accuracy*100.00))