# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:56:24 2021

@author: msagming
"""
import pandas as pd  
import numpy as np  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Importing the dataset
dataset = pd.read_csv('Dataset_Statistics_15_percent_random.txt', header = None)
dataset.columns = 'Column' + dataset.columns.astype(str)

X = dataset.iloc[:, :24].values
y = dataset.iloc[:, 24].values
 

#Define kernels to transform the data to higher dimension
kernels = ['Linear', 'Polynomial', 'RBF', 'Sigmoid'] 

#Define a function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Linear kernel
        return SVC(kernel='linear', gamma="auto")
        
    elif ktype == 1:
        # Radial Basis Function kernel
        return SVC(kernel='rbf', gamma="auto")
    
    elif ktype == 2:
        # Sigmoid kernel
        return SVC(kernel='sigmoid', gamma="auto")
    
    else:
        # Polynomial kernel
        return SVC(kernel='poly', degree=3, gamma="auto")
    
#Train the SVM classifier by calling the SVC() method from sklearn 
#and fit the model to the data
for i in range(4):
    # Separate data into test and training sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
    # Train a SVC model using different kernel
    svclassifier = getClassifier(i) 
    svclassifier.fit(X_train, Y_train)
    # Make prediction
    Y_pred = svclassifier.predict(X_test)
    # Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(Y_test,Y_pred))
    
#Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma
param_grid = [{'C': [0.05, 0.1, 0.5, 1, 5, 10, 100, 200, 500, 1000], 'kernel': ['linear']},
              {'C': [0.05, 0.1, 0.5, 1, 5, 10, 100, 200, 500, 1000], 'gamma': [1, 0.1, 0.01, 0.001],'kernel': ['linear','rbf','poly','sigmoid']}
             ]


#Create a GridSearchCV object and fit it to the training data
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=2, verbose=2)
grid.fit(X_train,Y_train)

#print optimal estimator
print(grid.best_estimator_)


#Take this grid model to create some predictions using the test set 
#and then create classification reports and confusion matrices
grid_predictions = grid.predict(X_test)
cm = confusion_matrix(Y_test, grid_predictions)
accuracy = cm.trace()/cm.sum()
print("Accuracy: %.2f%%" % (accuracy*100.00))
print(classification_report(Y_test, grid_predictions))

