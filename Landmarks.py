# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:23:39 2021

@author: msagming
"""
import pandas as pd
import numpy as np
import gudhi
from sklearn import preprocessing
import matplotlib.pyplot as pls 

#Import the dataset
dataset = pd.read_csv('Churn_tuned_dataset.csv', header=None)

#Nonormalising the dataset
dataset = dataset.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
dataset_scaled = min_max_scaler.fit_transform(dataset)
norm_dataset = pd.DataFrame(dataset_scaled)

#Extract churn customer data
churn_dataset = norm_dataset.loc[norm_dataset[31] == 1]

#Extract non-churn customer data
no_churn_dataset = norm_dataset.loc[norm_dataset[31] == 0]

#Define Witnesses and landmarks

witnesses_churn = np.array(churn_dataset)
witnesses_no_churn = np.array(no_churn_dataset.sample(n=3672))


for i in range(500):
    
    #Churn Landmark
    landmarks_churn = gudhi.pick_n_random_points(points=witnesses_churn, nb_points=551)
    #landmarks_churn = gudhi.choose_n_farthest_points(points=witnesses_churn, nb_points=184)
    landmarks_churn = pd.DataFrame(landmarks_churn)
    #similar_test = landmarks_churn.loc[landmarks_churn[0] == 0.0095627689528768]
    
    #No_Churn Landmark
    landmarks_no_churn = gudhi.pick_n_random_points(points=witnesses_no_churn, nb_points=551)
    #landmarks_no_churn = gudhi.choose_n_farthest_points(points=witnesses_no_churn, nb_points=184)
    landmarks_no_churn = pd.DataFrame(landmarks_no_churn)
    
    #Plot test
    #pls.plot(x=norm_dataset.index, y=["0", "1", "2"])
    
  
    #Write Landmarks to file
    filename_churn = 'C:\\Users\\msagming\\Desktop\\PhD Research\\PhD Source Code\\Code\\Churn_Landmarks\\Churn_'+str(i+1)+'.csv'
    with open(filename_churn, 'w') as newfile_churn:
        newfile_churn.writelines(landmarks_churn.to_csv(header=None, index=False))
    newfile_churn.close()
    
    filename_no_churn = 'C:\\Users\\msagming\\Desktop\\PhD Research\\PhD Source Code\\Code\\No_Churn_Landmarks\\No_Churn_'+str(i+1)+'.csv'
    with open(filename_no_churn, 'w') as newfile_no_churn:
        newfile_no_churn.writelines(landmarks_no_churn.to_csv(header=None, index=False))
    newfile_no_churn.close()
    
#Witness complex
#witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
#simplex_tree = witness_complex.create_simplex_tree
            #(
                #max_alpha_square=args.max_alpha_square,
                #limit_dimension=args.limit_dimension
            #)