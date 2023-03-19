# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:44:37 2021

@author: Sagming Marcel
"""

#Importing libraries
import numpy as np
import pandas as pd
from ripser import ripser
from persim import plot_diagrams


def Ripser_Code():


    #===============================================================================================================================================
    for j in range(2):
            if j==0:
                filename_churn = 'C:\\Users\\msagming\\Desktop\\PhD Research\\PhD Source Code\\Code\\No_Churn_Landmarks\\No_Churn_'
            
            elif j==1:
                
                filename_churn = 'C:\\Users\\msagming\\Desktop\\PhD Research\\PhD Source Code\\Code\\Churn_Landmarks\\Churn_'


            #loop and change file names then use ripser and save H1 to file
            for i in range(500):
               
                        filename = filename_churn+str(i+1)+'.csv'
                        #Read data file 
                        landmark_set = pd.read_csv(filename, header=None)
                
                        #call ripser
                        points = ripser(landmark_set)['dgms']
                
                        #plot H0 and H1
                        #plot_diagrams(points, show=True)
                
                        #extract H0 & H1
                        points_h0 = np.array(points[0])
                        points_h1 = np.array(points[1])
                
                        #extract the relavant columns
                        points_h0_0 = points_h0[:,0]
                        points_h0_1 = points_h0[:,1]
                        points_h1_0 = points_h1[:,0]
                        points_h1_1 = points_h1[:,1]
                
                        #replace infinity with a number 1.41421
                        points_h0_1[~np.isfinite(points_h0_1)]=1.41421
                        points_h1_0[~np.isfinite(points_h1_0)]=1.41421
                        points_h1_1[~np.isfinite(points_h1_1)]=1.41421
                
                        #compute statistics for H0
                        length_0 = abs(points_h0_1 - points_h0_0);
                        y_max_0 = np.max(points_h0_1);
                        ymlength_0 = y_max_0 - points_h0_1;
                
                        a11 = np.mean(points_h0_0);         a12 = np.mean(points_h0_1);         a13 = np.mean(length_0);        a14 = np.mean(ymlength_0);
                        a21 = np.median(points_h0_0);       a22 = np.median(points_h0_1);       a23 = np.median(length_0);      a24 = np.median(ymlength_0);
                        a31 = np.std(points_h0_0);          a32 = np.std(points_h0_1);          a33 = np.std(length_0);         a34 = np.std(ymlength_0);
                
                        #compute statistics for H1
                        length_1 = abs(points_h1_1 - points_h1_0);
                        y_max_1 = np.max(points_h1_1);
                        ymlength_1 = y_max_1 - points_h1_1;
                
                        b11 = np.mean(points_h1_0);         b12 = np.mean(points_h1_1);         b13 = np.mean(length_1);        b14 = np.mean(ymlength_1);
                        b21 = np.median(points_h1_0);       b22 = np.median(points_h1_1);       b23 = np.median(length_1);      b24 = np.median(ymlength_1);
                        b31 = np.std(points_h1_0);          b32 = np.std(points_h1_1);          b33 = np.std(length_1);         b34 = np.std(ymlength_1);
                
                
                        #Compute the statistics matrices
                        M0 = [a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34];
                        M1 = [b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34];
                
                        #flatten and concatenate the statistics matrices
                        M = np.concatenate((M0, M1), axis=None)
                
                        #Add relevant label to the output matrix
                        M = np.append(M,j)
                
                        #convert list to comma separated string
                        Output = ','.join(['%.7f' % num for num in M])
                
                        #write list to file
                        filename = 'Dataset_Statistics_15_percent.txt'
                        with open(filename, 'a') as file:
                                file.write('%s\n' % Output)
                        file.close()
                        filename = ''

    #==================================================================================
    #Randomize the statistics file content
    import random
    filename = 'Dataset_Statistics_15_percent.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    f.close()
    filename = 'Dataset_Statistics_15_percent_random.txt'
    with open(filename, 'a') as newfile:
        newfile.writelines(lines)
    newfile.close()
    #==================================================================================

if __name__ == '__main__':
    Ripser_Code()

