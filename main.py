# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:42:47 2021

@author: Marcel Sagming
"""

import ripserCode
import ann
import kernel_svm
import knn

if __name__ == '__main__':
    ripserCode.Ripser_Code() # run ripser code and generate the statistics for HO & H1
    print("ANN Accuracy: ", ann.Neural_Network())# run neural network code and compute the accuracy
    print("knn Accuracy: ", knn.K_Nearest_Neighbour()) # run knn code and compute the accuracy
    print("Kernal_SVM Accuracy: ", kernel_svm.Kernel_SVM()) # run kernel svm code and compute the accuracy
    print("====================================================================")
