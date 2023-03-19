# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:03:43 2021

@author: msagming
"""

import numpy as np
import pandas as pd

def Statistics_Code():
    #================================================
    # T-test
    from scipy.stats import ttest_ind
    
    v1 = np.random.normal(size=100)
    v2 = np.random.normal(size=100)
    
    #For pvalue only
    #res = ttest_ind(v1, v2).pvalue
    
    res1 = ttest_ind(v1, v2)
    print(res1)
    
    #===============================================
    # KS-test
    from scipy.stats import kstest
    
    v3 = np.random.normal(size=100)
    res2 = kstest(v3, 'norm')
    
    print(res2)
    
    #===============================================
    #Statistical Description
    from scipy.stats import describe
    
    v4 = np.random.normal(size=100)
    res3 = describe(v4)
    
    print(res3)
    
    #===============================================
    #Skewness and Kurtosis
    from scipy.stats import skew, kurtosis
    
    v5 = np.random.normal(size=100)
    
    print(skew(v5))
    print(kurtosis(v5))
    
    #===============================================
    #Normal Test
    from scipy.stats import normaltest
    
    v6 = np.random.normal(size=100)
    
    print(normaltest(v6))
      
#=============================================================================

