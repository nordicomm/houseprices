# -*- coding: utf-8 -*-

'''
    Summary: name and functions
        house_categorical_features.py

    Details:
    Processing the Categorical feature and:
        - understanding them
        - label encoding them
        - one-hot encoding
        
        and possibly look into the embedding
'''

#importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def handle_categorical_features(df_train):
    '''
        find the categorical features in the dataset, and having insights about them.
    '''
    
    category_features_index = df_train.dtypes[df_train.dtypes == "object"].index
    # print(len(category_features_index)) # 39 variables
    
    
    
