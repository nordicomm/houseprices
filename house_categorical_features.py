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



def handle_categorical_features(df_train, df_test):
    '''
        find the categorical features in the dataset, and having insights about them.
        
        Important: we will concatinate the training and testing data, in order to have same labeling.
    '''
    
    # concatinating both data points: 
    all_data = pd.concat([df_train, df_test], axis=0)
    # print(all_data.shape) # (2919, 77) end result
    
    #taking the categories and doing multicolumns onehot encoding
    category_to_one_hot_simple(all_data)
    
    
#function
def category_to_one_hot_simple(df):
    '''
        converting all category features into one-hot
    
    '''
    category_features_index = df.dtypes[df.dtypes == "object"].index
    print(len(category_features_index)) # 39 variables
    
    
