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
    all_data_with_one_hot = category_to_one_hot_simple(all_data)
    #print("before removing duplication: ", all_data_with_one_hot.shape)
    
    #remove duplicated columns
    all_data_with_one_hot = all_data_with_one_hot.loc[:, ~all_data_with_one_hot.columns.duplicated()]
    #print("after removing: ", all_data_with_one_hot.shape)
    #print(all_data_with_one_hot.head(10))
    
    return all_data_with_one_hot
    
#function
def category_to_one_hot_simple(df):
    '''
        converting all category features into one-hot
    
    '''
    category_features_index = df.dtypes[df.dtypes == "object"].index
    #print(len(category_features_index)) # 39 variables
        
    df_dummy = df # copying the data in a different variable
    counter = 0 # counter to avoid concatination error.
    
    for cat in category_features_index:
        
        # creating one_hot columns for the category
        one_hot_columns = pd.get_dummies(df[cat], drop_first = True)
        
        #print([df[cat], one_hot_columns])
        # dropping the category column converted into category
        df.drop([cat], axis = 1, inplace = True)
        
        if counter == 0:
            df_dummy = one_hot_columns.copy() # first instance
            
        else:
            df_dummy = pd.concat([df_dummy, one_hot_columns], axis = 1)
        
        counter += 1
        
    '''
    Now:
        - our category features are converted into one-hot columns
        - original category features are deleted
        - we need to concatinate the all one-hot columns with non-category features
    '''
    df_dummy = pd.concat([df, df_dummy], axis = 1)
    
    return df_dummy


    
    
    
