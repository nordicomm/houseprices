# -*- coding: utf-8 -*-
'''
Kaggle Project: House Prices - Advanced Regression Techniques
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/leaderboard

Predict sales prices and practice feature engineering, RFs, and gradient boosting

'''

#importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#function files created to support
from house_missing_data import *
from house_categorical_features import *
from house_data_exploration import *
from house_model import *


'''
    1- Loading data from the csv file. 
'''

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

#print(train.head()) # load 5 rows, and 81 columns... 

'''
    2- Understanding missing data. 
    There are several approaches for it. In order to keep the code clean, 
    I have build a separate function to process the missing data of the project. 
    
'''

df_train, df_test = understanding_missing_data(train, test)
    
'''
    3- handle categorical feature before processing.
    We have quite many categorical feature in the data, and we need to
    analyze and convert them before using them into our movel.
    
    For this purpose, we have concatinated both train and test data together
    
'''
all_data = handle_categorical_features(df_train, df_test)


'''
    4- After processing the data now, we will try to figure out, 
    how this data is related to sales price. 
'''

df_train_n, df_test_n = understanding_data(all_data, df_train, df_test)

# next todo,... build a model and run the code from it. 
'''
    5- regularization
    Now we are going to use regularized linear regression model from scikit learn
    module:
        - Lasso regularization
        - Ridge regularization

'''

# redo model flag set "True" will make us repeat the regression again. 
redo_modeling_flag = True

y_pred = data_regularization(df_train_n, df_test_n, redo_modeling_flag)

sns.distplot(y_pred);

'''
    6- saving the results into submission file. 

'''

pred = pd.DataFrame(y_pred)

read_sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([read_sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)

    


    



