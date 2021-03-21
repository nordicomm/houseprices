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

understanding_data(all_data)

    


    



