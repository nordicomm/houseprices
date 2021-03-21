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

# checking the path of the file
from os import path

#function files created to support
from house_missing_data import *
from house_categorical_features import *


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
# checking if missing variable analysis has already been done. 
formulated_train = 'train_forumulated.csv'

if path.exists(formulated_train) == False:
    understanding_missing_data(train, test)
    #checking the heatmap, if there is any value missing
    
    sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
    train.to_csv('train_forumulated.csv', index=False)
    
    df_train = train

else: 
    df_train = pd.read_csv("./train_forumulated.csv")
    print("Missing analysis already done :-) ")
    print("df_traing.shape: ", df_train.shape)

    

'''
    3- handle categorical feature before processing.
    We have quite many categorical feature in the data, and we need to
    analyze and convert them before using them into our movel.
    
'''
handle_categorical_features(train)
    


    



