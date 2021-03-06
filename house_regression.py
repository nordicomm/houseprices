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

calculate_flag = True
df_train, df_test = understanding_missing_data(train, test, calculate_flag)

#df_train_1, df_train_2 = further_understanding(df_train)
#df_test_1, df_test_2 = further_understanding(df_test)



    
'''
    3- handle categorical feature before processing.
    We have quite many categorical feature in the data, and we need to
    analyze and convert them before using them into our movel.
    
    For this purpose, we have concatinated both train and test data together
    
'''
#all_data_1 = handle_categorical_features(df_train_1, df_test_1)
#all_data_2 = handle_categorical_features(df_train_2, df_test_2)
all_data_complete = handle_categorical_features(df_train, df_test)

print("Checking Data")
print("all_data_1: ", all_data_1.shape)
print("all_data_2: ", all_data_2.shape)
print("all_data_complete: ", all_data_complete.shape)


'''
    4- After processing the data now, we will try to figure out, 
    how this data is related to sales price. 
'''


#df_train_n1, df_test_n1 = understanding_data(all_data_1, df_train_1, df_test_1)
#df_train_n2, df_test_n2 = understanding_data(all_data_2, df_train_2, df_test_2)
df_train_n, df_test_n = understanding_data(all_data_complete, df_train, df_test)




# next todo,... build a model and run the code from it. 
'''
    5- regularization
    Now we are going to use regularized linear regression model from scikit learn
    module:
        - Lasso regularization
        - Ridge regularization
    
    Model Number: defines which model, we will be running here. 
    Ridge Model             1
    XGB Model              10
    Lasso Model           100    
    Elastic Net Model    1000

'''

# redo model flag set "True" will make us repeat the regression again. 
redo_modeling_flag = True
model_number = 100

# model number: change the model number here


#y_pred_1 = data_regularization(df_train_n1, df_test_n1, redo_modeling_flag, model_number)
#y_pred_2 = data_regularization(df_train_n2, df_test_n2, redo_modeling_flag, model_number)
y_pred_lasso = data_regularization(df_train_n, df_test_n, redo_modeling_flag, model_number)
y_pred_xgb = data_regularization(df_train_n, df_test_n, False, 10)
y_pred_ridge = data_regularization(df_train_n, df_test_n, False, 1)


# pred_int = [0] * len(test)
# counter = 0

# for idx in df_test_1['Id']:
#     pred_int[idx-1460] = y_pred_1[0][counter]
#     counter += 1

# counter = 0
# for idx in df_test_2['Id']:
#     if counter < 318:
#         pred_int[idx-1460] = y_pred_2[0][counter]
#     counter += 1


y_pred = 0.7 * y_pred_lasso[0] + 0.3 * y_pred_ridge[0]
# RMSE error: 0.12773




'''
    6- saving the results into submission file. 

'''

pred = pd.DataFrame(y_pred)

read_sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([read_sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)

    


    



