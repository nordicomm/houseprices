# -*- coding: utf-8 -*-
'''
    Summary: name and functions
        house_model.py
            data_regularization

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

# libraries to run the regularization
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model, X_train, y):
    ''' rmse calculation model taken from internet'''
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


def data_regularization(df_train, df_test):
    '''
    regularization of data
    '''
    print("regularization section: ", df_train.shape)

    #creating matrices for sklearn:
    X_train = df_train
    X_test =  df_test
    y = df_train['SalePrice']
    
    # print(X_train.head())
    # print(X_test.head())
    # print(y)
    
    ridge_model(X_train, y)
    
def ridge_model(X_train, y):
    ''' 
    The main tuning parameter for the Ridge model is alpha - a 
    regularization parameter that measures how flexible our model is. 
    
    The higher the regularization the less prone our model will be to overfit. 
    However it will also lose flexibility and 
    might not capture all of the signal in the data.
    '''
    
    model_ridge = Ridge()
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha), X_train, y).mean() for alpha in alphas]
    
    cv_ridge = pd.Series(cv_ridge, index = alphas)
    cv_ridge.plot(title = "Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    
    
    
    