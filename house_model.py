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
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# libraries for xgboost
import xgboost

# importing the model
import pickle

# checking the path of the file
from os import path

def rmse_cv(model, X_train, y):
    ''' rmse calculation model taken from internet'''
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


def data_regularization(df_train, df_test, redo_modeling_flag):
    '''
    regularization of data
    '''
    print("regularization section: ", df_train.shape)

    #creating matrices for sklearn:
    X_train = df_train.drop(['SalePrice'],axis=1)
    X_test =  df_test
    y = df_train['SalePrice']
    
    # print(X_train.head())
    # print(X_test.head())
    # print(y)
    
    cv_ridge = ridge_model(X_train, y)
    
    # if you want to redo the model
    
    pridct_y = xgboost_model(X_train, y, X_test, redo_modeling_flag)
    
    return pridct_y
    
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
    cv_ridge.plot(title = "Validation")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    
    print(cv_ridge.min())
    return cv_ridge
    
    
def xgboost_model(X_train, y, X_test, redo_modeling_flag ):
    '''
    Prediction and selecting the model
    '''
    ridge_xgb_regressor_f = 'ridge_xgb_regressor.pkl'
    check_model_exist = path.exists(ridge_xgb_regressor_f)
    
    if check_model_exist == False or redo_modeling_flag == True:

        classifier = xgboost.XGBRegressor()
        regressor = xgboost.XGBRegressor()
        
        booster=['gbtree','gblinear']
        base_score=[0.25,0.5,0.75,1]
        
        # hyper parameter optimization
        n_estimators = [100, 500, 900, 1100, 1500]
        max_depth = [2, 3, 5, 10, 15]
        booster=['gbtree','gblinear']
        learning_rate=[0.05,0.1,0.15,0.20]
        min_child_weight=[1,2,3,4]
        
        # Define the grid of hyperparameters to search
        hyperparameter_grid = {
            'n_estimators': n_estimators,
            'max_depth':max_depth,
            'learning_rate':learning_rate,
            'min_child_weight':min_child_weight,
            'booster':booster,
            'base_score':base_score
            }
        
        random_cv = RandomizedSearchCV(estimator=regressor,
                param_distributions=hyperparameter_grid,
                cv=5, n_iter=50,
                scoring = 'neg_mean_absolute_error',n_jobs = 4,
                verbose = 5, 
                return_train_score = True,
                random_state=42)
        
        random_cv.fit(X_train,y)
        print(random_cv.best_estimator_)
        
        regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
           n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1)
        
        regressor.fit(X_train,y)
    
        y_pred=np.expm1(regressor.predict(X_test))
        
        # saving the model
        pickle.dump(regressor, open(ridge_xgb_regressor_f, 'wb'))
        
        print(y_pred)
    
    else: 
        with open(ridge_xgb_regressor_f, "rb") as input_file:
            regressor = pickle.load(input_file)
        
        y_pred=np.expm1(regressor.predict(X_test))
        
        print("using previously compiled model: ")
        print(y_pred)

    
    return y_pred

    
    
    
    