# -*- coding: utf-8 -*-

'''
    Summary: name and functions
        house_data_exploration.py
            understanding_data
            

    Details:
    Learning about our data
'''

#importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy import stats
from scipy.stats import skew



def understanding_data(all_data, train_old, test_old): 
    
    '''
        In order to understand the data we will:
            - correrlation matrix , in heatmap
            - 'SalePrice' correlation matrix with highest correlation. 
            - scatter plto between most correlated variables
    '''
    # separating the training data from training 
    df_train = all_data.iloc[:len(train_old), :]
    
    # corrmat = df_train.corr()
    # # f, ax = plt.subplots(figsize=(12, 9))
    # # sns.heatmap(corrmat, vmax=.8, square=True);
    
   
    
    #histogram and normal probability plot
    # sns.distplot(df_train['SalePrice'], fit=norm);
    # fig = plt.figure()
    # res = stats.probplot(df_train['SalePrice'], plot=plt)
    
    #df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    
    skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    
     # separating the data 
    df_train = all_data.iloc[:len(train_old), :].copy()
    df_test = all_data.iloc[len(train_old):, :].copy()
    
    #drop the SalePrice data from df_test
    df_test.drop(['SalePrice'], axis = 1, inplace = True)
    
    return df_train, df_test
    
    # corrmat = df_train.corr()

    # #saleprice correlation matrix
    # k = 20 #number of variables for heatmap
    # cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    # cm = np.corrcoef(df_train[cols].values.T)
    # sns.set(font_scale=0.7)
    # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.1f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)
    # plt.show()
    
def further_understanding(df):
    '''
    divide the data based on the GrLivArea
    where ['GrLivArea'] >= 1776
    '''
    # X_train = df_train.drop(['SalePrice'],axis=1)
    # X_test =  df_test
    # y = df_train['SalePrice']
    
    # save copy of the training data
    X_train1  = df.copy()
    X_train2  = df.copy()
    
    #dividing data into two parts. 
    X_train1.drop(X_train1.loc[X_train1['GrLivArea'] >= 1776].index, inplace=True)
    X_train2.drop(X_train2.loc[X_train2['GrLivArea'] < 1776].index, inplace=True)    
      
    # print("further_understanding")
    # print(X_train1.describe())
    # print(X_train2.describe())

    # #append logic
    # X_train = X_train1.append(X_train2, ignore_index = True)
    # X_train = X_train.sort_values(by = ['Id'])
    

    return X_train1, X_train2
    
    
