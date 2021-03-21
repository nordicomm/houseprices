# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import statsmodels.formula.api as sm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot






def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


#loading data. 

def fig_draw(var1, var1_title, var2, var2_title):
    '''
    drawing histogram of two variables. 
    '''
    prices = pd.DataFrame({var1_title:var1, var2_title:var2})

    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices.hist()
# end of fig_draw

def missing_data(df_train):
    '''
    Important questions when thinking about missing data:
        - How prevalent is the missing data?
        - Is missing data random or does it have a pattern?
        
    
    '''
    #print(df_train['GarageCond'].describe())

    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    print(missing_data.head(20))
    
    
    #dealing with missing data
    
    # We'll consider that when more than 15% of the data is missing, 
    # we should delete the corresponding variable and pretend it never existed. 
    # This means that we will not try any trick to fill the missing data in these cases.
    
    
    df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
    
    # In summary, to handle missing data, we'll delete all the variables 
    # with missing data, except the variable 'Electrical'. 
    # In 'Electrical' we'll just delete the observation with missing data.
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    
    #just checking that there's no missing data missing...
    #print(df_train.isnull().sum().max()) 
        
    return df_train

# end of missing_data

def labelencoder_train(df_train):
    '''
    We will lebel encode following variables: 
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
       'PavedDrive', 'SaleType', 'SaleCondition', 'YearBuilt'
    '''
    #dealing with columns with non-integer values. 
    onehot_req_index = df_train.dtypes[df_train.dtypes == "object"].index
   
    
    for ind in onehot_req_index:
        labelencoder_X = LabelEncoder()
        df_train[ind] = labelencoder_X.fit_transform(df_train[ind])
        unique =len( df_train[ind].unique())
        print(ind + ": (unique)  : " + str(unique))
        
        
    # sns.set()
    cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
        'PavedDrive', 'SaleType', 'SaleCondition']
    
    # fig, ax = plt.subplots() # Create the figure and axes object
    # df_train.plot(x = 'Id', y = 'Neighborhood', ax = ax) 

    # df_train.plot(x = 'Id', y = 'SalePrice', ax = ax, secondary_y = True) 
    
    #box plot overallqual/saleprice
    # var = 'SaleCondition'
    # data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    # f, ax = plt.subplots(figsize=(8, 6))
    # fig = sns.boxplot(x=var, y="SalePrice", data=data)
    # fig.axis(ymin=0, ymax=800000);
    
    print(df_train.count())
    
    '''
    trying to find out the feature using SelectKBest statistical model based 
    on https://machinelearningmastery.com/feature-selection-with-categorical-data/
    '''
    # x = df_train[cols]
    # y = df_train['SalePrice']
    
    
    # fs = SelectKBest(score_func=chi2, k='all')
    # fs.fit(x, y)
    # x_train_fs = fs.transform(x)
    
    
    # what are scores for the features
    # i = 0
    # for c in cols:
    #     print(f"{c}, {fs.scores_[i]: .2f}")
    #     i += 1
        
    # # plot the scores
    # pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # pyplot.show()

    

    
    


    # #f, ax = plt.subplots(figsize=(12, 9))
    # #sns.heatmap(corrmat, vmax=.8, square=True);
    
    # k = 10
    # cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    # cm = np.corrcoef(df_train[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    # plt.show()
    
    # print(df_train['YearBuilt'])
    
    


    
    

def doing_one_hot_encoding(train):
    '''
    finding the variables that could be used for the one-hot encoding
    Conditions to use the variable: 
        1. Priority
        2. checking how many missing values the variable has. 
    
    '''
    print(train.isnull().sum())
    #handling the missing values
    #df_train= missing_data(train)
    
    #df_train_label_encoded = labelencoder_train(df_train)
    
    
    # missing_perc = []
    # del_list = []
    
    # for ind in onehot_val:
    #     perc = train[ind].isnull().sum()/1460 * 100.0
        
    #     missing_perc.append([ind, perc])
    #     if perc > 10:
    #         del_list.append(ind)
    #         print(f" removed {perc:.2f}, {ind}")
    
    # labelencoder_S = LabelEncoder()
    # train['Street'] = labelencoder_S.fit_transform(train['Street'])
    # onehotencoder = OneHotEncoder(categorical_features= [2])
    # train = onehotencoder.fit_transform(train).toarray()
    
    
        
    

'''
    Loading data from the csv file. 
'''
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

all_train_data =  train.loc[:,'MSSubClass':'SaleCondition']

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

'''
    Data processing
    ****
'''

# checking the normalization of the price variable. 
#fig_draw(train["SalePrice"], "price",  np.log1p(train["SalePrice"]), "log(price + 1)" )


'''
    Convert the variables into usable parameters through one-hot-encoding. 
    
'''

numeric_val = all_data.dtypes[all_data.dtypes != "object"].index


req_enc = doing_one_hot_encoding(train)

# data processing
# first trasforming the skewed numeric features by taking log(feature +1)
# this will make the feature more normal


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#separate numeric values
#get the value to process in the regression. 




skewed_val = train[numeric_val].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_val = skewed_val[skewed_val > 1.0]
skewed_val = skewed_val.index

all_data[skewed_val] = np.log1p(all_data[skewed_val])

#refining the data a bit. 
all_data = pd.get_dummies(all_data)

#filling NA data with the mean. 
all_data = all_data.fillna(all_data.mean())

#creating matrices for the sklearn
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
y = train.SalePrice













