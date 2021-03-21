# -*- coding: utf-8 -*-


'''
    Summary: name and functions
    house_missing_data.py
        understanding_missing_data
        fill_and_drop_missing_data
        find_missing_data
    
    Details:
    This file enclosed all the investigation related to the missing data. 
    
    Important questions when thinking about missing data:
        - How prevalent is the missing data?
        - Is missing data random or does it have a pattern?
        
    As evaluating missing data can lead to reduction of sample size. 
    
    Important: We need to ensure that the missing data process is not biased and
    hiding an inconvenient truth. 
        
'''

#importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



''' /* Division from the main code */ '''

def understanding_missing_data(df_train, df_test):
    '''
    In order to understand missing data, we follow following steps:
        i. understand the data
    '''

    
    # **********************
    # i. understand the data
    
    # print(df_train.shape) # (1460, 81)
    # print(df_test.shape) # (1460, 80) SalePrice data is missing
    
    missing_train_data = find_missing_data(df_train)
    missing_test_data = find_missing_data(df_test)
    
    # dropping and filling the missing data
    fill_and_drop_missing_data(df_train, missing_train_data)
    fill_and_drop_missing_data(df_test, missing_test_data)

    
    
    
''' ******************************************'''

#function
def fill_and_drop_missing_data(df, missing_train_data):
    '''
    first analyzing training , and test data: 
        Key results: 
            - 
    '''
    print(missing_train_data.head(25))
    
    missing_index = (missing_train_data[missing_train_data['Total'] > 0]).index
    
    
    '''
    Analyzing Training Data df_train
    
                  Total    Percent
    PoolQC         1453  99.520548
    MiscFeature    1406  96.301370
    Alley          1369  93.767123
    Fence          1179  80.753425
    FireplaceQu     690  47.260274
    LotFrontage     259  17.739726
    GarageCond       81   5.547945
    GarageType       81   5.547945
    GarageYrBlt      81   5.547945
    GarageFinish     81   5.547945
    GarageQual       81   5.547945
    BsmtExposure     38   2.602740
    BsmtFinType2     38   2.602740
    BsmtFinType1     37   2.534247
    BsmtCond         37   2.534247
    BsmtQual         37   2.534247
    MasVnrArea        8   0.547945
    MasVnrType        8   0.547945
    Electrical        1   0.068493
    Utilities         0   0.000000
    
    It does seem like (PoolQC, MiscFeature, Alley, Fence, FireplaceQU, LotFrontage)
    might not be useful as they have quite high numbers of missing data. 
    
    Let's see what kind of of significance they have with sales data.
    
    PoolQC: (remove)
        There are only 7 datapoints available. It could be removed. 
    
    MiscFeature: (remove)
        
    Alley: (Keeping) -> next step, one-hot encoding
        two options, pave and gravel. It seems like that houses with the 
        indications on Pave are in a little higher range than the houses in gravel. 
        we do need additional help for the house between sales price of 200K-300K. 
        
    Fence: (Keeping -> GdPrv and its effect on the price)
        GdPrv (Good privacy) does effect the price a bit higher 
        where it is dealing with the same area
    
    FireplaceQu: (Keepting -> Ex and its effect on the price)
        Ex - Excellent and exceptional masonary fireplace effect the price majorly for the similar size house. 
    
    LotFrontage: (Pending)
        Linear feet connected to the street have a linear relationship 
        with the LotArea under 25000. 
        
    Garage Data: (Keeping) -> This data will need further one-hot encoding investigation, before we use it. 
    
    '''
    
    # ii. what is the effect of really high percentage of missing data. 
    
    # checking the box plot for every variable sale price
    # var = 'GarageCond'
    # data = pd.concat([df['SalePrice'], df[var]], axis=1)
    # f, ax = plt.subplots(figsize=(8, 6))
    # fig = sns.boxplot(x=var, y="SalePrice", data=data)
    # fig.axis(ymin=0, ymax=800000);
    
    # checking the box plat with LotArea,... seeting variable impact with area, and saleprice. 
    # assumption if there is a price difference with same area and some sort of indication in the variable
    # its worth keeping it. 
    
    # data = pd.concat([df['LotArea'], df[var]], axis=1)
    # f, ax = plt.subplots(figsize=(8, 6))
    # fig = sns.boxplot(x=var, y="LotArea", data=data)
    # fig.axis(ymin=0, ymax=100000);
    
    # removing two variables. 
    
    # reason, insufficient amount of data in both train and test data
    drop_list_train = (missing_train_data[missing_train_data['Percent'] > 60]).index
    print(drop_list_train)

    # dropping the train varialbles
    df.drop(drop_list_train, axis=1, inplace=True)
    
    # remove the dropping list from the missing list
    missing_index = list(set(missing_index) - set(drop_list_train))
    
    
    # filling in the missing places.
    for idx in missing_index: 
        if df[idx].dtypes == "object":
            df[idx] = df[idx].fillna(df[idx].mode()[0])
        
        else: 
            df[idx] = df[idx].fillna(df[idx].mean())
    
    # print(df_train.shape)
    # print(df_train.isnull().sum().sum()) # equal to zero, if there are no missing values. 
    
    # printing heatmap to see if there are any variables with missing values. 
    #sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
    
   
    
    '''
    Understanding the testing data as well. 
    '''
    
#function
def find_missing_data(df):
    '''
    This function will take the dataframe, and return:
        - missing data concatinated as 
            - missing_data_in_numbers
            - missing_data_in_percent
    
    '''
    missing_data_in_numbers = df.isnull().sum().sort_values(ascending=False)
    
    # calculating percentage as well
    missing_data_in_percent = (df.isnull().sum() / 
                               df.isnull().count() * 
                               100).sort_values(ascending=False)
    
    # concatinating the missing data in numbers and percent
    missing_data = pd.concat([missing_data_in_numbers, 
                              missing_data_in_percent], 
                             axis=1, 
                             keys=['Total', 'Percent'])
    
    
    return missing_data

    