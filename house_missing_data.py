# -*- coding: utf-8 -*-

'''
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

def understanding_missing_data(df):
    '''
    In order to understand missing data, we follow following steps:
        i. understand the data
    '''
    
    # **********************
    # i. understand the data
    
    # print(df.shape) # (1460, 81)
    
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
    
    # print(missing_data.head(20))
    '''
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
    
    Let's see what kind of of significance they have with sales data. '
    '''
    
    # **********************
    # ii. what is the effect of really high percentage of missing data. 
    
    var = 'PoolQC'
    data = pd.concat([df['LotArea'], df[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="LotArea", data=data)
    fig.axis(ymin=0, ymax=100000);
    
    
    
    
    
    