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

def data_regularization(df_train, df_test):
    '''
    regularization of data
    '''
    print("regularization section: ", df_train.shape)