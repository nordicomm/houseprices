# -*- coding: utf-8 -*-
import pandas as pd


def strip_categories_in_variable(text):
    
    '''
    reading the different categories from data_description file and making
    a label encoded usable list out of it. 
    
    This will help us to label the data better. 
    '''
    
    col = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
        'PavedDrive', 'SaleType', 'SaleCondition'] 
    
    counter = 0
    cat_items_list = []
    ret_strip_cat = pd.DataFrame(columns = col)
    var_found_flag = False
    label = 'MSZoning'

    for read_line in text:
        #split the first line
        var_name = read_line.split(":")
        
        if len(var_name) > 1:
                
            if var_name[0] in col:
                var_found_flag = True
                if counter != 0:
                    ret_strip_cat = ret_strip_cat.append(pd.DataFrame(cat_items_list, columns=[label]))
                    cat_items_list = []
                    label = var_name[0]
            
            else:
                var_found_flag = False
                
                    
                # skipping the very first iteration
            counter += 1
        
        elif var_found_flag == True:
            b = read_line.split("\t")
            if len(b) > 1:
                if b[0].strip() != '':
                    cat_items_list.append(b[0].strip())
        
    #eof check for last variable: 
    if var_found_flag == True: 
        ret_strip_cat = ret_strip_cat.append(pd.DataFrame(cat_items_list, columns=[label]))

    print(ret_strip_cat.dtypes.index)
    print(ret_strip_cat['SaleCondition'].describe())
    



# reading file. 
f = open("./data_description.txt", "r")

strip_categories_in_variable(f)



