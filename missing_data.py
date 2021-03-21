#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:01:55 2020

@author: habibahmed
"""


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


# added to ease the path direction in mac
import os 


path = "~/Downloads/train.csv"
df = pd.read_csv(os.path.expanduser(path))