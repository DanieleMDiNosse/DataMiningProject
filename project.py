# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:29:52 2020

@author: Daniele
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Test_HR_Employee_Attrition.csv')
print(df.iloc[0], df.dtypes)

# df.hist(column=14)
plt.hist(df.values[:,14])




