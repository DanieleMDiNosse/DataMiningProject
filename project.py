# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:29:52 2020

@author: Daniele
"""
<<<<<<< HEAD

=======
import numpy as np
>>>>>>> c2ef56f47511dd8477667407b0eefee2d7679ed3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Test_HR_Employee_Attrition.csv')
print(df.iloc[0], df.dtypes)

<<<<<<< HEAD
# df.hist(column=14)
plt.hist(df.values[:,14])
=======
plt.figure()
plt.hist(df.values[:,3],bins=13)
plt.xticks(np.arange(111,1594,106))
plt.xlabel('Daily rate')
plt.ylabel('Number of employed')
plt.show()

>>>>>>> c2ef56f47511dd8477667407b0eefee2d7679ed3




