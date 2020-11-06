import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

df = pd.read_csv('Test_HR_Employee_Attrition.csv')

# print(df.iloc[0], df.dtypes) # Tipo degli attributi




############ Istogrammi #############

# l = [0,3,5,6,8,10,11,12,14,16,17,18,21,22,23,24,25,26,27,28,29,30,31,32]
# for i in range(len(l)):
#    print(f' {l[i]} : min e max di {df.columns[l[i]]} sono {min(df.values[:,l[i]])} e {max(df.values[:,l[i]])}. Range --> {max(df.values[:,l[i]])-min(df.values[:,l[i]])}.')

# plt.figure()
# plt.hist(df.values[:,24],bins = np.arange(1, 80 + 1.5) - 0.5, edgecolor='k') #edge color per gli hist separati da una linea NEGRA
# plt.xticks(np.arange(60,120,10))
# plt.xlabel('Standard Hours')
# plt.ylabel('Number of employes')
# plt.show()

# handles=[Line2D([0],[0],marker='o',color='w',linestyle='', markerfacecolor='w'),Line2D([0],[0],marker='o',color='w', markerfacecolor='w'),Line2D([0],[0],marker='o',color='w', markerfacecolor='w'),Line2D([0],[0],marker='o',color='w', markerfacecolor='w')]
# labels=['1 Low','2 Medium','3 High','4 Very high']
# plt.legend(handles,labels, edgecolor='w', loc='upper left')

# figure, axes = plt.subplots(4, 4)
# # df['Age'].plot(ax=axes[0])
# for i in range(0,4):
#     for j in range(0,4):
#         df['Age'].plot(ax=axes[i,j])

##########################

############ Stastistica ################

statistica = df.describe()
# statistica.to_excel('statistica.xlsx')











