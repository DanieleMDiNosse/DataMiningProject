import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

df = pd.read_csv('Test_HR_Employee_Attrition.csv')
print(df.iloc[0], df.dtypes)

# df.hist(column=14)
# plt.hist(df.values[:,5])

plt.figure()
plt.hist(df.values[:,32],bins = 5, edgecolor='k') #edge color per gli hist separati da una linea NEGRA
plt.xticks(np.arange(0,20.4, 3.4))
plt.xlabel('Years With Curr Manager ')
plt.ylabel('Number of employes')



# l = [0,3,5,6,8,10,11,12,14,16,17,18,21,22,23,24,25,26,27,28,29,30,31,32]
#for i in range(len(l)):
#    print(f' {l[i]} : min e max di {df.columns[l[i]]} sono {min(df.values[:,l[i]])} e {max(df.values[:,l[i]])}. Range --> {max(df.values[:,l[i]])-min(df.values[:,l[i]])}.')


#bins = np.arange(1, 4 + 1.5) - 0.5, edgecolor='k' ###per centrare i bin
# print(df.dtypes)
# print(df.iloc[0])

# figure, axes = plt.subplots(4, 4)
# # df['Age'].plot(ax=axes[0])
# for i in range(0,4):
#     for j in range(0,4):
#         df['Age'].plot(ax=axes[i,j])


plt.show()
