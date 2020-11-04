import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

df = pd.read_csv('Test_HR_Employee_Attrition.csv')
print(df.iloc[0], df.dtypes)

# df.hist(column=14)
# plt.hist(df.values[:,5])

plt.figure()
plt.hist(df.values[:,32],bins = 5, edgecolor='k')
plt.xticks(np.arange(0,20.4, 3.4))
plt.xlabel('Years With Curr Manager ')
plt.ylabel('Number of employes')




#bins = np.arange(1, 4 + 1.5) - 0.5, edgecolor='k'
# print(df.dtypes)
# print(df.iloc[0])

# figure, axes = plt.subplots(4, 4)
# # df['Age'].plot(ax=axes[0])
# for i in range(0,4):
#     for j in range(0,4):
#         df['Age'].plot(ax=axes[i,j])
# plt.show()


plt.show()
