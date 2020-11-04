import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Test_HR_Employee_Attrition.csv')
print(df.iloc[0], df.dtypes)

# df.hist(column=14)
plt.hist(df.values[:,14])

plt.figure()
plt.hist(df.values[:,3],bins=13)
plt.xticks(np.arange(111,1594,106))
plt.xlabel('Daily rate')
plt.ylabel('Number of employed')
plt.show()





