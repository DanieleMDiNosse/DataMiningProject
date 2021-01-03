import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fim import apriori
'''
Rapresentative set of frequent itemsets:
Maxiaml Frequent Itemset --> An itemset is said to be maximal frequent if none of its immediate supersets are frequent.
Maximal frequent itemsets rapresents the blocks from which you can construct all the other frequent itemsets. Anyway
they do not include any information about the support except that they satify the min sup condition.

Closed Itemset --> An intemset is said to be closed if none of its immediate supersets have the same support count.
One interesting characteristic of closed itemsets is that if we know their support count we can compute the support
count of all the other itemsets using the definition of CI and the anti-monotone propriety of the support count.

Closed Frequent Itemset --> An itemset is said to bea closed frequent itemset if it's closed an its support is greater
than min supp.
Closed frequent itemsets provide a compact rapresentation of the support count of all the frequent itemsets
'''

df0 = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome.csv')
df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome_Reversed.csv')
df['Age'] = df0['Age'].values

column2drop = ['Age', 'PerformanceRating', 'TrainingTimesLastYear', 
                'StockOptionLevel', 'YearsInCurrentRole', 'NumCompaniesWorked','PercentSalaryHike',
                'OverTime', 'RateIncome', 'FractionYearsAtCompany']
df.drop(column2drop, axis=1, inplace=True)

# Binning
df['DistanceFromHomeBin'] = pd.cut(df['DistanceFromHome'].astype(int), 10 , right=False) 

df = df.drop(columns='DistanceFromHome')

# Divide in Train and Test
train = df.iloc[:630]
test = df.iloc[630:]



basket = train.values.tolist()
lunghezza_supp = []
for s in range(1,100):
    itemsets = apriori(basket, supp=s, zmin=2, target='a')
    lunghezza_supp.append(len(itemsets))
plt.figure()
plt.plot(range(1,100), lunghezza_supp)
plt.grid(True)
plt.show()

# print('Number of itemsets:', len(itemsets))


plt.figure()
for s in range(5,25,5):
    lunghezza_conf = []
    for c in range(0,100):
        rules = apriori(basket, supp=s, zmin=2, target='r', conf=c, report='ascl')
        lunghezza_conf.append(len(rules))
    plt.plot(range(0,100), lunghezza_conf, label = f'min_supp = {s}')
    plt.grid(True)
plt.legend()
plt.show()


# lift = []
# for c in range(0,1):
#     rules = apriori(basket, supp=10, zmin=2, target='r', conf=c, report='ascl')
#     print(len(rules))
#     lift.append(rules[4])
# plt.figure()
# plt.plot(range(0,100), lift)
# plt.grid(True)
# plt.show()


# print('Number of rule:', len(rules))

for r in rules:
    if r[0] == 'Male':
        print(r)
    if r[0] == 'Female':
        print(r)