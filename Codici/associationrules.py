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


# df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome_Reversed.csv')

df = pd.read_csv('C:/Users/lasal/Desktop/UNIPI notes/Data Mining/DataMiningProject/Codici/TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome_Reversed.csv')

# == DATA FRAME PREPARATION  ====================================================================================================================================================================

column2drop = ['PerformanceRating', 'TrainingTimesLastYear', 
                'StockOptionLevel', 'YearsInCurrentRole', 'NumCompaniesWorked','PercentSalaryHike',
                'OverTime', 'RateIncome', 'FractionYearsAtCompany']
df.drop(column2drop, axis=1, inplace=True)

# Binning
df['DistanceFromHomeBin'] = pd.cut(df['DistanceFromHome'].astype(int), 10 , right=False) 

df = df.drop(columns='DistanceFromHome')
df['JobInvolvement']=df['JobInvolvement'].astype('str')+'_JobInv'
df['EnvironmentSatisfaction']=df['EnvironmentSatisfaction'].astype('str')+'_EnvSat'
df['JobSatisfaction']=df['JobSatisfaction'].astype('str')+'_JobSat'
df['WorkLifeBalance']=df['WorkLifeBalance'].astype('str')+'_WorkLifeBalance'
# Divide in Train and Test
train = df.iloc[:630]
test = df.iloc[630:]
# ======================================================================================================================================================================

# ======================================================================================================================================================================
association_rule=False

# ======================================================================================================================================================================


basket = train.values.tolist()
itemsets_tot = []
itemsets_maximal = []
itemsets_closed = []
for s in np.linspace(1,25,50):
    itemsets = apriori(basket, supp=s, zmin=2, target='a')
    itemsets_tot.append(len(itemsets))

    itemsets = apriori(basket, supp=s, zmin=2, target='c')
    itemsets_closed.append(len(itemsets))

    itemsets = apriori(basket, supp=s, zmin=2, target='m')
    itemsets_maximal.append(len(itemsets))

plt.figure('Frequen, maximal and closed itemsets')
plt.plot(np.linspace(1,25,50), itemsets_tot, label='Frequent')
plt.plot(np.linspace(1,25,50), itemsets_maximal, label='Maximal')
plt.plot(np.linspace(1,25,50), itemsets_closed, label='Closed')
plt.legend()
plt.grid(True)

itemsets_10 = apriori(basket, supp=10, zmin=2, target='c')
interest_itemsets = []
for i in itemsets_10:
    if (i[1]>200) and (len(i[0])>2):
        interest_itemsets.append(i)
print(f'Number of itemsets s > 10: {len(itemsets_10)}')
print(f'Number of interest itemsets s > 200 and shape=3: {len(interest_itemsets)}')
for i in interest_itemsets:
    print(i)
# plt.show()


# print('Number of itemsets:', len(itemsets))

if association_rule:
    plt.figure()
    for s in range(5,25,5):
        lunghezza_conf = []
        for c in range(0,100):
            rules = apriori(basket, supp=s, zmin=2, target='r', conf=c, report='ascl')
            lunghezza_conf.append(len(rules))
        plt.plot(range(0,100), lunghezza_conf, label = f'min_supp = {s}')
        plt.grid(True)
    plt.legend()
    # plt.show()


    lift_list = []
    lift_sup = 2
    rules = apriori(basket, supp=10, zmin=2, target='r', conf=70, report='ascl')
    print(f'Number of rules: {len(rules)}')
    for r in rules: 
        if r[-1]>lift_sup:     
            # print(f'lift:{r[-1]}')
            lift_list.append(r[-1])
    print(f'Number of rules with lift > {lift_sup}: {len(lift_list)}')

plt.show()


# print('Number of rule:', len(rules))

# for r in rules:
#     if r[0] == 'Male':
#         print(r)
#     if r[0] == 'Female':
#         print(r)