import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fim import apriori
import tqdm
from scipy.optimize import curve_fit
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


df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome_Reversed.csv')

# df = pd.read_csv('C:/Users/lasal/Desktop/UNIPI notes/Data Mining/DataMiningProject/Codici/TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome_Reversed.csv')

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
itemsets_gen = False

# ======================================================================================================================================================================


basket = train.values.tolist()
if itemsets_gen:
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
    
    plt.figure('Frequent maximal and closed itemsets')
    plt.plot(np.linspace(1,25,50), itemsets_tot, label='Frequent')
    plt.plot(np.linspace(1,25,50), itemsets_maximal, label='Maximal')
    plt.plot(np.linspace(1,25,50), itemsets_closed, label='Closed')
    plt.plot(np.linspace(1,25,50), 72483*(np.linspace(1,25,50)**1.875), linestyle = '--', c='k',alpha = 0.4, label = r'Exponential fit: $Ae^{bs}$')
    # plt.legend()
    plt.xlabel('Support')
    plt.ylabel('Number of itemsets')
    plt.grid(True)
    
    
      
    t1=np.linspace(1,25,50)
    dist1=itemsets_tot
    def model(t1,m,q):
        return m*(1/t1**q)
    popt,pocov=curve_fit(model,t1,dist1)
    
    
    
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
        for c in np.linspace(1,100,200):
            rules = apriori(basket, supp=s, zmin=2, target='r', conf=c, report='ascl')
            lunghezza_conf.append(len(rules))
        plt.plot(np.linspace(1,100,200), lunghezza_conf, label = f'min_supp = {s}')
        plt.xlabel('Confidence')
        plt.ylabel('Number of rules')
        plt.grid(True)
    plt.legend()

# Lift Histogram
    lift_list = []
    lift_sup = 2
    plt.figure()
    rules = apriori(basket, supp=10, zmin=2, target='r', conf=70, report='ascl')
   
    print(f'Number of rules with c=70: {len(rules)}')
    for r in rules: 
        # print(f'lift:{r[-1]}')
        lift_list.append(r[-1])
    
    plt.hist(lift_list, bins=np.arange(0,6,1),label='Confidence=70', color='k')
    rules = apriori(basket, supp=10, zmin=2, target='r', conf=50, report='ascl')
    print(f'Number of rules with c=50: {len(rules)}')
    for r in rules: 
        # print(f'lift:{r[-1]}')
        lift_list.append(r[-1])
    plt.hist(lift_list, bins=np.arange(0,6,1), alpha=0.4, label='Confidence=50',color='blue')
    
    rules = apriori(basket, supp=80, zmin=2, target='r', conf=30, report='ascl')
    print(f'Number of rules with c=30: {len(rules)}')
    interesting_rules = []
    for r in rules: 
        # print(f'lift:{r[-1]}')
        lift_list.append(r[-1])
        lift_thr = 2
        if (r[-1]>lift_thr):
            interesting_rules.append(r)
            print(r)
    print(f'Number of rules with lift > {lift_thr}: {len(interesting_rules)}')
    
    plt.hist(lift_list, bins=np.arange(0,6,1), alpha=0.2, label='Confidence=30')
    plt.grid(True)
    plt.xlabel('Lift')
    plt.ylabel('Number of rules')
    plt.legend()
plt.show()

# ======================================================================================================================================================================

basket_test = test.values.tolist()
print(len(basket_test))
compared_itemsets = []
for item in basket_test:
    if (item[0] == 'No') and (item[1] == 'Research & Development') and (item[7] == 3.0):
        compared_itemsets.append(item)
print(f'Number of occurance rule 1: {len(compared_itemsets)}')


f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'No') and (item[1] == 'Research & Development') and (item[7] == 3.0) and (item[8] == 'Research Director'):
        f11.append(item)
    if (item[0] == 'No') and (item[1] == 'Research & Development') and (item[7] == 3.0) and (item[8] != 'Research Director'):
        f10.append(item)
    if (item[0] != 'No') and (item[1] != 'Research & Development') and (item[7] != 3.0) and (item[8] == 'Research Director'):
        f01.append(item)
    if (item[0] != 'No') and (item[1] != 'Research & Development') and (item[7] != 3.0) and (item[8] != 'Research Director'):
        f00.append(item)
# print(f'f11: {len(f11)}')
# print(f'f01: {len(f01)}')
# print(f'f10: {len(f10)}')
# print(f'f00: {len(f00)}')

# RULE 4
f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'No') and (item[1] == 'Sales') and (item[7] == 2.0000000000000004) and (item[8] == 'Sales Executive'):
        f11.append(item)
    if (item[0] == 'No') and (item[1] == 'Sales') and (item[7] == 2.0000000000000004) and (item[8] != 'Sales Executive'):
        f10.append(item)
    if (item[0] != 'No') and (item[1] != 'Sales') and (item[7] != 2.0000000000000004) and (item[8] == 'Sales Executive'):
        f01.append(item)
    if (item[0] != 'No') and (item[1] != 'Sales') and (item[7] != 2.0000000000000004) and (item[8] != 'Sales Executive'):
        f00.append(item)
print(f'f11: {len(f11)}')
print(f'f01: {len(f01)}')
print(f'f10: {len(f10)}')
print(f'f00: {len(f00)}')

print('======== Predicting Attrition ======== ')
rules = apriori(basket, supp=10, zmin=2, target='r', conf=20, report='ascl')
print(f'Number of rules with c=30: {len(rules)}')
inter_rulesYesUP = []
inter_rulesNoUP = []
inter_rulesYesDOWN = []
inter_rulesNoDOWN = []
lift_list = []
i = 0
for r in rules: 
    lift_list.append(r[-1])
    lift_thrUP = 1.8
    if (r[-1]>lift_thrUP) and (r[0] == 'Yes'):
        inter_rulesYesUP.append(r)
        i += 1
        print(f'Rule {i}: {r}')
    if (r[-1]>lift_thrUP) and (r[0] == 'No'):
        inter_rulesNoUP.append(r)
        # print(r)
    lift_thrDOWN = 0.8
    if (r[-1]<lift_thrDOWN) and (r[0] == 'Yes'):
        inter_rulesYesDOWN.append(r)
        # print(r)
    if (r[-1]<lift_thrDOWN) and (r[0] == 'No'):
        inter_rulesNoDOWN.append(r)
        # print(r)
print()
print('--- Report ---')
print(f'Number of rules with lift > {lift_thrUP}: Yes --> {len(inter_rulesYesUP)}, No --> {len(inter_rulesNoUP)}')
print(f'Number of rules with lift < {lift_thrDOWN}: Yes --> {len(inter_rulesYesDOWN)}, No --> {len(inter_rulesNoDOWN)}')

print()
print('------ Contingecy Tabel Rule 1 ------')
f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'Yes') and (item[4] == 'Low_EnvSat') and (item[12] == 'Better_WorkLifeBalance'):
        f11.append(item)
    if (item[0] != 'Yes') and (item[4] == 'Low_EnvSat') and (item[12] == 'Better_WorkLifeBalance'):
        f10.append(item)
    if (item[0] == 'Yes') and (item[4] != 'Low_EnvSat') and (item[12] != 'Better_WorkLifeBalance'):
        f01.append(item)
    if (item[0] != 'Yes') and (item[4] != 'Low_EnvSat') and (item[12] != 'Better_WorkLifeBalance'):
        f00.append(item)
print(f'f11: {len(f11)}')
print(f'f01: {len(f01)}')
print(f'f10: {len(f10)}')
print(f'f00: {len(f00)}')
acc = len(f11) / (len(f11)+len(f10))
print(f'Accuracy: {acc}')

print()
print('------ Contingecy Tabel Rule 2 ------')
f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'Yes') and (item[4] == 'Low_EnvSat') and (item[5] == 'Male'):
        f11.append(item)
    if (item[0] != 'Yes') and (item[4] == 'Low_EnvSat') and (item[5] == 'Male'):
        f10.append(item)
    if (item[0] == 'Yes') and (item[4] != 'Low_EnvSat') and (item[5] != 'Male'):
        f01.append(item)
    if (item[0] != 'Yes') and (item[4] != 'Low_EnvSat') and (item[5] != 'Male'):
        f00.append(item)
print(f'f11: {len(f11)}')
print(f'f01: {len(f01)}')
print(f'f10: {len(f10)}')
print(f'f00: {len(f00)}')
acc = len(f11) / (len(f11)+len(f10))
print(f'Accuracy: {acc}')

print()
print('------ Contingecy Tabel Rule 3 ------')
f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'Yes') and (item[4] == 'Low_EnvSat'):
        f11.append(item)
    if (item[0] != 'Yes') and (item[4] == 'Low_EnvSat'):
        f10.append(item)
    if (item[0] == 'Yes') and (item[4] != 'Low_EnvSat'):
        f01.append(item)
    if (item[0] != 'Yes') and (item[4] != 'Low_EnvSat'):
        f00.append(item)
print(f'f11: {len(f11)}')
print(f'f01: {len(f01)}')
print(f'f10: {len(f10)}')
print(f'f00: {len(f00)}')
acc = len(f11) / (len(f11)+len(f10))
print(f'Accuracy: {acc}')

print()
print('------ Contingecy Tabel Rule 4 ------')
f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'Yes') and (item[7] == 1.0) and (item[3] == 'Life Sciences'):
        f11.append(item)
    if (item[0] != 'Yes') and (item[7] == 1.0) and (item[3] == 'Life Sciences'):
        f10.append(item)
    if (item[0] == 'Yes') and (item[7] != 1.0) and (item[3] != 'Life Sciences'):
        f01.append(item)
    if (item[0] != 'Yes') and (item[7] != 1.0) and (item[3] != 'Life Sciences'):
        f00.append(item)
print(f'f11: {len(f11)}')
print(f'f01: {len(f01)}')
print(f'f10: {len(f10)}')
print(f'f00: {len(f00)}')
acc = len(f11) / (len(f11)+len(f10))
print(f'Accuracy: {acc}')

print()
print('------ Contingecy Tabel Rule 5 ------')
f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'Yes') and (item[1] == 'Research & Development') and (item[7] == 1) and (item[6] == 'High_JobInv'):
        f11.append(item)
    if (item[0] != 'Yes') and (item[1] == 'Research & Development') and (item[7] == 1) and (item[6] == 'High_JobInv'):
        f10.append(item)
    if (item[0] == 'Yes') and (item[1] != 'Research & Development') and (item[7] != 1) and (item[6] != 'High_JobInv'):
        f01.append(item)
    if (item[0] != 'Yes') and (item[1] != 'Research & Development') and (item[7] != 1) and (item[6] != 'High_JobInv'):
        f00.append(item)
print(f'f11: {len(f11)}')
print(f'f01: {len(f01)}')
print(f'f10: {len(f10)}')
print(f'f00: {len(f00)}')
acc = len(f11) / (len(f11)+len(f10))
print(f'Accuracy: {acc}')

print()
print('------ Contingecy Tabel Rule 8 ------')
f11 = []
f01 = []
f10 = []
f00 = []
for item in basket_test:
    if (item[0] == 'Yes') and (item[1] == 'Sales') and (item[11] == 'High'):
        f11.append(item)
    if (item[0] != 'Yes') and (item[1] == 'Sales') and (item[11] == 'High'):
        f10.append(item)
    if (item[0] == 'Yes') and (item[1] != 'Sales') and (item[11] != 'High'):
        f01.append(item)
    if (item[0] != 'Yes') and (item[1] != 'Sales') and (item[11] != 'High'):
        f00.append(item)
print(f'f11: {len(f11)}')
print(f'f01: {len(f01)}')
print(f'f10: {len(f10)}')
print(f'f00: {len(f00)}')
acc = len(f11) / (len(f11)+len(f10))
print(f'Accuracy: {acc}')


