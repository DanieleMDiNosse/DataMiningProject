import math
import re # regular expression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv('Test_HR_Employee_Attrition.csv')
# print(df.head(3)) # Print prime 3 righe
# print(df.tail(3)) # Print ultime 3 righe
# print(df.columns) # Print attribute
# print(df['Age'][0:5]) # Print una sola colonna e prime 5 righe
# print(df[['Age', 'JobLevel']]) # Print più colonne

# df_xlsx = pd.read_excel('name of file', delimiter='\t) # Importare da file excel con delimiter tra i record
# print(df.iloc[1:4]) # Stampa intere righe (ultima esclusa)
# print(df.iloc[1,1]) # Stampa un elemento preciso in questo caso riga 1 colonna 1

# for index, row in df.iterrows(): # Iteration for each row
#     print(index, row)            # alternativa print(index, row['Age']) 

# print(df.loc[df['Attrition'] == 'Yes']) # Finding a specific value for the attribute in a dataset
# print(df.loc[df['Attrition'].str.contains('Yes|no', flags=re.I, regex=True)])
# print(df.dtypes) # Stampa il tipo degli attributi
# print(df.shape) # Shape del DataFrame

############# Istogrammi #############

# plt.figure()
# plt.hist(df.values[:,32],bins = 5, edgecolor='k') #edge color,hist separati da linea NEGRA
# plt.xticks(np.arange(0,20.4, 3.4))
# plt.xlabel('Years With Curr Manager ')
# plt.ylabel('Number of employes')

# handles=[Line2D([0],[0],marker='o',color='w',linestyle='', markerfacecolor='w'),Line2D([0],[0],marker='o',color='w', markerfacecolor='w'),Line2D([0],[0],marker='o',color='w', markerfacecolor='w'),Line2D([0],[0],marker='o',color='w', markerfacecolor='w')]
# labels=['1 Low','2 Medium','3 High','4 Very high']
# plt.legend(handles,labels, edgecolor='w', loc='upper left')


#sub plot organizzati in matrici, una sola figura più plot
# figure, axes = plt.subplots(4, 4)
# # df['Age'].plot(ax=axes[0])
# for i in range(0,4):
#     for j in range(0,4):
#         df['Age'].plot(ax=axes[i,j])

# plt.show()

#bins = np.arange(x0, x1 + 1.5) - 0.5 # per centrare i bin

############# STATISTICA #############

df = df.drop(columns=['StandardHours']) # Rimuovere colonne

corrmatrix = df.corr()
# print(corrmatrix)
for index in range(len(corrmatrix.columns)): # Iteration for each columns
    vecmax = corrmatrix.iloc[index]
    vecmax2 = [i for i in vecmax if i < 1]
    print(max(vecmax2))

# pd.plotting.scatter_matrix(df.iloc[:,0:15], diagonal='kde')
# plt.show()



for c in df['Attrition'].unique():
    dfc = df[df['Attrition'] == c]
    plt.scatter(dfc['DistanceFromHome'], dfc['HourlyRate'], label=c)
plt.legend()
plt.show()
# print(corrmatrix.min())
# corrmatrix.to_excel('MatriceDiCorrelazione.xlsx')
# vecmax = corrmatrix.iloc[i]
# vecmax2 = [i for i in vecmax if i < 1]
# print(vecmax)
# print(max(vecmax2))



# print(df.sort_values(['Age', 'YearsWithCurrManager'], ascending=[1,0])) #sorted data (NaN in coda)

#print(df.describe()) # print count,mean,std,min,25%,50%,75%,max

#--------------------------------------------------------------------------

#Stampa minimo e massimo dei valori nelle singole colonne e i loro range:
# 1° modo
# l = [0,3,5,6,8,10,11,12,14,16,17,18,21,22,23,24,25,26,27,28,29,30,31,32]
# for i in range(len(l)):
#    print(f' {l[i]} : min e max di {df.columns[l[i]]} sono {min(df.values[:,l[i]])} e {max(df.values[:,l[i]])}. Range --> {max(df.values[:,l[i]])-min(df.values[:,l[i]])}.')

# 2° modo
# print(df.max())
# print(df.min())

#--------------------------------------------------------------------------

# statistica = df.describe()
# statistica.to_excel('statistica.xlsx') #esporta la tabella in excel

#--------------------------------------------------------------------------

# etamin = (df[(df['Age'] > 17) & (df['Age'] < 19)]) #seleziona solo le righe con valori di età >17 & < 19
# print(etamin)
# etamin = etamin.reset_index(drop=True) # Resetta indici

# print(df.loc[~df['MaritalStatus'].str.contains('Single')]) # Drop tutte le righe con single in marital status

# print(df['Age'].unique()) # valori non ripetuti attributo "Age"
# print(df.groupby(['Age']).mean().sort_values('YearsWithCurrManager', ascending=False)) # valori divisi in gruppi per età with mean

# for i in range(len(df)):
#     print (f'Age {df.values[:,0][i]}  == {df.values[:,-1][i]}')

#conoscere i valori max e min del guadagno mensile di gruppi organizzati per relazione coniugale
# print(df.groupby(['MaritalStatus'], sort=False)['MonthlyIncome'].max())
# print(df.groupby(['MaritalStatus'], sort=False)['MonthlyIncome'].min())

# df['T'] = df[''] + df[''] # Addizionare colonne
# df['T'] = df.iloc[:, i:j].sum(axis=1) #axis=1 somma orizzontalmente, =0 verticalmente
# df.to_csv = ('.csv', index=False)
# df.to_excel = ('.xlsx', index=False)
# df.to_csv = ('.txt', index=False, sep='\t')
