##!pip install outlier_utils

import math
import re # regular expression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from outliers import smirnov_grubbs as grubbs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

df=pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv',index_col = 0)
print(df.columns)
# print(df.describe())
# print(df.head())
# df['RateIncome']=np.sqrt(df['RateIncome'])
# df.to_csv('TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome.csv',index=False)
# print(df.head())
# numeric=df.select_dtypes('number')
# categorical=df.select_dtypes(exclude='number')

# df['MonthlyIncome']=np.exp(df['MonthlyIncome'])
# df['MonthlyRate']=(df['MonthlyRate']**2)
# df.insert(20, 'RateIncome', df['MonthlyIncome']/df['MonthlyRate'], True)
# #Income=log, #MontlyRate=sqrt
# # print(df[['MonthlyIncome','MonthlyRate','RateIncome']])

# for vak in df['RateIncome']:
#     fil = (df['RateIncome'] >= 1)
#     df.drop(list(df.index[fil == True]), inplace=True)

# df['MonthlyIncome']=np.log(df['MonthlyIncome'])
# df['MonthlyRate']=np.sqrt(df['MonthlyRate'])

# print(df['RateIncome'].mean(), df['RateIncome'].std(), df['RateIncome'].skew())
# df['RateIncome']=np.sqrt(df['RateIncome'])
# print(df['RateIncome'].mean(), df['RateIncome'].std(), df['RateIncome'].skew())
# plt.figure()
# plt.hist(df['RateIncome'])
# plt.show()
#_____________________________________________________________________________________
KM=False
knee=False
kM_3d=True

correlazione=False
#_____________________________________________________________________________________

#  KMEANS
from scipy.stats import mode
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

numeric = df.select_dtypes('number')
df = df[['PercentSalaryHike', 'TrainingTimesLastYear', 'MonthlyIncome', 'Age']]
numeric = df.select_dtypes('number')
scaler = MinMaxScaler()
X = scaler.fit_transform(numeric.values)


if kM_3d:
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values)
    
    kmeans = KMeans(init = 'k-means++', n_clusters = 4, n_init = 100, max_iter=300)
    kmeans.fit(X)

    print('SSE %s' % kmeans.inertia_)
    print('Silhouette %s' % silhouette_score(X, kmeans.labels_))
    # np.unique(kmeans.labels_, return_counts=True) # grandezza di ogni cluster
    hist, bins = np.histogram(kmeans.labels_, bins=range(0, len(set(kmeans.labels_)) + 1))
    # print(dict(zip(bins, hist))) 
    i = 0
    for val in df:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[val], df['Age'], df['PercentSalaryHike'], c=kmeans.labels_, s=30)
        ax.set_xlabel(val)
        ax.set_ylabel('Age')
        ax.set_zlabel('PercentSalaryHike')
        plt.show()
    
    
    # for idx, columns in df.iteritems():
    #     for index, columns in df.iteritems():
    #         for index_n, columns in df.iteritems():
    #             if idx != index:
    #                 if index != index_n:
    #                     if idx != index_n:
    #                 #print(np.unique(dbscan.labels_, return_counts = True))
    #                         fig = plt.figure()
    #                         ax = fig.add_subplot(111, projection='3d')
    #                         ax.scatter(df[index_n], df[index], df[idx], c=kmeans.labels_, s=30)
    #                         ax.set_xlabel(index_n)
    #                         ax.set_ylabel(index)
    #                         ax.set_zlabel(idx)
    #                         # plt.show()

    # centers=scaler.inverse_transform(kmeans.cluster_centers_)

    # plt.figure()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(df['DistanceFromHome'], df['PercentSalaryHike'], df['NumCompaniesWorked'], c=kmeans.labels_, s=30)
    # ax.scatter(centers[:,1],centers[:,0],centers[:,4],s=500,marker='*',c='crimson',edgecolor='k')
    # ax.set_xlabel('DistanceFormHome')
    # ax.set_ylabel('PercentSalaryHike')
    # ax.set_zlabel('NumCompaniesWorked')
    # # plt.show()  

    # plt.figure('paralle cor centroids')
    # print(centers)

    # plt.plot(centers[2], marker='o',label='Cluster 2', c='teal')
    # plt.plot(centers[0], marker='o',label='Cluster 0', c='gold')
    # plt.plot(centers[1], marker='o',label='Cluster 1', c='purple')
    # plt.plot(centers[3], marker='o',label='Cluster 3', c='mediumaquamarine')
    # plt.legend()
    # plt.show()
    
if KM:
    kmeans = KMeans(n_clusters = 3, n_init = 100, max_iter=300, init='k-means++')
    kmeans.fit(X)

    print('SSE %s' % kmeans.inertia_)
    print('Silhouette %s' % silhouette_score(X, kmeans.labels_))
    # np.unique(kmeans.labels_, return_counts=True) # grandezza di ogni cluster
    hist, bins = np.histogram(kmeans.labels_, bins=range(0, len(set(kmeans.labels_)) + 1))
    # print(dict(zip(bins, hist))) 
    i = 0
    for index, columns in numeric.iteritems():
        for index_n, columns in numeric.iteritems():
            if index != index_n:
                #print(np.unique(dbscan.labels_, return_counts = True))
                # plt.scatter(numeric[index_n], numeric[index], c=kmeans.labels_, s=30)
                # plt.xlabel(index_n)
                #  plt.ylabel(index)
                i += 1
                # plt.show()
                print(i)
    centers=scaler.inverse_transform(kmeans.cluster_centers_)
    plt.figure()
    plt.scatter(df['DistanceFromHome'], df['FractionYearsAtCompany'],s=20,c=kmeans.labels_)
    plt.scatter(centers[:,3],centers[:,1],s=200,marker='*',c='crimson')
    plt.xlabel('DistanceFromHome')
    plt.ylabel('FractionYearsAtCompany')
    # plt.show()  

    plt.figure()
    print(centers)

    plt.plot(centers[2], marker='o',label='Cluster 2', c='teal')
    plt.plot(centers[0], marker='o',label='Cluster 0', c='gold')
    plt.plot(centers[1], marker='o',label='Cluster 1', c='purple')
    plt.legend()
    plt.show()

if knee:
    sse_list = list()
    max_k = 50
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100)
        kmeans.fit(X)
        
        sse = kmeans.inertia_
        sse_list.append(sse)
        
    plt.plot(range(2, len(sse_list) + 2), sse_list)
    plt.ylabel('SSE', fontsize=22)
    plt.xlabel('K', fontsize=22)
    plt.show()

        
#__________________________________________________________________________

#____________________________________________________________________________
#CORRELAZIONE
if correlazione:
    import seaborn as sns
    plt.figure(figsize=(20,12))
    plt.subplots_adjust(bottom=0.2)
    heatmap = sns.heatmap(df.corr(), annot = True)
    plt.savefig('MCHeatMap_final.png', dpi=1000)

# print(df.head(3)) # Print prime 3 righe
# print(df.tail(3)) # Print ultime 3 righe
# print(df.columns) # Print attribute

# print(df['Age'][0:5]) # Print una sola colonna e prime 5 righe
# print(df[['Age', 'JobLevel']]) # Print più colonne

# df_xlsx = pd.read_excel('name of file', delimiter='\t) # Importare da file excel con delimiter tra i record
# print(df.iloc[1:4]) # Stampa intere righe (ultima esclusa)
# print(df.iloc[1,1]) # Stampa un elemento preciso in questo caso riga 1 colonna 1

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

#bins = np.arange(x0, x1 + 1.5) - 0.5 # per centrare i bin
############# STATISTICA #############
# statistica = df.describe()
# statistica.to_excel('statistica.xlsx') #esporta la tabella in excel
# df = df.drop(columns=['StandardHours']) # Rimuovere colonne
# df = df.drop(columns=['Over18'])
# #--------------------------------------------------------------------------------
# #TRASFORMAZIONE DEI DATI NUMERICI IN CATEGORICI(OVVIAMENTE PER QUELLI CHE LO PREVEDONO)
# df.replace({'EnvironmentSatisfaction':{1 : 'Low', 2 : 'Medium', 3 : 'High', 4 : 'Very High'}},inplace=True)
# df.replace({'JobInvolvement':{1 : 'Low', 2 : 'Medium', 3 : 'High', 4 : 'Very High'}},inplace=True)
# df.replace({'JobSatisfaction':{1 : 'Low', 2 : 'Medium', 3 : 'High', 4 : 'Very High'}},inplace=True)
# df.replace({'RelationshipSatisfaction':{1 : 'Low', 2 : 'Medium', 3 : 'High', 4 : 'Very High'}},inplace=True)
# df.replace({'Education':{1 : 'Below College', 2 : 'College', 3 : 'Bachelor', 4 : 'Master', 5 : 'Doctor'}},inplace=True)
# df.replace({'WorkLifeBalance':{1 : 'Bad', 2 : 'Good', 3 : 'Better', 4 : 'Best'}},inplace=True)
# df.replace({'PerformanceRating':{1 : 'Low', 2 : 'Good', 3 : 'Excellent', 4 : 'Outstanding'}},inplace=True)
# # # joblevel valori numerici ma chiaramente categorici, non abbiamo ulteriori informazioni, chiediamo aiuto, per piacere rispondete. stop.
# # # totalworkingyear non ha molto senso a nostro avviso, cioè a 18 anni non puoi aver lavorato una media di 8 anni(QUALITÀ BASSA)
# df.replace({'JobLevel':{1 : '1', 2 : '2', 3 : '3', 4 : '4', 5 : '5'}},inplace=True)
# # df.to_excel('nuovo.xlsx') # NUOVO DATA FRAME MODIFICATO

# categorical = df.select_dtypes(exclude='number') # SELEZIONA SOLO LE COLONNE CATEGORICHE
# numeric = df.select_dtypes('number') # SELEZIONA SOLO LE COLONNE NUMERICHE
# # print(categorical.iloc[0]) # PRINT DELLA PRIMA RIGA DI CATEGORICAL
# categorical = categorical.dropna()
# numeric = numeric.dropna()
# #-------------------------------------------------------------------------------
# # NUOVO DATAFRAME SENZA MISSING (PER ORA SOLO NEGLI ATTRIBUTI CATEGORICI)
# categorical = categorical.fillna('MISSING')



# #----------------------------------------------------------------------------------------------------------------------
# # RICERCA OUTLIERS NELLE COLONNE CON UN PACCHETTINO IMPORTATO SOPRA E ALPHA CHE NON SAPPIAMO COME SCEGLIERNE IL VALORE
# # for index, columns in numeric.iteritems():
# #     outremove = grubbs.test(numeric[index], alpha=0.05)
# #     print(numeric.shape, outremove.shape)
# #----------------------------------------------------------------------------------------------------------------------
# # RICERCA OUTLIERS NELLE COLONNE CON UN DBSCAN (NEL FARLO SI SONO ELIMINTATI I VALORI NaN)

# scaler = MinMaxScaler()
# X = scaler.fit_transform(numeric)
# dbscan = DBSCAN(eps=1.09, min_samples=5)
# dbscan.fit(X)
# print(np.unique(dbscan.labels_, return_counts=True))
# plt.scatter(numeric['Age'], numeric['YearsInCurrentRole'], c=dbscan.labels_)
# plt.show()

# dist = pdist(X, 'euclidean') #pair wise distance
# dist = squareform(dist) #distance matrix given the vector dist

# k = 5
# kth_distances = list()
# for d in dist:
#     index_kth_distance = np.argsort(d)[k]
#     kth_distances.append(d[index_kth_distance])


# plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
# plt.ylabel('dist from %sth neighbor' % k)
# plt.xlabel('sorted distances')
# plt.show()

#-----------------------------MATRICE CORRELLAZIONE-----------------------------
# corrmatrix = df.corr()
# print(corrmatrix)
# for index in range(len(corrmatrix.columns)): # Iteration for each columns
#     vecmax = corrmatrix.iloc[index]
#     vecmax2 = [i for i in vecmax if i < 1]
#     print(max(vecmax2))
# corrmatrix.to_excel('MatriceDiCorrelazione.xlsx') # ESPORTA MATRICE CORRELAZIONE
# pd.plotting.scatter_matrix(df.iloc[:,:], diagonal='kde') # PLOT MATRICE SCATTER
# plt.show()
#----------------------------------------------------------------------------
# SCATTER FOTTUTI PLOT (TUTTI)
# for index_n, columns in numeric.iteritems():
#     if index_n !='Age':
#         for index, columns in categorical.iteritems():
#             for c in categorical[index].unique():
#                     dfc = df[df[index] == c]
#                     plt.scatter(dfc['Age'], dfc[index_n], label=c)
#             plt.legend(bbox_to_anchor=(1,1))
#             plt.xlabel('Age')
#             plt.title('Attribute: '+index)
#             plt.ylabel(index_n)
#             plt.show()
# for index, columns in categorical.iteritems():
#     for c in categorical[index].unique():
#         dfc = df[df[index] == c]
#         plt.scatter(dfc['YearsInCurrentRole'], dfc['YearsWithCurrManager'], label=c)
#     plt.legend(bbox_to_anchor=(1,1))
#     plt.show()
#----------------------------------------------------------------------------
# BOX PLOT
# for element in numeric.columns:
#     plt.figure()
#     df.boxplot(element)
# #---------------------------------------------------------------------------   
# # PRINCIPAL COMPONENT ANALYSIS
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# fig=plt.figure()
# ax=fig.add_subplot(111,projection='3d')

# x0 = numeric.values
# x1 = StandardScaler().fit_transform(x0)
# pca = PCA(n_components = 3) 
# principalComponents = pca.fit_transform(x1)
# principalDf = pd.DataFrame(data = principalComponents
#               , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
# finalDf = pd.concat([principalDf, df[['Attrition']]], axis = 1)
# targets = ['Yes', 'No']
# colors = ['r','b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Attrition'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                 , finalDf.loc[indicesToKeep, 'principal component 2'],
#                 finalDf.loc[indicesToKeep, 'principal component 3'],
#                 c=color, s=50)
# plt.legend(targets)
# print(pca.explained_variance_ratio_)
# #---------------------------------------------------------------------------
 
# df['TrainingTimesLastYear'].value_counts(normalize=True) * 100 # percentuale degli attributi categorici
# # print(df.sort_values(['Age', 'YearsWithCurrManager'], ascending=[1,0])) #sorted data (NaN in coda)

#print(df.describe()) # print count,mean,std,min,25%,50%,75%,max

#Stampa minimo e massimo dei valori nelle singole colonne e i loro range:
# 1° modo
# l = [0,3,5,6,8,10,11,12,14,16,17,18,21,22,23,24,25,26,27,28,29,30,31,32]
# for i in range(len(l)):
#    print(f' {l[i]} : min e max di {df.columns[l[i]]} sono {min(df.values[:,l[i]])} e {max(df.values[:,l[i]])}. Range --> {max(df.values[:,l[i]])-min(df.values[:,l[i]])}.')

# 2° modo
# print(df.min())
# print(numeric.max() - numeric.min())
# print((numeric.max() - numeric.min()).idxmax())
# print(df[(numeric.max() - numeric.min()).idxmax()])

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
