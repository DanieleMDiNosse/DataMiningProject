import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv', index_col = 0)
numeric = df.select_dtypes('number')
methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

s = open('df.txt', 'w', encoding='utf-8')
for val in numeric:
    for vak in numeric:
        for va in numeric:
            if vak != val:
                if va != val:
                    if va != vak: 
                        df = numeric[[val,vak,va]]
                        scaler = MinMaxScaler()
                        X = scaler.fit_transform(df)
                        for i in range(2,6):
                            for mtd in methods:
                                data_dist = pdist(X, metric='euclidean')
                                data_link = linkage(data_dist, method=mtd, optimal_ordering = True)
                                fclu = fcluster(data_link, i, criterion="maxclust")
                                if len(np.unique(fclu)) > 1:
                                    a = silhouette_score(X, fclu, metric='euclidean')
                                    if a > 0.3:
                                        print(a, mtd, i, vak, val, 'PercentSalaryHike')
                                        s.write(f'dataframe: {vak},{val},{va}\n')
                                        s.write(f'sil: {a}, method: {mtd}, clusters: {i}\n')
                                        s.write(f'\n')
                                        s.write(f'=================================================\n')
                                        s.flush()
s.close()
print('FINITO')