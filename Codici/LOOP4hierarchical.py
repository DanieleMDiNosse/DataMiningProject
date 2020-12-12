import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv', index_col = 0)
numeric = df.select_dtypes('number')

s = open('dfhierarchical4.txt', 'w')
methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
for vah in numeric:
    for vaj in numeric:
        for val in numeric:
            for vak in numeric:
                if (vak != val) and (vak != vaj) and (vak!=vah) and (val != vaj) and (val != vah) and (vaj!=vah):
                    df = numeric[[val,vak,vah,vaj]]
                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(df)
                    for i in range(3,5):
                        for mtd in methods:
                            data_dist = pdist(X, metric='euclidean')
                            data_link = linkage(data_dist, method=mtd, optimal_ordering = True)
                            fclu = fcluster(data_link, i, criterion="maxclust")
                            if len(np.unique(fclu))>1:
                                a = silhouette_score(X, fclu, metric='euclidean')
                                if a > 0.3:
                                    print(a, mtd, i, vak, val, vah, vaj )  
                                    s.write(f'dataframe: {vak} {val} {vaj} {vah}')
                                    s.write(f'sil: {a}, method: {mtd}, clusters: {i}')
                                    s.write(f'\n')
                                    s.write(f'=================================================')
                                    s.flush()
                            if a<0.2:
                                print('Angelo! Sei solo come un cane, aspetta ancora...')

                            if a<0.1:
                                print('Angelo! Nessuno ti vuole, meglio morire...')

                            else:
                                print('Angelo! Fai skifo, aspetta di più...')

s.close()

