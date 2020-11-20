import math
import re # regular expression
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


from scipy.stats import mode
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv('TrasfAttributeFraction_RateIncome_NOMonthlyRateMonthlyIncome.csv')
dfprova = df[['DistanceFromHome','FractionYearsAtCompany','PercentSalaryHike','RateIncome']]
# dfprova = df[[,'FractionYearsAtCompany','YearsInCurrentRole',]]
from sklearn.preprocessing import RobustScaler

scaler = StandardScaler()
X = scaler.fit_transform(dfprova)
i = 0
a = np.arange(0.1 ,2 , 0.01)
for vall in a:
    dbscan = DBSCAN(eps = vall, min_samples=10, algorithm='ball_tree', metric='euclidean', leaf_size=100).fit(X)
    print (f'Clusters esistenti:{set(dbscan.labels_)} con eps = {vall:.3f} ')
    count=np.unique(dbscan.labels_, return_counts=True)
    print(count)
    if (len(count[0]) > 2) and (len(count[0]) < 7)  and (count[1][0]<100) and (count[1][1]>80):
        print(f'{len(count[0])} Clusters  con {count[1][0]} noise points')
        if silhouette_score(X, dbscan.labels_) < 0.25:
            print(silhouette_score(X, dbscan.labels_))
            for vak in dfprova:
                for val in dfprova:
                    for va in dfprova:
                        if (val != vak) and (val != va) and (va != vak):
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(dfprova[vak], dfprova[val], dfprova[va], c=dbscan.labels_)
                            ax.set_xlabel(vak)
                            ax.set_ylabel(val)
                            ax.set_zlabel(va)
                            i += 1
    print(i)
    plt.show()
dist = pdist(X, 'euclidean') #pair wise distance
dist = squareform(dist) #distance matrix given the vector dist
k = 10
kth_distances = list()
for d in dist:
    index_kth_distance = np.argsort(d)[k]
    kth_distances.append(d[index_kth_distance])

plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
plt.ylabel('dist from %sth neighbor' %k)
plt.xlabel('sorted distances')
plt.grid(True)
plt.show()