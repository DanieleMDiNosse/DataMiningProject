import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv', index_col = 0)
df = df[['PercentSalaryHike', 'FractionYearsAtCompany', 'YearsInCurrentRole', 'RateIncome', 'NumCompaniesWorked']]
scaler = MinMaxScaler()
X = scaler.fit_transform(df.select_dtypes('number'))

methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

# for mtd in methods:
data_dist = pdist(X, metric='euclidean')
data_link = linkage(data_dist, method='ward', metric='w', optimal_ordering = True)
plt.figure()
# plt.title(f'{mtd}')
res = dendrogram(data_link, truncate_mode = 'level', color_threshold=6)
plt.xticks(fontsize=18)

el = []
for element in res['ivl']:
    el.append(int(element))
print(el)


l1 = el[:178]
l2 = el[179:334]
l3 = el[335:538]
l4 = el[539:805]
l5 = el[806:]
print(l1,l2,l3,l4,l5)