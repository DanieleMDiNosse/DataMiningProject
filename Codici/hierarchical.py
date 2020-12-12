import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv', index_col = 0)

df1 = df[['StockOptionLevel','Age','DistanceFromHome']]
# df1 = df[['StockOptionLevel','MonthlyIncome','TrainingTimesLastYear']]
# df1 = df[['StockOptionLevel','Age','TrainingTimesLastYear']]

scaler = MinMaxScaler()
X = scaler.fit_transform(df1.select_dtypes('number'))
data_dist = pdist(X, metric='euclidean')
data_link = linkage(data_dist, method='ward', optimal_ordering = True)
res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=8.5)
fclu = fcluster(data_link, 2, criterion="maxclust")
a = silhouette_score(X, fclu, metric='euclidean')
print(a)
plt.title('Ward')
plt.show()