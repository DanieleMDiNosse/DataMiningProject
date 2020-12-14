import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv', index_col = 0)


# df1 = df[['TrainingTimesLastYear', 'StockOptionLevel', 'MonthlyIncome', 'Age']]
# scaler = MinMaxScaler()
# X = scaler.fit_transform(df1)
# data_dist = pdist(X, metric='euclidean')
# data_link = linkage(data_dist, method='ward', metric='w', optimal_ordering = True)
# # plt.figure()
# plt.title('DF1 sil: 0.40, method: ward, clusters: 2')
# # res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=0.6)
# fclu = fcluster(data_link, 2, criterion="maxclust")
# a = silhouette_score(X, fclu, metric='euclidean')
# print(f'DF1 sil: {a}')


# df2 = df[['YearsInCurrentRole', 'StockOptionLevel', 'MonthlyIncome', 'Age']]
# scaler = MinMaxScaler()
# X = scaler.fit_transform(df2)
# data_dist = pdist(X, metric='euclidean')
# data_link = linkage(data_dist, method='complete', metric='w', optimal_ordering = True)
# # plt.figure()
# plt.title('DF2 sil: 0.386, method: median, clusters: 2')
# # res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=0.6)
# fclu = fcluster(data_link, 2, criterion="maxclust")
# a = silhouette_score(X, fclu, metric='euclidean')
# print(f'DF2 sil: {a}')



df3 = df[['PercentSalaryHike', 'TrainingTimesLastYear', 'MonthlyIncome', 'Age']]
scaler = MinMaxScaler()
X = scaler.fit_transform(df3)
data_dist = pdist(X, metric='euclidean')
data_link = linkage(data_dist, method='ward', metric='w', optimal_ordering = True)
# plt.figure()
# plt.title('DF3 sil: 0.375, method: median, clusters: 4')
res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=6.5)
fclu = fcluster(data_link, 2, criterion="maxclust")
a = silhouette_score(X, fclu, metric='euclidean')
print(f'DF3 sil: {a}')



# df4 = df[['YearsInCurrentRole', 'StockOptionLevel', 'MonthlyRate', 'MonthlyIncome']]
# scaler = MinMaxScaler()
# X = scaler.fit_transform(df4)
# data_dist = pdist(X, metric='euclidean')
# data_link = linkage(data_dist, method='ward', metric='w', optimal_ordering = True)
# # plt.figure()
# plt.title('DF4 sil: 0.374, method: ward, clusters: 3')
# # res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=0.6)
# fclu = fcluster(data_link, 2, criterion="maxclust")
# a = silhouette_score(X, fclu, metric='euclidean')
# print(f'DF4 sil: {a}')


# df5 = df[['Age', 'YearsInCurrentRole', 'TrainingTimesLastYear', 'MonthlyIncome']]
# scaler = MinMaxScaler()
# X = scaler.fit_transform(df5)
# data_dist = pdist(X, metric='euclidean')
# data_link = linkage(data_dist, method='centroid', metric='w', optimal_ordering = True)
# plt.figure()
# plt.title('DF5 sil: 0.341, method: centroid, clusters: 3')
# res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=0.6)
# fclu = fcluster(data_link, 3, criterion="maxclust")
# a = silhouette_score(X, fclu, metric='euclidean')
# print(f'DF5 sil: {a}')


# df6 = df[['FractionYearsAtCompany', 'PercentSalaryHike', 'MonthlyRate', 'DistanceFromHome']]
# scaler = MinMaxScaler()
# X = scaler.fit_transform(df6)
# data_dist = pdist(X, metric='euclidean')
# data_link = linkage(data_dist, method='centroid', metric='w', optimal_ordering = True)
# plt.figure()
# plt.title('DF6 sil: 0.329, method: centroid, clusters: 3')
# res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=0.565)
# fclu = fcluster(data_link, 4, criterion="maxclust")
# a = silhouette_score(X, fclu, metric='euclidean')
# print(f'DF6 sil: {a}')


# df7 = df[['YearsInCurrentRole', 'TrainingTimesLastYear', 'NumCompaniesWorked', 'Age']]
# scaler = MinMaxScaler()
# X = scaler.fit_transform(df7)
# data_dist = pdist(X, metric='euclidean')
# data_link = linkage(data_dist, method='ward', metric='w', optimal_ordering = True)
# plt.figure()
# plt.title('DF7 sil: 0.309, method: single, clusters: 3')
# res = dendrogram(data_link, truncate_mode = 'lastp', color_threshold=0.6)
# fclu = fcluster(data_link, 2, criterion="maxclust")
# a = silhouette_score(X, fclu, metric='euclidean')
# print(f'DF7 sil: {a}')