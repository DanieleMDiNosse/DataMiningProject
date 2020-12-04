import numpy as np
import random
import pandas as pd
from scipy.stats import mode
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# == RANDOM ATTRIBUTES ==================================

def random_attributes(min_val,max_val,integer=None):
    """
    Build a list of random uniform variables
    name: str, name of the attributes
    min_val: float, min values
    max_val: float, max values
    integer: bool (opt), if true the values are integer
    """ 
    name=list()
    for i in range(899):
        if integer:    
            name.append(int(random.uniform(min_val,max_val)))
        else:
            name.append(random.uniform(min_val,max_val))
    return name

DistanceFromHome=random_attributes(1.0,5.3851)
Age=random_attributes(18,60,integer=True)
RateIncome=random_attributes(0.044,0.997)
FractionYearAtCompany=random_attributes(0.0,1.0)

#== DATA FRAME ====================================================

def random_dataframe(list_att,nomi_att):
    """
    From attibutes return the data frame
    """
    dic={}
    for a,d in zip(nomi_att, list_att):
        dic.update({a:d})
    dataframe=pd.DataFrame(dic)
    return dataframe

df=random_dataframe([DistanceFromHome,Age,RateIncome,FractionYearAtCompany],('DistanceFromHome','Age','RateIncome','FractionYearAtCompany',))
# print(df.head())
# ===============================================================

# K MEANS==================================================

scaler = MinMaxScaler()
X = scaler.fit_transform(df.values)
kmeans=KMeans(init='k-means++', n_clusters=4, n_init=100, max_iter=300)
kmeans.fit(X)
sse=kmeans.inertia_
# print(sse)
# ============================================================= 

# CICLO FOR ==================================================
ist_sse=[]
for time in range(500):
    DistanceFromHome=random_attributes(1.0,5.3851)
    Age=random_attributes(18,60,integer=True)
    RateIncome=random_attributes(0.044,0.997)
    FractionYearAtCompany=random_attributes(0.0,1.0)

    df=random_dataframe([DistanceFromHome,Age,RateIncome,FractionYearAtCompany],('DistanceFromHome','Age','RateIncome','FractionYearAtCompany',))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values)
    kmeans=KMeans(init='k-means++', n_clusters=4, n_init=100, max_iter=300)
    kmeans.fit(X)
    sse=kmeans.inertia_
    print(time, sse)
    ist_sse.append(sse)

ist_sse=np.array(ist_sse)
print(f'meam = {ist_sse.mean()}')
print(f'meam = {ist_sse.std()}') 
plt.hist(ist_sse)
plt.show()







