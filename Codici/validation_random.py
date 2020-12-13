import numpy as np
import random
import pandas as pd
from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt





#== PARAMETERS ===============================================================================
num_records = 899         #numer of records for each attributes (see random_attributes)
validation_times = 35000    #numer of random data frame (see ciclo for)
#============================================================================================


#== IF CHECK =================================================================================
kmean_validation = False
db_scan = True
hierarchical_validation = False

# ===========================================================================================



# == DEF FUNCTIONS: RANDOM ATTRIBUTES - RANDOM DATAFRAME==================================
def random_attributes(name,min_val,max_val,integer=None):
    """
    Build a dict of random uniform variables {'name'= [list_value]}
    name: str, name of the attributes
    min_val: float, min values
    max_val: float, max values
    integer: bool (opt), if true the values are integer
    """ 
    list_value=list()
    for i in range(num_records):
        if integer:    
            list_value.append(int(random.uniform(min_val,max_val)))
        else:
            list_value.append(random.uniform(min_val,max_val))
    dictionary={name:list_value}
    return dictionary

def random_dataframe(list_att):
    """
    From list of dictonary return the data frame
    """
    dic={}
    for a in list_att:
        dic.update(a)
    dataframe=pd.DataFrame(dic)
    return dataframe
# ======================================================================= 

# K MEANS CICLO FOR VALIDATION==================================================
if kmean_validation:

    ist_sse=[]
    for time in range(validation_times):
        
        PercentSalaryHike = random_attributes('PercentSalaryHike', 3.316, 5.0)
        FractionYearsAtCompany = random_attributes('FractionYearsAtCompany',0.0,1.0)
        TrainingTimesLastYear = random_attributes('TrainingTimesLastYear', 0.0, 6.0)
        RateIncome = random_attributes('RateIncome', 0.044, 0.997)
        NumCompaniesWorked = random_attributes('NumCompaniesWorked', 0.0, 3.0)

        df = random_dataframe([PercentSalaryHike,FractionYearsAtCompany,TrainingTimesLastYear,RateIncome,NumCompaniesWorked])

        scaler = MinMaxScaler()
        X = scaler.fit_transform(df.values)
        kmeans=KMeans(init='k-means++', n_clusters=4, n_init=100, max_iter=300)
        kmeans.fit(X)
        sse=kmeans.inertia_
        print(f'{time+1}) SSE: {sse}')
        ist_sse.append(sse)

    ist_sse=np.array(ist_sse)
    with open("validation_kmeans_1.txt",'w', encoding='utf-8') as f:
        f.write(f'DATA FRAME\n')
        f.write(f'{df.head()}\n')
        f.write(f'\n')
        f.write(f'number of values for each attributes: {num_records}\n')
        f.write(f'number of random data frames: {validation_times}\n')
        f.write(f'_____________________________________________________________\n')
        f.write(f'\n')
        f.write(f'STATISTIC\n')
        f.write(f'mean (SSE): {ist_sse.mean()}\n')
        f.write(f'std (SSE): {ist_sse.std()}\n')
    
    plt.hist(ist_sse, bins=int((np.log2(num_records)+1)), edgecolor='k')
    plt.xlabel('SSE')
    plt.show()


# == DB-SCAN ====================================================================
if db_scan:
    sil_list=list()
    for time in range(validation_times):
        PercentSalaryHike = random_attributes('PercentSalaryHike', 3.316, 5.0)
        FractionYearsAtCompany = random_attributes('FractionYearsAtCompany',0.0,1.0)
        TrainingTimesLastYear = random_attributes('TrainingTimesLastYear', 0.0, 6.0)
        RateIncome = random_attributes('RateIncome', 0.044, 0.997)
        NumCompaniesWorked = random_attributes('NumCompaniesWorked', 0.0, 3.0)
        YearsInCurrentRole = random_attributes('YearsInCurrentRole', 0.0, 4.2426)

        df = random_dataframe([PercentSalaryHike,FractionYearsAtCompany,YearsInCurrentRole,RateIncome])

        scaler = MinMaxScaler()
        X = scaler.fit_transform(df)

        dist = pdist(X, 'euclidean') #pair wise distance
        dist = squareform(dist) #distance matrix given the vector dist
        k = 7
        kth_distances = list()
        for d in dist:
            index_kth_distance = np.argsort(d)[k]
            kth_distances.append(d[index_kth_distance])

        
        vall=np.mean(sorted(kth_distances)[770:820])

        dbscan = DBSCAN(eps = vall, min_samples=10, algorithm='ball_tree', metric='euclidean', leaf_size=100).fit(X)
        count=np.unique(dbscan.labels_, return_counts=True)
        if (len(count[0]) > 2):
            print(f'{time+1}) sil:{silhouette_score(X, dbscan.labels_)}')
            sil_list.append(silhouette_score(X, dbscan.labels_))
        else:
            print(f'{time+1}) 1 cluster')

    sil_list=np.array(sil_list)
    with open("validation_dbscan_1.txt",'w', encoding='utf-8') as f:
        f.write(f'DATA FRAME\n')
        f.write(f'{df.columns}\n')
        f.write(f'\n')
        f.write(f'number of values for each attributes: {num_records}\n')
        f.write(f'number of random data frames: {validation_times}\n')
        f.write(f'\n')
        f.write(f'________________________________________________________________________\n')
        f.write(f'\n')
        f.write(f'STATISTIC\n')
        f.write(f'numer of data: {len(sil_list)}\n')
        f.write(f'mean (silhouette): {sil_list.mean()}\n')
        f.write(f'std (silouette): {sil_list.std()}\n')

    plt.hist(sil_list, bins=int((np.log2(num_records)+1)), edgecolor='k')
    plt.xlabel('Silhouette')
    plt.axvline(x=0.26)
    plt.show()
        

# == HIERARCHICAL ================================================================

if hierarchical_validation:
    silhouette_list=list()
    for time in range(validation_times):
        PercentSalaryHike = random_attributes('PercentSalaryHike', 3.316, 5.0)
        FractionYearsAtCompany = random_attributes('FractionYearsAtCompany',0.0,1.0)
        TrainingTimesLastYear = random_attributes('TrainingTimesLastYear', 0.0, 6.0)
        RateIncome = random_attributes('RateIncome', 0.044, 0.997)
        NumCompaniesWorked = random_attributes('NumCompaniesWorked', 0.0, 3.0)
        Age = random_attributes('Age', 18, 60, integer=True)
        StockOptionLevel = random_attributes('StockOptionLevel', 0, 3 , integer=True)
        MonthlyRate = random_attributes('MonthlyRate', 45.9, 164.4)
        MonthlyIncome = random_attributes('MonthlyIncome', 6.915, 9.905)
        DistanceFromHome = random_attributes('DistanceFromHome', 1, 5.38)

        df = random_dataframe([PercentSalaryHike,Age,MonthlyIncome,TrainingTimesLastYear])

        scaler = MinMaxScaler()
        X = scaler.fit_transform(df)

        n_clu=2
        meth='average'
        data_dist = pdist(X, metric='euclidean')
        data_link = linkage(data_dist, method=meth, metric='euclidean', optimal_ordering = True)
        fclu = fcluster(data_link, n_clu, criterion="maxclust")
        # res = dendrogram(data_link, color_threshold=7.8)
        if len(np.unique(fclu))>1:   
            sil = silhouette_score(X, fclu, metric='euclidean')
            print(f'{time+1}) silhouette: {sil}')
            silhouette_list.append(sil)
    
    silhouette_list=np.array(silhouette_list)
    with open("validation_hierarchical_1.txt",'w', encoding='utf-8') as f:
        f.write(f'DATA FRAME\n')
        f.write(f'{df.columns}\n')
        f.write(f'\n')
        f.write(f'number of values for each attributes: {num_records}\n')
        f.write(f'number of random data frames: {validation_times}\n')
        f.write(f'\n')
        f.write(f'number of clusters: {n_clu}\n')
        f.write(f'method: {meth}\n')
        f.write(f'________________________________________________________________________\n')
        f.write(f'\n')
        f.write(f'STATISTIC\n')
        f.write(f'mean (silhouette): {silhouette_list.mean()}\n')
        f.write(f'std (silouette): {silhouette_list.std()}\n')

    plt.hist(silhouette_list, bins=int((np.log2(num_records)+1)), edgecolor='k')
    plt.xlabel('Silhouette')
    plt.axvline(x=0.375)
    plt.show()
