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




#== PARAMETERS ===============================================================================
num_records=899         #numer of records for each attributes (see random_attributes)
validation_times=500    #numer of random data frame (see ciclo for)
#============================================================================================

#== IF CHECK =================================================================================
kmean_validation= True
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
# 

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
        print(time, sse)
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
