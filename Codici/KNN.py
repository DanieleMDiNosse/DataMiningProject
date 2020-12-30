import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
start = time.time()


grid_search_cv = False
model_tuning = True

df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv', index_col = 0) 
# df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv',index_col = 0)
# df = pd.read_csv('C:/Users/lasal/Desktop/UNIPI notes/Data Mining/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv',index_col = 0)

df.replace({'Gender':{'Female' : 0., 'Male' : 1.}}, inplace = True)
df.replace({'OverTime':{'No' : 0., 'Yes' : 1.}}, inplace = True)
df.replace({'EnvironmentSatisfaction':{'Low': 0, 'Medium': 0, 'High': 1, 'Very High': 1}}, inplace = True)
df.replace({'WorkLifeBalance':{'Bad': 0, 'Good': 0, 'Better': 1, 'Best': 1}}, inplace = True)
df.replace({'JobInvolvement':{'Low': 0, 'Medium': 0, 'High': 1, 'Very High': 1}}, inplace = True)
df.replace({'MaritalStatus':{'Single':0, 'Married': 1, 'Divorced': 0}}, inplace = True)

def oversampling(df, attribute, value, n_duplicate):
    for v in df[attribute].unique():
        if v == value:
            indici = np.array(df[df[attribute] == v].index)
    for i in range(n_duplicate):
        nrand = int(random.uniform(0,len(indici)))
        df = df.append(df.iloc[nrand])
    return df     

print(df.shape)
df = oversampling(df,'Attrition', 'Yes',700)
print(df.shape)
            

numeric = df.select_dtypes('number')
scaler = MinMaxScaler()
X = scaler.fit_transform(numeric)

y = df['Attrition'] # Target class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100, stratify = y)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)

scores = cross_val_score(neigh, X_train, y_train, cv=5)
print('Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

scores = cross_val_score(neigh, X_train, y_train, cv=5, scoring='f1_macro')
print('F1-score: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

if grid_search_cv:
    K = {'n_neighbors':list(range(1,150))}
    scores = ['precision', 'recall']
    
    best_par_list=list()
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(KNeighborsClassifier(weights='distance'), K, scoring='%s_macro' % score, cv=3)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
    
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        best_par_list.append(clf.best_params_)
    
    print(best_par_list)

if model_tuning:
    XX_train, XX_val, yy_train, yy_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 100, stratify = y_train)
    for n in range(1,30):
        
        neigh = KNeighborsClassifier(n_neighbors=n, weights='distance')
        neigh.fit(XX_train, yy_val)
        y_true, y_pred = yy_val, neigh.predict(XX_val)
        precision = precision_score(y_true, y_pred)
        recall =  recall_score(y_true, y_pred)
        if precision > 0.75:
            if recall > 0.55:
                print(f'Precision, Recall, N neighbors: {precision}, {recall}. {n}')
        
        # print(precision_score(y_true, y_pred))
        # print(recall_score(y_true, y_pred))
        # print(' Confusion Matrix Test \n', confusion_matrix(y_true, y_pred))





end = time.time()
elaps = (end-start)/60
print('Elapsed time: %.2f minutes' %elaps)
