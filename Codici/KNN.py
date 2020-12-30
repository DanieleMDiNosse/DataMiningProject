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

make_dataframe = False
grid_search_cv = False
overfitting_knn = True
model_tuning = False

# df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv', index_col = 0) 
# df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv',index_col = 0)
df = pd.read_csv('C:/Users/lasal/Desktop/UNIPI notes/Data Mining/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv',index_col = 0)

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
        df = df.append(df.iloc[indici[nrand]])
    return df    


if make_dataframe:
    print(df.shape)
    df = oversampling(df,'Attrition', 'Yes',150)
    print(df.shape)

    df.to_csv('knn_plus150_attriction_yes.csv', index=False)


# df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/knn_plus150_attriction_yes.csv', index_col = 0) 
# df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/knn_plus150_attriction_yes.csv',index_col = 0)
# df = pd.read_csv('C:/Users/lasal/Desktop/UNIPI notes/Data Mining/DataMiningProject/Excel/knn_plus150_attriction_yes.csv',index_col = 0)
            

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
    
    par_list=list()
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(KNeighborsClassifier(weights='distance'), K, scoring='%s_macro' % score, cv=3)
        clf.fit(X_train, y_train)
    
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        list_score=list()
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            list_score.append(mean)

        par_list.append(list_score)
   
    plt.figure()
    plt.plot(list(range(1,150)),par_list[0])
    plt.plot(list(range(1,150)),par_list[1])
    plt.show()



if overfitting_knn:
    ER_train = list()
    ER_test = list()
    k_plot=list(range(1,150))
    for k_value in k_plot:
        clf = KNeighborsClassifier(n_neighbors=k_value, weights='distance') #c=5 recall raff
        XX_train, XX_val, yy_train, yy_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 100, stratify = y_train)
        clf.fit(X_train, y_train)


        y_pred = clf.predict(X_test)
        y_pred_tr = clf.predict(X_train)

        tmp_test = (confusion_matrix(y_test, y_pred)[0][1] + confusion_matrix(y_test, y_pred)[1][0]) / (np.sum(confusion_matrix(y_test, y_pred))) 
        tmp_train = (confusion_matrix(y_train, y_pred_tr)[0][1] + confusion_matrix(y_train, y_pred_tr)[1][0]) / (np.sum(confusion_matrix(y_train, y_pred_tr)))
        ER_train.append(tmp_train)
        ER_test.append(tmp_test)

        print(' Confusion Matrix Test \n', confusion_matrix(y_train, y_pred_tr), k_value)
        precision = precision_score( y_train, y_pred_tr, pos_label= 'Yes')
        print(f'Precision, Recall: {precision}')
        print('_____________________________________________________')

    plt.figure('knn decrese Overfitting')
    plt.plot(k_plot, ER_train, label = 'Error Rate Train')
    plt.plot(k_plot, ER_test, label = 'Error Rate Val')
    plt.legend()
    plt.grid(True)
    plt.show()


if model_tuning:   
    neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
    neigh.fit(X_train, y_train)
    y_true, y_pred = y_test, neigh.predict(X_test)
    precision = precision_score( y_true, y_pred, pos_label= 'Yes')
    recall =  recall_score(y_true, y_pred, pos_label= 'Yes')
    
    print(f'Precision, Recall: {precision}, {recall}')
    
    print(' Confusion Matrix Test \n', confusion_matrix(y_true, y_pred))

end = time.time()
elaps = (end-start)/60
print('Elapsed time: %.2f minutes' %elaps)