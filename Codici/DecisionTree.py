import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus

import pydotplus
from sklearn import tree
from IPython.display import Image

# import os
# os.environ['PATH'] += os.pathsep + 'C:/Users/raffy/Anaconda3/Library/bin/graphviz'
# import os
# os.environ['PATH'] += os.pathsep + 'C:/Users/raffy/.conda/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'

'''The objective is to classify the records based on the Attrition. In other words, Attrition will be our class 
target y while all the other attributes will be the vector x'''

df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv')
# df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv',index_col = 0)
categorical = df.select_dtypes(exclude = 'number')

'''Before everything we must preprocess the data. We need to convert string attributes into some numerical attributes
in order to make the algorithm works. For the features variable (X matrix) we must use OneHotEncoder that returns a 
sparse matrix. Then, for y label, we must use LabelEncoder if the labels are greater than 2 or LabelBinarizer if
y is binary, like Yes or No (that's the case for Attrition).
                             
Anyway, teacher suggested not to use categorical attributes in the decision tree classification, except for those one that
has a small number of possibile values, like Gender'''

# ohe = OneHotEncoder(sparse = False)

# gender_ohe = ohe.fit_transform(df['Gender'].values.reshape(-1,1)) # 0 = Female, 1 = Male
# overtime_ohe = ohe.fit_transform(df['OverTime'].values.reshape(-1,1)) # 0 = No, 1 = Yes

df.replace({'Gender':{'Female' : 0., 'Male' : 1.}}, inplace = True)
df.replace({'OverTime':{'No' : 0., 'Yes' : 1.}}, inplace = True)
# df.replace({'Attrition':{'No' : 0., 'Yes' : 1.}}, inplace = True)
numeric = df.select_dtypes('number')

'''The first thing to do is to divide the dataset into train and test. Be aware that the training set has to be
further splitted into an actual training set and a validation set used for estimating the generalization error.
Random_state specify the specific random splitting, while stratify keep the distribution of the target class label'''

attributes = [col for col in numeric.columns if col != 'Attrition']
X = df[attributes].values
y = df['Attrition'] # Target class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100, stratify = y)

    
'''Build the decision tree.
min_samples_split is the minimun number of samples required by a node to be splitted, while min_samples_leaf
is the minimun number of samples required to consider a node a leaf node. There's also another important parameter,
min_impurity_decrease that is an early-stopping createrion for the tree. Growing is stopped if the decrease of impurity 
is less that the number set'''

clf = DecisionTreeClassifier(criterion='gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease = 0.005, class_weight={'Yes':20}) # Class weight set the weight of the classes during the splitting procedure
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_tr = clf.predict(X_train)

'''In scikit-learn, we implement the importance as described in
(often cited, but unfortunately rarely read…). It is sometimes called “gini importance” 
or “mean decrease impurity” and is defined as the total decrease in node impurity 
(weighted by the probability of reaching that node (which is approximated by the proportion of samples
 reaching that node)) averaged over all trees of the ensemble.'''

# for col, imp in zip(attributes, clf.feature_importances_):
#     print(col, imp)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=attributes, class_names=clf.classes_, filled=True, rounded=True, special_characters=True, impurity=True)  
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_pdf("iris.pdf")

'''Performance
Accuracy = Number of correct predictions / Total Number of predictions
F1-Score = TP / (TP + 0.5*(FP + FN)) or the same with TN oppure F1 = 2 * (precision * recall) / (precision + recall) : best value at 1 and worst score at 0
'''

print('Train Accuracy %s' % accuracy_score(y_train, y_pred_tr)) # Accuratezza basata sul training set (Indica quanto il modello ha imparato bene dal training set)
print('Train F1-score %s' % f1_score(y_train, y_pred_tr, average=None))

print('Test Accuracy %s' % accuracy_score(y_test, y_pred)) # Accuratezza sul test set
print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))

print('                   Classification Report for Test \n ', classification_report(y_test, y_pred))

print('                   Classification Report for Train \n ', classification_report(y_train, y_pred_tr))

print(' Confusion Matrix Test \n', confusion_matrix(y_test, y_pred))

print(' Confusion Matrix Train \n', confusion_matrix(y_train, y_pred_tr))

'''Compute the ROC (Receiver Operating Characteristic) curve. ROC curve rapresents in a
graphical fashion the performance of different classifiers varying the thresholds by which
the model assigns the class labels. FPR on x axis and TPR on y axis. To compute the ROC curve
you need the probabilities of the class label predictions of the various istances, sort these
and then draw a point in the (FPR,TPR)-plane for different thresholds.'''

# Seguendo il codice Titanic, se non converto in valori binari y_pred mi da errore
for i,val in enumerate(y_pred):
    if val == 'Yes':
        y_pred[i] = 0
    else:
        y_pred[i] = 1
# In ogni caso, dalla procedura per costruire la ROC curve, mi sembra di dover calcolare
# queste probabilità ed usare queste al posto di y_pred. Il modulo predict_proba 
# restituisce la probabilità predetta per la classe di ogni istanza. Tale probabilità
# è la frazione dei valori degli attributi aventi la stessa classe in un leaf node
predictions = clf.predict_proba(X_test, check_input = True)
fpr, tpr, thresholds = roc_curve(y_test, predictions[:,1], pos_label = 'Yes')
# fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 'Yes')
roc_auc = auc(fpr, tpr)
print(roc_auc)

roc_auc = roc_auc_score(y_test, predictions[:,1], average=None)
# roc_auc = roc_auc_score(y_test, y_pred, average=None)
print(roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate',)
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", fontsize=14, frameon=False)
plt.show()