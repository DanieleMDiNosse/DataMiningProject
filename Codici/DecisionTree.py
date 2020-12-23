import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv',index_col = 0)
# df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv',index_col = 0)
categorical = df.select_dtypes(exclude = 'number')

'''Before everything we must preprocess the data. We need to convert string attributes into some numerical attributes
in order to make the algorithm works. For the features variable (X matrix) we must use OneHotEncoder that returns a 
sparse matrix. Then, for y label, we must use LabelEncoder if the labels are greater than 2 or LabelBinarizer if
y is binary, like Yes or No (that's the case for Attrition).
                             
Anyway, teacher suggested not to use categorical attributes in the decision tree classification, except for those one that
has a small number of possibile values, like Gender'''

ohe = OneHotEncoder(sparse = False)

gender_ohe = ohe.fit_transform(df['Gender'].values.reshape(-1,1)) # 0 = Female, 1 = Male
overtime_ohe = ohe.fit_transform(df['OverTime'].values.reshape(-1,1)) # 0 = No, 1 = Yes
df.replace({'Gender':{'Female' : 0., 'Male' : 1.}}, inplace = True)
df.replace({'OverTime':{'No' : 0., 'Yes' : 1.}}, inplace = True)
# df.replace({'Attrition':{'No' : 0., 'Yes' : 1.}}, inplace = True)
numeric = df.select_dtypes('number')

'''The first thing to do is to divide the dataset into train and test. Be aware that the training set has to be
further splitted into an actual training set and a validation set used for estimating the generalization error.
Random_state specify the specific random splitting, while stratify keep the distribution of the target class'''

attributes = [col for col in numeric.columns if col != 'Attrition']
X = df[attributes].values
y = df['Attrition'] # Target class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100, stratify = y)

    
'''Build the decision tree.
min_samples_split is the minimun number of samples required by a node to be splitted, while min_samples_leaf
is the minimun number of samples required to consider a node a leaf node. There's also another important parameter,
min_impurity_decrease that is an early-stopping createrion for the tree. Growing is stopped if the decrease of impurity 
is less that the number set'''

clf = DecisionTreeClassifier(criterion='gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease = 0.0005)
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
F1-Score = TP / (TP + 0.5*(FP + FN))  oppure F1 = 2 * (precision * recall) / (precision + recall)
best value at 1 and worst score at 0
'''

print('Train Accuracy %s' % accuracy_score(y_train, y_pred_tr))
print('Train F1-score %s' % f1_score(y_train, y_pred_tr, average=None))

print('Test Accuracy %s' % accuracy_score(y_test, y_pred))
print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))
# print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))