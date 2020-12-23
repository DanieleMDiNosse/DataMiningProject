import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

'''The objective is to classify the records based on the Attrition. In other words, Attrition will be our class 
#target y while all the other attributes will be the vector x'''

df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/TrasfAttributeFraction_RateIncome.csv',index_col = 0)

'''Before everything we must preprocess the data. We need to convert string attributes into some numerical attributes
in order to make the algorithm works. For the features variable (X matrix) we must use OneHotEncoder that returns a 
sparse matrix. Then, for y label, we must use LabelEncoder if the labels are greater than 2 or LabelBinarizer if
y is binary, like Yes or No (that's the case for Attrition).
                             
Anyway, teacher suggested not to use categorical attributes in the decision tree classification, except for those one that
has a small number of possibile values, like Gender'''


          

    
'''The first thing to do is to divide the dataset into train and test. Be aware that the training set has to be
further splitted into an actual training set and a validation set used for estimating the generalization error.
Random_state specify the specific random splitting, while stratify keep the distribution of the target class'''

# attributes = [col for col in df.columns if col != 'Attrition']
# X = df[attributes].values
# y = df['Attrition'] # Target class

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100, stratify = y)

    
'''Build the decision tree.
min_samples_split is the minimun number of samples required by a node to be splitted, while min_samples_leaf
is the minimun number of samples required to consider a node a leaf node. There's also another important parameter,
min_impurity_decrease that is an early-stopping createrion for the tree. Growing is stopped if the decrease of impurity 
is less that the number set'''

# clf = DecisionTreeClassifier(criterion='gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1)
# clf.fit(X_train, y_train)