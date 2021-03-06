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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import time
import pydotplus
from sklearn import tree
from IPython.display import Image

start = time.time()
'''The objective is to classify the records based on the Attrition. In other words, Attrition will be our class 
target y while all the other attributes will be the vector x'''

# df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/TRANSFknn_plus150_attriction_yes90.csv') 
df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/decisiontree_plus100_attriction_yes.csv')
# df = pd.read_csv('C:/Users/lasal/Desktop/UNIPI notes/Data Mining/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv',index_col = 0)


print(df['Attrition'].value_counts())

categorical = df.select_dtypes(exclude = 'number')

# == IF CKECKS ==============================================================================================================
impurity_decrese_if = False
max_depth_if = False
roc_curve_if = False
cross_validation_if = False

numeric = df.select_dtypes('number')


attributes = [col for col in numeric.columns if col != 'Attrition']
X = numeric.values
y = df['Attrition'] # Target class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 100, stratify = y)

# [{'max_depth': 7, 'min_impurity_decrease': 0.0024242424242424242, 'min_samples_leaf': 1, 'min_samples_split': 5} precision
# {'max_depth': 7, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 3}] recall cv=3

# clf = DecisionTreeClassifier( criterion='gini', max_depth = 8, min_samples_split = 5, min_samples_leaf = 2, min_impurity_decrease= 0.0)
# clf = DecisionTreeClassifier(criterion='gini', max_depth = 3, min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease=0.0)
# clf = DecisionTreeClassifier(criterion='gini', max_depth = 12, min_samples_split = 5, min_samples_leaf = 1, min_impurity_decrease=0.001) 
# clf = DecisionTreeClassifier(criterion='gini', max_depth = 6, min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease=0.0033)

# clf = DecisionTreeClassifier(criterion='gini', max_depth = 7, min_samples_split = 3, min_samples_leaf = 1, min_impurity_decrease= 0.003)
clf.fit(X_train, y_train)
print('')
print( 'BASIC DESCRIPTION\n')
print(f'Parameters: see DecisionTreeClassifier for more details ')
# print(f'1. criterion = {criterion}')
print(f'number of nodes: {clf.tree_.node_count}')
print('_________________________________________________________________')

y_pred = clf.predict(X_test)
y_pred_tr = clf.predict(X_train)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=attributes, class_names=clf.classes_, filled=True, rounded=True, special_characters=True, impurity=True)  
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_pdf("1.pdf")

print('')
print( 'PERFORMANCE\n')
print('Train Accuracy %s' % accuracy_score(y_train, y_pred_tr)) # Accuratezza basata sul training set (Indica quanto il modello ha imparato bene dal training set)
print('Train F1-score %s' % f1_score(y_train, y_pred_tr, average=None))

print('Test Accuracy %s' % accuracy_score(y_test, y_pred)) # Accuratezza sul test set
print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))
print('')
print('                   Classification Report for Test \n ', classification_report(y_test, y_pred))


print('                   Classification Report for Train \n ', classification_report(y_train, y_pred_tr))

print(' Confusion Matrix Test \n', confusion_matrix(y_test, y_pred))
print('')
print(' Confusion Matrix Train \n', confusion_matrix(y_train, y_pred_tr))

print('-----------------------------------------------------')
# Confidence Interval

N = len(X_test)
acc = accuracy_score(y_test, y_pred)
Z_a = {'99%':2.58, '98%':2.33, '95%':1.96, '90%':1.65, '80%':1.28, '70%':1.04, '50%':0.67}
confidence_interval_low = (2*N*acc + Z_a['95%']**2 - Z_a['95%']*np.sqrt(Z_a['95%']**2 + 4*N*acc - 4*N*acc**2))/(2*(N + Z_a['95%']**2))
confidence_interval_high = (2*N*acc + Z_a['95%']**2 + Z_a['95%']*np.sqrt(Z_a['95%']**2 + 4*N*acc - 4*N*acc**2))/(2*(N + Z_a['95%']**2))
print('Confidence Interval: %.2f -- %.2f' %(confidence_interval_low, confidence_interval_high), '\nwhile accuracy is %.2f, so' %acc)
if (acc > confidence_interval_low) and (acc < confidence_interval_high):
    print('the empirical accuracy is statistically acceptable')
print('_________________________________________________________________')

# == OVERFITTINFIG AND UNDERFITTING  ===============================================================================================================================
if impurity_decrese_if:
    ER_train = list()
    ER_test = list()
    nodes_list = list()
    for depth in range(1,50):
        clf = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 3, min_samples_leaf = 1, min_impurity_decrease=0.01, class_weight={'Yes':1.78, 'No':0.38}) #c=5 recall raff
        clf.fit(X_train, y_train)

        nodes=clf.tree_.node_count
        

        y_pred = clf.predict(X_test)
        y_pred_tr = clf.predict(X_train)

        tmp_test = (confusion_matrix(y_test, y_pred)[0][1] + confusion_matrix(y_test, y_pred)[1][0]) / (np.sum(confusion_matrix(y_test, y_pred))) 
        tmp_train = (confusion_matrix(y_train, y_pred_tr)[0][1] + confusion_matrix(y_train, y_pred_tr)[1][0]) / (np.sum(confusion_matrix(y_train, y_pred_tr)))
        print(nodes,nodes)
        ER_train.append(tmp_train)
        ER_test.append(tmp_test)
        nodes_list.append(nodes)

    plt.figure('Impurity decrese Overfitting')
    plt.plot(nodes_list, ER_train, label = 'Error Rate Train')
    plt.plot(nodes_list, ER_test, label = 'Error Rate Test')
    plt.legend()
    plt.grid(True)

if max_depth_if:
    ER_train = list()
    ER_test = list()
    nodes_list = list()

    max_val_depth=50
    for depth in range(1, max_val_depth):
        clf = DecisionTreeClassifier(criterion='gini', max_depth = depth, min_samples_split = 4, min_samples_leaf = 1, min_impurity_decrease=0.02, class_weight={'Yes': 2})
        clf.fit(X_train, y_train)

        nodes=clf.tree_.node_count
        
        y_pred = clf.predict(X_test)
        y_pred_tr = clf.predict(X_train)

        tmp_test = (confusion_matrix(y_test, y_pred)[0][1] + confusion_matrix(y_test, y_pred)[1][0]) / (np.sum(confusion_matrix(y_test, y_pred))) 
        tmp_train = (confusion_matrix(y_train, y_pred_tr)[0][1] + confusion_matrix(y_train, y_pred_tr)[1][0]) / (np.sum(confusion_matrix(y_train, y_pred_tr)))
        print(depth,nodes)
        ER_train.append(tmp_train)
        ER_test.append(tmp_test)
        nodes_list.append(nodes)

    plt.figure('Max depth Overfitting')
    plt.plot(nodes_list, ER_train, label = 'Error Rate Train')
    plt.plot(nodes_list, ER_test, label = 'Error Rate Test')
    plt.legend()
    plt.grid(True)
    
plt.show()
# =================================================================================================================================================

# CROSS - VAIDATION ============================================================================================================================================
if cross_validation_if:
    tuned_parameters = [{'max_depth': list(range(2,8)), 'min_samples_split': list(range(2,6)), 'min_samples_leaf': list(range(1,6)), 'min_impurity_decrease': list(np.linspace(0.0,0.02,100))}]

    scores = ['precision', 'recall']

    best_par_list=list()
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, scoring='%s_macro' % score, cv=3)
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

# ======================================================================================================================================================

# == ROC CURVE =====================================================================================================================================
'''Compute the ROC (Receiver Operating Characteristic) curve. ROC curve rapresents in a
graphical fashion the performance of different classifiers varying the thresholds by which
the model assigns the class labels. FPR on x axis and TPR on y axis. To compute the ROC curve
you need the probabilities of the class label predictions of the various istances, sort these
and then draw a point in the (FPR,TPR)-plane for different thresholds.'''
if roc_curve_if:

# [{'max_depth': 7, 'min_impurity_decrease': 0.0024242424242424242, 'min_samples_leaf': 1, 'min_samples_split': 5} precision
# {'max_depth': 7, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 3}] recall cv=3
    clf = DecisionTreeClassifier( criterion='gini', max_depth = 8, min_samples_split = 5, min_samples_leaf = 2, min_impurity_decrease= 0.0)
    clf1 = DecisionTreeClassifier(criterion='gini', max_depth = 3, min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease=0.0)
    clf2 = DecisionTreeClassifier(criterion='gini', max_depth = 12, min_samples_split = 3, min_samples_leaf = 1, min_impurity_decrease=0.0) 
    clf3 = DecisionTreeClassifier(criterion='gini', max_depth = 6, min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease=0.0033)

    XX_train, XX_val, yy_train, yy_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 100, stratify = y_train)
    
    clf.fit(XX_train, yy_train)
    clf1.fit(XX_train, yy_train)
    clf2.fit(XX_train, yy_train)
    clf3.fit(XX_train, yy_train)

    predictions = clf.predict_proba(XX_val, check_input = True)
    predictions1 = clf1.predict_proba(XX_val, check_input = True)
    predictions2 = clf2.predict_proba(XX_val, check_input = True)
    predictions3 = clf3.predict_proba(XX_val, check_input = True)

    fpr, tpr, thresholds = roc_curve(yy_val, predictions[:,1], pos_label = 'Yes')
    fpr1, tpr1, thresholds1 = roc_curve(yy_val, predictions1[:,1], pos_label = 'Yes')
    fpr2, tpr2, thresholds2 = roc_curve(yy_val, predictions2[:,1], pos_label = 'Yes')
    fpr3, tpr3, thresholds3 = roc_curve(yy_val, predictions3[:,1], pos_label = 'Yes')
    
    roc_auc = roc_auc_score(yy_val, predictions[:,1], average=None)
    roc_auc1 = roc_auc_score(yy_val, predictions1[:,1], average=None)
    roc_auc2 = roc_auc_score(yy_val, predictions2[:,1], average=None)
    roc_auc3 = roc_auc_score(yy_val, predictions3[:,1], average=None)
    print(roc_auc, roc_auc1, roc_auc2, roc_auc3)

    plt.figure('ROC CURVE')
    plt.plot(fpr, tpr, label='Modello 1 (area = %0.2f)' % (roc_auc))
    plt.plot(fpr1, tpr1, label='Modello 2 (area = %0.2f)' % (roc_auc1))
    plt.plot(fpr2, tpr2, label='Modello 3  (area = %0.2f)' % (roc_auc2))
    plt.plot(fpr3, tpr3, label='Modello 4  (area = %0.2f)' % (roc_auc3))
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate',)
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.show()
# ==================================================================================================================================================

end = time.time()
elaps = (end-start)/60
print('Elapsed time: %.2f minutes' %elaps)