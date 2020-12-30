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

# df = pd.read_csv('/home/danielemdn/Documenti/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv', index_col = 0) 
# df = pd.read_csv('C:/Users/raffy/Desktop/temp/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv',index_col = 0)
df = pd.read_csv('C:/Users/lasal/Desktop/UNIPI notes/Data Mining/DataMiningProject/Excel/DataFrameWMWO_Reversed.csv',index_col = 0)


categorical = df.select_dtypes(exclude = 'number')

# == IF CKECKS ==============================================================================================================
impurity_decrese_if = False
max_depth_if = False
roc_curve_if = True
cross_validation_if = False
# ===========================================================================================================================

# == PREPROCESSING OF DATA - COVERTS INTO NUMERICAL =========================================================================
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
df.replace({'EnvironmentSatisfaction':{'Low': 0, 'Medium': 0, 'High': 1, 'Very High': 1}}, inplace = True)
df.replace({'WorkLifeBalance':{'Bad': 0, 'Good': 0, 'Better': 1, 'Best': 1}}, inplace = True)
df.replace({'JobInvolvement':{'Low': 0, 'Medium': 0, 'High': 1, 'Very High': 1}}, inplace = True)
df.replace({'MaritalStatus':{'Single':0, 'Married': 1, 'Divorced': 0}}, inplace = True)

numeric = df.select_dtypes('number')
# ==========================================================================================================================


# == DIVIDE DATA SET INTO TRAINING SET (train-validation) AND TESTING SET ==================================================
'''The first thing to do is to divide the dataset into train and test. Be aware that the training set has to be
further splitted into an actual training set and a validation set used for estimating the generalization error.
Random_state specify the specific random splitting, while stratify keep the distribution of the target class label'''

attributes = [col for col in numeric.columns if col != 'Attrition']
X = df[attributes].values
y = df['Attrition'] # Target class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100, stratify = y)

# ==========================================================================================================================

# == BUILD DECISION TREE ===================================================================================================
'''Build the decision tree.
min_samples_split is the minimun number of samples required by a node to be splitted, while min_samples_leaf
is the minimun number of samples required to consider a node a leaf node. There's also another important parameter,
min_impurity_decrease that is an early-stopping createrion for the tree. Growing is stopped if the decrease of impurity 
is less that the number set'''

# clf = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 7, min_samples_leaf = 2, min_impurity_decrease=0.0, class_weight={'Yes':2})  #c=4 recall
# clf1 = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 4, min_samples_leaf = 2, min_impurity_decrease=0.0, class_weight={'Yes':3}) #c=5 recall
# clf2 = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 4, min_samples_leaf = 1, min_impurity_decrease=0.02, class_weight={'Yes':2}) #c=4 precision
# clf3 = DecisionTreeClassifier(criterion='gini', max_depth = 6, min_samples_split = 2, min_samples_leaf = 2, min_impurity_decrease=0.007, class_weight={'Yes':3}) #c=5 recall raff


clf = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 4, min_samples_leaf = 1, min_impurity_decrease=0.02, class_weight={'Yes':2}) #c=4 precision
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
graph.write_pdf("Modello_4_raff1.pdf")

'''In scikit-learn, we implement the importance as described in
(often cited, but unfortunately rarely read…). It is sometimes called “gini importance” 
or “mean decrease impurity” and is defined as the total decrease in node impurity 
(weighted by the probability of reaching that node (which is approximated by the proportion of samples
 reaching that node)) averaged over all trees of the ensemble.'''
print('')
print('ATTRIBUTES IMPORTANCE\n')
for col, imp in zip(attributes, clf.feature_importances_):
    print(col, imp)
print('_________________________________________________________________')

'''Performance
Accuracy = Number of correct predictions / Total Number of predictions
Error Rate = Number of wrong predictions / Total Number of predictions
F1-Score = TP / (TP + 0.5*(FP + FN)) or the same with TN oppure F1 = 2 * (precision * recall) / (precision + recall) : best value at 1 and worst score at 0
Recall = TN / (TN+FP)
Precision = TP / (TP+FP)
'''
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
    tuned_parameters = [{'max_depth': list(range(4,11)), 'min_samples_split': list(range(3,10)), 'min_samples_leaf': list(range(1,8)), 'min_impurity_decrease': list(np.linspace(0.0,0.07,7)), 'class_weight':[{'Yes': i, 'No': j} for i,j in zip(np.linspace(1.5,3.5,7), np.linspace(0.2,1.5,7))]}]

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
    clf = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 7, min_samples_leaf = 2, min_impurity_decrease=0.0, class_weight={'Yes':2})  #c=4 recall
    clf1 = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 4, min_samples_leaf = 2, min_impurity_decrease=0.0, class_weight={'Yes':3}) #c=5 recall
    clf2 = DecisionTreeClassifier(criterion='gini', max_depth = 4, min_samples_split = 4, min_samples_leaf = 1, min_impurity_decrease=0.02, class_weight={'Yes':2}) #c=4 precision
    clf3 = DecisionTreeClassifier(criterion='gini', max_depth = 6, min_samples_split = 2, min_samples_leaf = 2, min_impurity_decrease=0.007, class_weight={'Yes':3}) #c=5 recall raff

    XX_train, XX_val, yy_train, yy_val = train_test_split(X_train, y_train, test_size = 0.4, random_state = 100, stratify = y_train)
    
    clf.fit(XX_train, yy_train)
    clf1.fit(XX_train, yy_train)
    clf2.fit(XX_train, yy_train)
    clf3.fit(XX_train, yy_train)

    # In ogni caso, dalla procedura per costruire la ROC curve, mi sembra di dover calcolare
    # queste probabilità ed usare queste al posto di y_pred. Il modulo predict_proba 
    # restituisce la probabilità predetta per la classe di ogni istanza. Tale probabilità
    # è la frazione dei valori degli attributi aventi la stessa classe in un leaf node
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