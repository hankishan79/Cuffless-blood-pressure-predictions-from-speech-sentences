import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.svm import LinearSVC
# import SVC classifier
from sklearn.svm import SVC
from sklearn import svm
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score # import KFold
import numpy as np
import sklearn_relief as sr
from sklearn.metrics import roc_auc_score, roc_curve
# Import the library
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# loading train and test data
data = pd.read_excel(r'C:\Users\hankishan\Desktop\S_H_B_Database_10132022\pyhton codes\features_train.xlsx')
data.head()
X = data.drop(columns=['DBP-2CLASSES','SBP-2CLASSES','MIXED-CLASSES', 'SBP', 'DBP','WEIGHT','HEIGHT'])
y=data['SBP-2CLASSES']
data_test = pd.read_excel(r'C:\Users\hankishan\Desktop\S_H_B_Database_10132022\pyhton codes\features_test.xlsx')
data_test.head()
X_test_other = data_test.drop(columns=['DBP-2CLASSES', 'SBP-2CLASSES','MIXED-CLASSES','SBP', 'DBP', 'WEIGHT', 'HEIGHT'])
y_test_other = data_test['SBP-2CLASSES']

X_test_other.head()
# Z-Score for other test data *********************************************************************************
X_test_other = stats.zscore(X_test_other,axis=1)
X_test_other=X_test_other.drop(['PAT_ID'], axis=1)

# Shuffle and dividing data as train and test for first stage *******
gs = GroupShuffleSplit(n_splits=2, test_size=.30, random_state=0)
train_ix, test_ix = next(gs.split(X, y, groups=data['PAT_ID']))
X_train = X.loc[train_ix]
y_train = y.loc[train_ix]
X_test = X.loc[test_ix]
y_test = y.loc[test_ix]

# Z-Score for train and test dataset with using same dataset *********************************************
X_train = stats.zscore(X_train,axis=1)
X_test = stats.zscore(X_test,axis=1)

X_train=X_train.drop(['PAT_ID'], axis=1)
X_test=X_test.drop(['PAT_ID'],axis=1)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Without SMOTE Results ************************************************************************************
# SVC Classifier *******************************************************************************************
svc=SVC(kernel='linear')
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred_svm=svc.predict(X_test)

# KNN classifier *********************************************************************************************
snf=KNeighborsClassifier(n_neighbors=300)
snf.fit(X_train, y_train)
y_pred_knn=snf.predict(X_test)

# Naive Bayes Classifier *************************************************************************************
nbc=GaussianNB()
nbc.fit(X_train,y_train)
y_pred_nbc=nbc.predict(X_test)
#LogisticRegression Regression Classifier ********************************************************************
log=LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train,y_train)
# Random forest Classifer ************************************************************************************
rnd=RandomForestClassifier(n_estimators=50).fit(X_train,y_train)
# Voting Classifier ******************************************************************************************
voting=VotingClassifier(estimators=[('DTC', DecisionTreeClassifier()),('LOG',log),('rf',rnd), ('SVC', SVC(gamma ='auto', probability = True))],voting='hard').fit(X_train,y_train)
y_pred_log=log.predict(X_test)
y_pred_rnd=rnd.predict(X_test)
y_pred_voting=voting.predict(X_test)
#print(confusion_matrix(y_test, y_pred_voting))
#print(classification_report(y_test, y_pred_voting))

print('ISVM Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_svm)))
print('IKNN Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_knn)))
print('INBC Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_nbc)))
print('ILOG Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_log)))
print('IRND Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_rnd)))
print('IVOTING Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_voting)))

# Test data is used other dataset **************************************************************************
y_svm_other = svc.predict(X_test_other)
y_knn_other = snf.predict(X_test_other)
y_nbc_other = nbc.predict(X_test_other)
y_log_other = log.predict(X_test_other)
y_rnd_other = rnd.predict(X_test_other)
y_voting_other = voting.predict(X_test_other)

print('SVM_OTHER Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_svm_other)))
print('KNN_OTHER Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_knn_other)))
print('NBC_OTHER Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_nbc_other)))
print('LOG_OTHER Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_log_other)))
print('RND_OTHER Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_rnd_other)))
print('VOTING_OTHER Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_voting_other)))

#print(classification_report(y_test_other, y_svm_other))
#print(classification_report(y_test_other, y_knn_other))
#print(classification_report(y_test_other, y_nbc_other))
#print(classification_report(y_test_other, y_log_other))
#print(classification_report(y_test_other, y_rnd_other))
#print(classification_report(y_test_other, y_voting_other))

# With SMOTE results *********************************************************************************
from collections import Counter
from imblearn.over_sampling import SMOTE
for l in range(1):
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print (X_train.shape, y_train.shape)
    print (X_res.shape, y_res.shape)
    print (X_test.shape, y_test.shape)
    svc = SVC(kernel='linear')
    svc.fit(X_res,y_res)
    # make predictions on test set
    y_pred_svm=svc.predict(X_test)
    # compute and print accuracy score
    #print(confusion_matrix(y_test, y_pred_svm))
    #print(classification_report(y_test, y_pred_svm))

    # KNN classifier *********************************************************************************************
    snf=KNeighborsClassifier(n_neighbors=10)
    snf.fit(X_res, y_res)
    y_pred_knn=snf.predict(X_test)

    # Naive Bayes Classifier *************************************************************************************
    model=GaussianNB()
    model.fit(X_res,y_res)
    y_pred_nbc=model.predict(X_test)

    # Logistic resgression & Random Forest & Voting  classifiers *********************************
    log = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
    rnd = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    voting = VotingClassifier(estimators=[('rf', rnd), ('SVC', SVC(gamma='auto', probability=True))],
                              voting='hard').fit(X_train, y_train)

    y_pred_log=log.predict(X_test)
    y_pred_rnd=rnd.predict(X_test)
    y_pred_voting=voting.predict(X_test)

    #print(confusion_matrix(y_test, y_pred_voting))
    #print(classification_report(y_test, y_pred_voting))
    print('SMOTE-ISVM Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_svm)))
    print('SMOTE-IKNN Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_knn)))
    print('SMOTE-INBC Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_nbc)))
    print('SMOTE-ILOG Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_log)))
    print('SMOTE-IRND Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_rnd)))
    print('SMOTE-IVOTING Test accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred_voting)))

    #print(classification_report(y_test, y_pred_svm))
    #print(classification_report(y_test, y_pred_knn))
    #print(classification_report(y_test, y_pred_nbc))
    #print(classification_report(y_test, y_pred_log))
    #print(classification_report(y_test, y_pred_rnd))
    #print(classification_report(y_test, y_pred_voting))

#  Test data with SMOTE & other database ******************************************************************************
y_svm_other_SMOTE = svc.predict(X_test_other)
y_knn_other_SMOTE = snf.predict(X_test_other)
y_nbc_other_SMOTE = nbc.predict(X_test_other)
y_log_other_SMOTE = log.predict(X_test_other)
y_rnd_other_SMOTE = rnd.predict(X_test_other)
y_voting_other_SMOTE = voting.predict(X_test_other)

print('SVM_OTHER_SMOTE Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_svm_other_SMOTE)))
print('KNN_OTHER_SMOTE Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_knn_other_SMOTE)))
print('NBC_OTHER_SMOTE Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_nbc_other_SMOTE)))
print('LOG_OTHER_SMOTE Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_log_other_SMOTE)))
print('RND_OTHER_SMOTE Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_rnd_other_SMOTE)))
print('VOTING_OTHER_SMOTE Test accuracy score with default hyperparameters: {0:0.4f}'.format(
    accuracy_score(y_test_other, y_voting_other_SMOTE)))

print(classification_report(y_test_other, y_svm_other_SMOTE))
print(classification_report(y_test_other, y_knn_other_SMOTE))
print(classification_report(y_test_other, y_nbc_other_SMOTE))
print(classification_report(y_test_other, y_log_other_SMOTE))
print(classification_report(y_test_other, y_rnd_other_SMOTE))
print(classification_report(y_test_other, y_voting_other_SMOTE))