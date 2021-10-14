import numpy as np
import pandas as pd
import warnings
## Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
## Sklearn Libraries
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, classification_report, recall_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

df_f = pd.read_csv("dataset/balanced-data.csv")
df_x = df_f.iloc[:,0:27]
df_f['application_protocol'] = pd.Categorical(df_f['application_protocol'])
application_protocol_data = pd.get_dummies(df_f['application_protocol'])
df_x = pd.concat([df_x, application_protocol_data], axis=1)

df_f['category'] = pd.Categorical(df_f['category'])
category_data = pd.get_dummies(df_f['category'])
df_x = pd.concat([df_x, category_data], axis=1)

le = LabelEncoder()
web_service_data = le.fit_transform(df_f['web_service'])
web_service_data = pd.DataFrame(data = web_service_data, index = range(390000), columns= ["application"])
df = pd.concat([df_x, web_service_data], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,[0,1,2,3,6,8,9,10,11,12,15,23,36,37,38,40,41,42,43,45]], df.iloc[:,46:47], test_size = 0.33, random_state = 0)

# scaling of data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.model_selection import cross_val_score

# K-Nearest Neighbors
print("----------")
print('KNN (Average Values)')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
           metric_params=None, n_jobs=None, n_neighbors=1, p=2,
           weights='uniform')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracies = cross_val_score(estimator = knn, X = X_train, y= y_train, cv = 5)
print("Deviasyon: ",np.std(accuracies))
print("Accuracy:(MAX)",max(accuracies))
print("Accuracy:",np.mean(accuracies))
print("F1-score:", np.mean(f1_score(y_test, y_pred, average = 'weighted')))
print("Recall:", np.mean(recall_score(y_test, y_pred, average = 'weighted')))
print("Precision:", np.mean(precision_score(y_test, y_pred, average = 'weighted')))
print("Error Rate:", np.mean(1 - accuracies))

# Logistic Regression
print("----------")
print('Logistic Regression (Average Values)')
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(C=1000.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=None,
          solver='warn', tol=0.0001, verbose=0, warm_start=False)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)
accuracies = cross_val_score(estimator = logr, X = X_train, y= y_train, cv = 5)
print("Deviasyon: ",np.std(accuracies))
print("Accuracy:(MAX)",max(accuracies))
print("Accuracy:",np.mean(accuracies))
print("F1-score:", np.mean(f1_score(y_test, y_pred, average = 'weighted')))
print("Recall:", np.mean(recall_score(y_test, y_pred, average = 'weighted')))
print("Precision:", np.mean(precision_score(y_test, y_pred, average = 'weighted')))
print("Error Rate:", np.mean(1 - accuracies))

# Support Vector Machine
print("----------")
print("SVM (Average Values)")
from sklearn.svm import SVC
svc = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
accuracies = cross_val_score(estimator = svc, X = X_train, y= y_train, cv = 5)
print("Deviasyon: ",np.std(accuracies))
print("Accuracy:(MAX)",max(accuracies))
print("Accuracy:",np.mean(accuracies))
print("F1-score:", np.mean(f1_score(y_test, y_pred, average = 'weighted')))
print("Recall:", np.mean(recall_score(y_test, y_pred, average = 'weighted')))
print("Precision:", np.mean(precision_score(y_test, y_pred, average = 'weighted')))
print("Error Rate:", np.mean(1 - accuracies))

# Gaussian Naive Bayes
print("----------")
print('GNB (Average Values)')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracies = cross_val_score(estimator = gnb, X = X_train, y= y_train, cv = 5)
print("Deviasyon: ",np.std(accuracies))
print("Accuracy:(MAX)",max(accuracies))
print("Accuracy:",np.mean(accuracies))
print("F1-score:", np.mean(f1_score(y_test, y_pred, average = 'weighted')))
print("Recall:", np.mean(recall_score(y_test, y_pred, average = 'weighted')))
print("Precision:", np.mean(precision_score(y_test, y_pred, average = 'weighted')))
print("Error Rate:", np.mean(1 - accuracies))

# Decision Tree
print("----------")
print('DTC (Average Values)')
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=9,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            splitter='best')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
accuracies = cross_val_score(estimator = dtc, X = X_train, y= y_train, cv = 5)
print("Deviasyon: ",np.std(accuracies))
print("Accuracy:(MAX)",max(accuracies))
print("Accuracy:",np.mean(accuracies))
print("F1-score:", np.mean(f1_score(y_test, y_pred, average = 'weighted')))
print("Recall:", np.mean(recall_score(y_test, y_pred, average = 'weighted')))
print("Precision:", np.mean(precision_score(y_test, y_pred, average = 'weighted')))
print("Error Rate:", np.mean(1 - accuracies))

# Random Forest
print("----------")
print('RFC (Average Values)')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=2000, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
accuracies = cross_val_score(estimator = rfc, X = X_train, y= y_train, cv = 10)
print("Deviasyon: ",np.std(accuracies))
print("Accuracy:(MAX)",max(accuracies))
print("Accuracy:",np.mean(accuracies))
print("F1-score:", np.mean(f1_score(y_test, y_pred, average = 'weighted')))
print("Recall:", np.mean(recall_score(y_test, y_pred, average = 'weighted')))
print("Precision:", np.mean(precision_score(y_test, y_pred, average = 'weighted')))
print("Error Rate:", np.mean(1 - accuracies))

# XGBoost
print("----------")
print('XGBoost')
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=0.8,
              tree_method='exact', validate_parameters=1, verbosity=None)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
accuracies = cross_val_score(estimator = xgb, X = X_train, y= y_train, cv = 5)
print("Deviasyon: ",np.std(accuracies))
print("Accuracy:(MAX)",max(accuracies))
print("Accuracy: ",np.mean(accuracies))
print("F1-score:", np.mean(f1_score(y_test, y_pred, average = 'weighted')))
print("Recall:", np.mean(recall_score(y_test, y_pred, average = 'weighted')))
print("Precision:", np.mean(precision_score(y_test, y_pred, average = 'weighted')))
print("Error Rate:", np.mean(1 - accuracies))
