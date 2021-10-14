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

from sklearn.model_selection import GridSearchCV

# K-Nearest Neighbors
print("----------")
print('KNN')
from sklearn.neighbors import KNeighborsClassifier
grid = {"n_neighbors":np.arange(1,30), "metric":["manhattan","euclidean","chebyshev","minkowski"]}
knn= KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV
knn_cv.fit(X_train,y_train)
print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_)

# Logistic Regression
print("----------")
print('Logistic Regression')
from sklearn.linear_model import LogisticRegression
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(X_train, y_train)
print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)

# Support Vector Machine
print("----------")
print('SVM: ')
from sklearn.svm import SVC
grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
svm = SVC()
svm_cv = GridSearchCV(svm, grid, refit = True, verbose = 3, cv = 10) 
svm_cv.fit(X_train, y_train) 
print("tuned hyperparameters: (best parameters): ",svm_cv.best_params_) 
print("tuned parametreye gore en iyi accuracy (best score): ",svm_cv.best_estimator_) 

# Gaussian Naive Bayes
print("----------")
print('Naive Bayes: ')
from sklearn.naive_bayes import GaussianNB
grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]}
gnb = GaussianNB()
gnb_cv = GridSearchCV(gnb,grid,cv=10)
gnb_cv.fit(X_train, y_train)
print("tuned hyperparameters: (best parameters): ",gnb_cv.best_params_) 
print("accuracy: ",gnb_cv.best_score_) 

# Random Forest
print("----------")
print('RFC: ')
from sklearn.ensemble import RandomForestClassifier
grid = {'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'criterion' : ['entropy','gini']}
rfc = RandomForestClassifier()
rfc_cv = GridSearchCV(rfc, grid,cv=10)
rfc_cv.fit(X_train, y_train)
print("tuned hyperparameters: (best parameters): ",rfc_cv.best_params_) 
print("accuracy: ",rfc_cv.best_estimator_) 

# Decision Tree
print("----------")
print('DTC: ')
from sklearn.tree import DecisionTreeClassifier

grid = {'criterion':['gini','entropy'],'max_depth': np.arange(1, 10)}

dt = DecisionTreeClassifier()
dt_cv = GridSearchCV(dt, grid, cv = 10)
dt_cv.fit(X_train, y_train)
print("tuned hyperparameters: (best parameters): ",dt_cv.best_params_) 
print("accuracy: ",dt_cv.best_score_) 

# XGBoost
print("----------")
print('XGBoost: ')
from xgboost import XGBClassifier

grid = {'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb = XGBClassifier()
xgb_cv = GridSearchCV(xgb, grid, cv = 10)
xgb_cv.fit(X_train, y_train)
print("tuned hyperparameters: (best parameters): ",xgb_cv.best_params_) 
print("accuracy: ",xgb_cv.best_estimator_) 



