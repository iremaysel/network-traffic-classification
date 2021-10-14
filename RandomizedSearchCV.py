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

# K-Nearest Neighbors
print("----------")
print('KNN')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
knn= KNeighborsClassifier()
grid = {"n_neighbors":np.arange(1,10), "metric":["manhattan","euclidean","chebyshev","minkowski"]}
rand = RandomizedSearchCV(knn, grid, cv=5)
rand.fit(X_train,y_train)
print("Best score : ",rand.best_score_)
print("Best Parametre: ",rand.best_params_)
print("Estimators: ",rand.best_estimator_)

# Decision Tree
print("----------")
print('DTC: ')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
grid = {'criterion':['entropy']}
dt = DecisionTreeClassifier()
rand = RandomizedSearchCV(dt, grid, cv = 5)
rand.fit(X_train, y_train)
print("Best score : ",rand.best_score_)
print("Best Parametre: ",rand.best_params_)
print("Estimators: ",rand.best_estimator_)

# Random Forest
print("----------")
print('RFC: ')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
grid = {'n_estimators': [1400, 1600, 1800, 2000],
        'criterion' : ['entropy','gini']}
rfc = RandomForestClassifier()
rand = RandomizedSearchCV(rfc, grid, cv = 5)
rand.fit(X_train, y_train)
print("Best score : ",rand.best_score_)
print("Best Parametre: ",rand.best_params_)
print("Estimators: ",rand.best_estimator_)

# XGBoost
print("----------")
print('XGBoost: ')
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
grid = {'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb = XGBClassifier()
rand = RandomizedSearchCV(xgb, grid, cv = 5)
rand.fit(X_train, y_train)
print("Best score : ",rand.best_score_)
print("Best Parametre: ",rand.best_params_)
print("Estimators: ",rand.best_estimator_)

# Support Vector Machine
print("----------")
print('SVM: ')
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
grid = {'kernel': ['rbf']}  
svm = SVC()
rand = RandomizedSearchCV(svm, grid, cv = 5)
rand.fit(X_train, y_train)
print("Best score : ",rand.best_score_)
print("Best Parametre: ",rand.best_params_)
print("Estimators: ",rand.best_estimator_)

# Logistic Regression
print("----------")
print('Logistic Regression')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} # l1 = lasso ve l2 = ridge
logreg = LogisticRegression()
rand = RandomizedSearchCV(logreg, grid, cv = 5)
rand.fit(X_train, y_train)
print("Best score : ",rand.best_score_)
print("Best Parametre: ",rand.best_params_)
print("Estimators: ",rand.best_estimator_)

