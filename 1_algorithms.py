import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'winequality-red.csv'
df = pd.read_csv(file_path)
df

X = df.iloc[:, :-1]
X

y = df.iloc[:, -1]
y
'''--------------------------------------------------Linear Regression--------------------------------------------------------------------'''
"""## Linear Regression"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

linear_reg = LinearRegression()
MSE = cross_val_score(linear_reg, X, y, scoring="neg_mean_squared_error",cv=5)
MSE

mean_MSE = np.mean(MSE)
mean_MSE
 
'''-------------------------------------------------Splitting dataset into train and test---------------------------------------------------'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


'''---------------------------------------------------Ridge-------------------------------------------------------------------'''
"""#Ridge"""
'''with hyper parameter tuning'''

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regression = GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error", cv=5 )
ridge_regression.fit(X,y)

ridge_regression.best_params_

ridge_regression.best_score_

ridge_prediction = ridge_regression.predict(X_test)

'''----------------------------------------------------Lasso Regression------------------------------------------------------------------'''
"""# Lasso Regression"""
'''with hyper parameter tuning'''

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regression = GridSearchCV(lasso, parameters, scoring="neg_mean_squared_error", cv=5 )
lasso_regression.fit(X,y)

lasso_regression.best_params_

lasso_regression.best_score_

lasso_prediction = lasso_regression.predict(X_test)

'''------------------------------------------------------Logistic Regression----------------------------------------------------------------'''
"""# Logistic Regression"""
'''with hyper parameter tuning'''

df['quality'].unique()

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

log_reg_model = LogisticRegression()
log_reg_model

parameters = {'C':[0.5, 1,2,3,4,5,6,7,8,9,10,20,30,40,50], 'penalty':['l1','l2','elasticnet']}

logistic_reg = GridSearchCV(log_reg_model, parameters, scoring='accuracy', cv=5)

logistic_reg.fit(X,y)

logistic_reg.best_params_

logistic_reg.best_score_

'''-----------------------------------------------------KNN-----------------------------------------------------------------'''
"""# KNN"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(multi_class = 'ovr', max_iter=200)

logistic_model.fit(X_train, y_train.values.ravel())

logistic_pred = logistic_model.predict(X_test)

logistic_pred

from sklearn.metrics import accuracy_score, classification_report

print("Logistic Regression Accuracy Score is: ", accuracy_score(y_test, logistic_pred))

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train.values.ravel())

knn_pred = knn_model.predict(X_test)

knn_pred

print("KNN Accuracy Score: ", accuracy_score(y_test, knn_pred))

'''-------------------------------------Decision Tree---------------------------------------------------------------------------------'''
"""## Decision Tree

"""

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_pred

print("Decision Tree Accuracy Score: ", accuracy_score(y_test, dt_pred))

'''--------------------------------------------------------SVM--------------------------------------------------------------'''
"""# SVM Kernal"""

from sklearn.svm import SVC, SVR

classifier = SVC(kernel='rbf')
classifier

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

classifier = SVC(kernel='poly')
classifier

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

SVM_regression = SVR(kernel ='rbf')
SVM_regression

SVM_regression.fit(X_train,y_train)

y_pred = SVM_regression.predict(X_test)
y_pred

SVM_regression.fit(X_train, y_train)

y_pred_SVR = SVM_regression.predict(X_test)
y_pred_SVR

'''---------------------------------------------------Random Forest-------------------------------------------------------------------'''
"""# Random Forest"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred)
print(f"Classification: {class_report}")

confusion_matrix(y_test, y_pred)

"""# HyperParameter Tuning"""

rf_tunned_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf_tunned_model.fit(X_train, y_train)

y_pred_tunned = rf_tunned_model.predict(X_test)
y_pred_tunned

accuracy_score(y_test, y_pred_tunned)

'''-------------------------------------------------- .pkl file--------------------------------------------------------------------'''
"""# Creating Pickle file"""

import pickle

pickle.dump(rf_tunned_model, open('Random_Forest_Model.pkl', 'wb'))

pickled_model = pickle.load(open('Random_Forest_Model.pkl', 'rb'))

'''----------------------------------------------------------------------------------------------------------------------'''
"""## Batch Input"""

pickled_model.predict(X_test)

X.head()

Feature_dict = {
'volatile acidity':1,
'citric acid':2,
'residual sugar':3,
'chlorides':4,
'free sulfur dioxide':5,
'total sulfur dioxide':6,
'density':7,
'pH':8,
'sulphates':9,
'alcohol':10
}

Feature_dict

Feature_dict.values()

list(Feature_dict.values())

'''----------------------------------------------------------------------------------------------------------------------'''
"""# Single input single output"""

list([Feature_dict.values()])

pickled_model.predict([list(Feature_dict.values())])[0]

'''------------------------------------------------------Naive Bayes----------------------------------------------------------------'''
'''Naive Bayes'''
from sklearn.naive_bayes import GaussianNB
Naive_Bayes_model = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Naive_Bayes_model.fit(X_train, y_train)
y_pred = Naive_Bayes_model.predict(X_test)
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
confusion_matrix(y_test, y_pred)

