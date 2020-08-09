# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('data.csv')


dataset.head()
dataset.isna()
dataset.info
dataset.dtypes
#only object diagnosis tumor or not
dataset['diagnosis']  #m malignant, b begnine : that is our outcome y

#encode object with onehotencoder
#y_encoded = pd.get_dummies(data_Y, drop_first = True)

#scale numerical data

data_2 = dataset.copy()
data_Y = data_2['diagnosis']
data_Y = data_Y.ravel()
data_Y_encoded = pd.get_dummies(data_Y, drop_first = True)
data_X = data_2.drop(['diagnosis'], axis =1)

data_X.dtypes #good only numerical values
data_X.isna().sum()
data_X_cleaned = data_X.drop(['Unnamed: 32'], axis = 1)
#error : input contains Nan, Infinity or a value too large for dtype('float32)
#drop unnamed column of numerical column with no data (569/569 missing)
data_X_cleaned.isna().sum()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_X_cleaned, data_Y_encoded, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#using voting classifiers to get better accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

rdn_clf = RandomForestClassifier()
svc_clf = SVC()
voting_clf = VotingClassifier(estimators = [('forest', rdn_clf), ('svm', svc_clf)], 
                                                            voting = 'hard')


from sklearn.metrics import accuracy_score
for clf in (rdn_clf, svc_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_pred, y_test)) 

'''accuracy
RandomForestClassifier: 0.979
SVC: 0.958
VotingClassifier: 0.965
'''







