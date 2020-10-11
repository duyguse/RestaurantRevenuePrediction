# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:52:51 2019

@author: u21w97
"""
import pandas as pd
import preprocess
import numpy as np

train = pd.read_csv('restaurantrevenueprediction/train.csv')
test = pd.read_csv('restaurantrevenueprediction/test.csv')
test_label = pd.read_csv('restaurantrevenueprediction/sampleSubmission.csv')

train_new = pd.DataFrame()

pp = preprocess.Preprocess(train)

train_new = pp.Age(train_new)
train_new = pp.City_Group(train_new)
train_new = pp.Type(train_new)
train_new = pp.P_Groups(train_new, True)

X_train = train_new.iloc[:70].drop(["revenue"],axis=1)
y_train = train_new.iloc[:70]["revenue"]
X_val = train_new.iloc[70:100].drop(["revenue"],axis=1)
y_val = train_new.iloc[70:100]["revenue"]

from GradientBoost import GradientBoost

reg = GradientBoost(0.1, 30, 10)
reg.fit(X_train.values, y_train.values, X_val.values, y_val.values)


import pickle
object_file = open("restaurantrevenueprediction/model.pickle","wb")  
pickle.dump(reg,object_file)

object_file = open("restaurantrevenueprediction/model.pickle","rb")  
gb = pickle.load(object_file)

X_test = train_new.iloc[100:].drop(["revenue"], axis=1)
y_test = train_new.iloc[100:]["revenue"]
y_pred = gb.predict(X_test.values)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test.values, y_pred)/np.var(y_test))

