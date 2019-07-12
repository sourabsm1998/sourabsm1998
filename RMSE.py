#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:06:32 2019

@author: vikyath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





data = pd.read_csv('50_Startups.csv')

data.columns
x = data.iloc[:,:4].values

y = data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder

lEncoder = LabelEncoder()

x[:,3] = lEncoder.fit_transform(x[:,3])

from sklearn.preprocessing import OneHotEncoder
ohEncoder = OneHotEncoder(categorical_features=[3])

x = ohEncoder.fit_transform(x).toarray()

x = x[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)
m = regressor.coef_
c = regressor.intercept_

y_pred = regressor.predict(x_test)

print('Coefficicents :\n',m)

print('score :{} '.format(regressor.score(x_test,y_test)))

from sklearn.metrics import mean_squared_error


mse = mean_squared_error(y_test,y_pred)**1/2

plt.scatter(regressor.predict(x_train),regressor.predict(x_train)-y_train, color="Red",s = 10 , label = "train data")

plt.scatter(regressor.predict(x_test),regressor.predict(x_test)-y_test, color="blue",s = 10 , label = "test data")

#%matplotlib auto

plt.show()



