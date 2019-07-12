#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:53:47 2019

@author: vikyath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data  = pd.read_csv('insurance.csv')

x = data.iloc[:,:6].values

y = data.iloc[:,6].values


from sklearn.preprocessing import LabelEncoder
lEncoder = LabelEncoder()
l1Encoder = LabelEncoder()
l2Encoder = LabelEncoder()


x[:,1] = lEncoder.fit_transform(x[:,1])
x[:,4] = l1Encoder.fit_transform(x[:,4])
x[:,5] = l2Encoder.fit_transform(x[:,5])

from sklearn.preprocessing import OneHotEncoder
ohEncoder = OneHotEncoder(categorical_features=[3])

x = ohEncoder.fit_transform(x).toarray()


a = input('Enter the sex of the individual')
b = lEncoder.inverse_transform([a])
c = input('Enter yes or no if the individual is a smoker')
d = lEncoder.inverse_transform([c])
e = input('Enter the region of the individual')
f = lEncoder.inverse_transform([e])


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .2, random_state=0)

from sklearn.linear_model import LinearRegression


