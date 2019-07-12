#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:07:17 2019

@author: vikyath
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

def estimate_coef(x, y): 

	n = np.size(x) 
	mx, my = np.mean(x), np.mean(y) 
	xy = np.sum(y*x) - n*my*mx 
	xx = np.sum(x*x) - n*mx*mx 


	b_1 = xy / xx 
	b_0 = my - b_1*mx 

	return(b_0, b_1) 

def plot_regression_line(x, y, b): 
	plt.scatter(x, y, color = "m", 
			marker = "o", s = 30) 


	y_pred = b[0] + b[1]*x 

	plt.plot(x, y_pred, color = "g") 

	plt.xlabel('x') 
	plt.ylabel('y') 

	plt.show() 

def main(): 

    data = pd.read_csv('headbrain.csv')
    x = data.iloc[:,2].values
    y = data.iloc[:,3].values
	
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {} \\nb_1 = {}".format(b[0], b[1])) 

	
    plot_regression_line(x, y, b)  


