#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:25:32 2019

@author: vikyath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def coef(x,y):
    
    
        n = np.size(x)
        
        mx , my = np.mean(x), np.mean(y)
        
        
        sum_xy = np.sum(x*y)- n*mx*my
        sum_xx = np.sum(x*x)- n*mx*mx
        
        b1 = sum_xy / sum_xx
        b0 = my / b1*mx
        
        return(b1,b0)
        
        
def linear(x,y,b):
    plt.plot(x,y,color="Red")
      
    plt.show()
            


def main():
    data = pd.read_csv('headbrain.csv')
    x = data.iloc[:,2].values
    y = data.iloc[:,3].values

    b = coef(x,y)
    linear()
    