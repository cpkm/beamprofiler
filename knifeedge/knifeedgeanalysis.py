# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:24:59 2016

@author: cpkmanchee
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import glob



file = glob.glob('*.csv')
data = np.loadtxt(file[0], delimiter = ',',skiprows = 1)
x = data[:,0]
power = data[:,1]