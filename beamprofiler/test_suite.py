# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:52:32 2017

@author: cpkmanchee

Tests of beamprofiler analysis
"""

import glob
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import uncertainties as un

from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from beamprofileranalysis import *

## Test M2 fitting

#Define input test points
zR_in = 2.0E-6
z0_in = 0
d0_in = 50.0E-6
M2_in = 1.0
wl_in = 1.03E-6

z = np.linspace(-5*zR_in,5*zR_in,20)
w_in = gaussianbeamwaist(z,z0_in,d0_in,M2_in,wl_in)
d_in = 2*w_in

ax1 = plt.subplot([1,1,1])
ax1.plot(z,d_in,'bx')
plt.show()
#Fit test data
#val, std = fit_M2(d_in,z,wl_in)

#Plot results
ax1 = plt.subplot([1,1,1])
ax1.plot(z,d_in,'bx')
ax1.plot(z,2*gaussianbeamwaist(z,*val[:,3]))

#Create table to compare values


## Test image processing
