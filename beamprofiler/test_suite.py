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
f1,(ax1,ax2) = plt.subplots(2,1)
rows = ['z0', 'd0', 'M2', 'theta', 'zR']
cols = ['Actual', 'Fit', 'Fit Sigma']

#Define input test points
z0_in = 0
d0_in = 50.0E-6
M2_in = 2.0
wl_in = 1.03E-6
zR_in = np.pi*(d0_in/2)**2/(wl_in*M2_in)
theta_in = 2*M2_in*wl_in/(np.pi*d0_in/2)
val_in = [z0_in, d0_in, M2_in, theta_in, zR_in]

# Noise amplitude
na = 0.02

def test_fit()

    z = np.linspace(-5*zR_in,5*zR_in,20)
    d_in = 2*gaussianbeamwaist(z,*val_in[:3],wl_in)*(1+(na*(np.random.randn(z.size))))

    #Fit test data
    val, std = fit_M2(d_in,z,wl_in)

    return val,std


#Plot results
ax1.plot(z,d_in,'bx')
ax1.plot(z,2*gaussianbeamwaist(z,*val[:3], wl_in))
ax2.axis('tight')
ax2.axis('off')
ax2.table(cellText=np.transpose([val_in,val,std]), rowLabels = rows, colLabels = cols, loc='center')
plt.show()
#Create table to compare values


## Test image processing
