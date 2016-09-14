# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:24:59 2016

@author: cpkmanchee
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import glob


def gaussian(x,x0,sig,amp,const=0):
    return amp*np.exp(-(x-x0)**2/(2*sig**2)) + const

def normalize(f):
    return (f-f.min())/(f.max()-f.min())

file = glob.glob('*.csv')
data = np.loadtxt(file[0], delimiter = ',',skiprows = 1)
x = data[:,0]
power = data[:,1]

prof = -np.gradient(power,np.gradient(x))
prof = normalize(prof)

avgx = np.average(x, weights = prof)
sig = np.sqrt(np.sum(prof*(x-avgx)**2)/prof.sum())

'''
p0 = [avgx, sig, 1,0]
popt, pcov =  opt.curve_fit(gaussian, x, prof, p0)
'''

