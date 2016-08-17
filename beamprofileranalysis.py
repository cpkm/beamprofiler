# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:31:57 2016

@author: cpkmanchee
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import glob


BITS = 8;       #image channel intensity resolution
SAT = [0,0,0];  #channel saturation detection
SATLIM = 0.02;  #fraction of non-zero pixels allowed to be saturated
PIXSIZE = 1.4;  #pixel size in um, assumed square


filedir = '/Users/cpkmanchee/Google Drive/PhD/Data/2016-07-29 beampro b07p79'

filename = 'WIN_20160729_15_36_54_Pro.jpg'

files = glob.glob(filedir+'/*.jpg')

file = filedir + '/' + filename

for X in files:
    
    im = plt.imread(X)

'''
    plt.subplot(2,2,1)
    plt.imshow(im[:,:,0])
    plt.subplot(2,2,2)
    plt.imshow(im[:,:,1])
    plt.subplot(2,2,3)
    plt.imshow(im[:,:,2])
    plt.subplot(2,2,4)
    plt.imshow(im)
'''
    