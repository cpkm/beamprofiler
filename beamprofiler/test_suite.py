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

def test_fit(z,val_in,na,wl_in):
    '''run test
    '''
    d_in = 2*gaussianbeamwaist(z,*val_in[:3],wl_in)*(1+(na*(np.random.randn(z.size))))

    #Fit test data
    val, err = fit_M2(d_in,z,wl_in)

    return val,err,d_in


def gaussian2D(XY, sx, sy, x0, y0, phi):
    '''2D gaussian
    XY - xy coordinate??
    sx - sigma x
    sy sigma y
    '''
    x,y = XY

    a = np.cos(phi)**2/(2*sx**2) + np.sin(phi)**2/(2*sy**2)
    b = -np.sin(2*phi)/(4*sx**2) + np.sin(2*phi)/(4*sy**2)
    c = np.sin(phi)**2/(2*sx**2) + np.cos(phi)**2/(2*sy**2)

    output = np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

    return output


def digitize_array(data, bits=8):
    '''digitize arrray. Round all values to integer. Saturate pixels over bit limit.
    '''
    data = data.astype(np.uint32)
    data[data>2**bits-1] = 2**bits-1

    return data


## Test M2 fitting
#Define input test points
z0_in = 0
d0_in = 50.0E-6
M2_in = 2.0
wl_in = 1.03E-6
zR_in = np.pi*(d0_in/2)**2/(wl_in*M2_in)
theta_in = 2*M2_in*wl_in/(np.pi*d0_in/2)
val_in = [z0_in, d0_in, M2_in, theta_in, zR_in]

N = 10

num_pts=10
na = 0.05
nd = np.floor(num_pts/4)

#create symmetric spacial array within ISO spec.
z = np.concatenate((np.linspace(-4*zR_in,-2*zR_in,nd), np.linspace(-zR_in,zR_in,2*nd), np.linspace(2*zR_in,4*zR_in,nd)))

#Perform test
v_sum=np.zeros(np.size(val_in))
v2_sum=np.zeros(np.size(val_in))
s_sum=np.zeros(np.size(val_in))
s2_sum=np.zeros(np.size(val_in))

for i in range(N):
    v,s,_ = test_fit(z,val_in,na,wl_in)

    v_sum = v_sum + np.array(v)
    v2_sum = v2_sum + np.array(v)**2
    s_sum = s_sum + np.array(s)
    s2_sum = s2_sum + np.array(s)**2

val_avg = v_sum/N
val_std = ((1/(N-1))*(v2_sum - v_sum**2/N))**(1/2)

std_avg = s_sum/N
std_std = ((1/(N-1))*(s2_sum - s_sum**2/N))**(1/2)

#Plot results
f1,ax1 = plt.subplots(1,1)
rows = ['z0', 'd0', 'M2', 'theta', 'zR']
cols = ['Actual', 'Fit', 'MCsig', 'Fit Sigma','MCsig']

ax1.axis('tight')
ax1.axis('off')
ax1.table(cellText=np.transpose([val_in,val_avg,val_std,std_avg,std_std]), rowLabels = rows, colLabels = cols, loc='center')
#Create table to compare values

#ax0.plot(z,d_in,'bx')
#ax0.plot(z,2*gaussianbeamwaist(z,*val[:3], wl_in))


## Test image processing
# Generate test image

#image properties
w = 1280
h = 720
bits = 8

x = pix2len(np.arange(w))
y = pix2len(np.arange(h))

#beam profile properties
sx = pix2len(h)/12
sy = sx/2
x0 = pix2len(w/2)
y0 = pix2len(h/2)
phi = 0
amp = 256


#generate image
data = amp*gaussian2D(np.meshgrid(x,y),sx,sy,x0,y0,phi)
img = digitize_array(data, bits=bits).reshape(h,w,1).repeat(3,2)

#analyze image
im,sat_det = flatten_rgb(img, force_chn=0)
beamwidths, roi, moments = calculate_beamwidths(im)

x0_out = pix2len(roi[0]) 
y0_out = pix2len(roi[2]) 

#Plot results
f2,ax2 = plt.subplots(1,1)
rows = ['dx', 'dy', 'phi', 'x0', 'y0']
cols = ['Actual', 'Fit', '"%" diff']

val_in = [4*sx,4*sy,phi,x0,y0]
val_out = beamwidths+[x0_out,y0_out]
ratio = np.asarray(val_out)/np.asarray(val_in)

ax2.axis('tight')
ax2.axis('off')
ax2.table(cellText=np.transpose([val_in,val_out,ratio]), rowLabels = rows, colLabels = cols, loc='center')

f3,ax3 = plt.subplots(1,1)
ax3.imshow(im)

plt.show()

