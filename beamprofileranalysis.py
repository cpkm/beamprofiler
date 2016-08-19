# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:31:57 2016

@author: cpkmanchee
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

import glob


def gaussian2D(X,x0,y0,sigx,sigy,amp,const):
    '''
    generates a 2D gaussian surface of size (n x m)
    
    Inputs:
    
        x = meshgrid of x array
        y = meshgrid of y array
        
    where x and y are of size (n x m)
    n = y.shape[0] (or x.) = number of rows
    m = x.shape[1] (or y.) = number of columns
    
        x0,y0 = peak location
    
        sig_ = standard deviation in x and y, gaussian 1/e radius
    
        amp = amplitude
    
        const = offset (constant)
    
    Output:
        
        g.ravel() = flattened array of gaussian amplitude data
    
    where g is the 2D array of gaussian amplitudes of size (n x m)
    '''
    
    x = X[0]
    y = X[1]
    g = amp*np.exp(-(x-x0)**2/(2*sigx**2) - (y-y0)**2/(2*sigy**2)) + const
       
    return g.ravel()

def gaussianbeamwaist(z,z0,w0,M2=1,lambda = 1.030)
    '''
    generate gaussian beam profile w(z)

    w(z) = w0*(1+((z-z0)/zR)^2)^(1/2)

    where zR is the Rayleigh length given by

    zR = pi*w0^2/lambda

    Units for waist and wavelength is um.
    Units for optical axis positon is mm.

    Inputs:

        z = array of position along optical axis in mm
        z0 = position of beam waist (focus) in mm
        w0 = beam waist in um
        M2 = beam parameter M^2, unitless
        lambda =  wavelenth in um, default to 1030nm (1.03um)

    Outputs:

        w = w(z) position dependent beam waist in um. Same size as 'z'
    '''

    w = w0*(1+((z-z0)*M2/(np.pi*w0**2/lambda))**2)**(1/2)

    return w



BITS = 8;       #image channel intensity resolution
SAT = [0,0,0];  #channel saturation detection
SATLIM = 0.001;  #fraction of non-zero pixels allowed to be saturated
PIXSIZE = 1.4;  #pixel size in um, assumed square


filedir = '2016-07-29 testdata'

filename = 'WIN_20160729_15_36_54_Pro.jpg'

files = glob.glob(filedir+'/*.jpg')

files = [filedir + '/' + filename]

for f in files:
    
    im = plt.imread(f)
    
    x = np.arange(im.shape[1])*PIXSIZE
    y = np.arange(im.shape[0])*PIXSIZE
    x,y = np.meshgrid(x,y)
    Nnnz = np.zeros(im.shape[2])
    Nsat = np.zeros(im.shape[2])
    data = np.zeros(im[...,0].shape, dtype = 'uint32')

    for i in range(im.shape[2]):
        
        Nnnz[i] = (im[:,:,i] != 0).sum()
        Nsat[i] = (im[:,:,i] >= 2**BITS-1).sum()
        
        if Nsat[i]/Nnnz[i] <= SATLIM:
            data += im[:,:,i]
            
    data = data.astype(float)
    
    (y0,x0) = np.unravel_index(data.argmax(), data.shape)   
    x0 *= PIXSIZE
    y0 *= PIXSIZE
    sigx = 50
    sigy = 50
    
    p0 = [x0,y0,sigx,sigy,data.max(),0]
    
    popt,pcov = opt.curve_fit(gaussian2D,(x,y),data.ravel(),p0)
            
    
    
    
    
    
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

plt.show()
    