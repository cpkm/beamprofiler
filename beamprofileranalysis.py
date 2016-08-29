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


def gaussian2D(X,x0,y0,sigx,sigy,amp,const,theta=0):
    '''
    generates a 2D gaussian surface of size (n x m)
    
    Inputs:
    
        X = [x,y]
        x = meshgrid of x array
        y = meshgrid of y array
        
    where x and y are of size (n x m)
    n = y.shape[0] (or x.) = number of rows
    m = x.shape[1] (or y.) = number of columns
    
        x0,y0 = peak location
    
        sig_ = standard deviation in x and y, gaussian 1/e radius
    
        amp = amplitude
    
        const = offset (constant)

        theta = rotation parameter, 0 by default
    
    Output:
        
        g.ravel() = flattened array of gaussian amplitude data
    
    where g is the 2D array of gaussian amplitudes of size (n x m)
    '''

    x = X[0]
    y = X[1]

    a = np.cos(theta)**2/(2*sigx**2) + np.sin(theta)**2/(2*sigy**2)
    b = -np.sin(2*theta)/(4*sigx**2) + np.sin(2*theta)/(4*sigy**2)
    c = np.sin(theta)**2/(2*sigx**2) + np.cos(theta)**2/(2*sigy**2)

    g = amp*np.exp(-(a*(x-x0)**2 -b*(x-x0)*(y-y0) + c*(y-y0)**2)) + const
       
    return g.ravel()


def fitgaussian2D(data, rot=None):
    '''
    fits data to 2D gaussian function

    Inputs:
    	data = 2D array of image data
    '''
    data = data.astype(float)

    x = np.arange(data.shape[1])*PIXSIZE
    y = np.arange(data.shape[0])*PIXSIZE
    x,y = np.meshgrid(x,y)
    
    (y0,x0) = np.unravel_index(data.argmax(), data.shape)   
    x0 *= PIXSIZE
    y0 *= PIXSIZE
    sigx = 50
    sigy = 50
    
    p0 = [x0,y0,sigx,sigy,data.max(),0]

    if rot is not None:
        p0 += [0]
    
    popt,pcov = opt.curve_fit(gaussian2D,(x,y),data.ravel(),p0)
    
    return popt,pcov
    


def gaussianbeamwaist(z,z0,w0,M2=1,wl=1.030):
    '''
    generate gaussian beam profile w(z)

    w(z) = w0*(1+((z-z0)/zR)^2)^(1/2)

    where zR is the Rayleigh length given by

    zR = pi*w0^2/wl

    Units for waist and wavelength is um.
    Units for optical axis positon is mm.

    Inputs:

        z = array of position along optical axis in mm
        z0 = position of beam waist (focus) in mm
        w0 = beam waist in um
        M2 = beam parameter M^2, unitless
        wl =  wavelenth in um, default to 1030nm (1.03um)

    Outputs:

        w = w(z) position dependent beam waist in um. Same size as 'z'
    '''
    z = np.asarray(z).astype(float)
    w = w0*(1+(((z-z0)*1000)*M2/(np.pi*w0**2/wl))**2)**(1/2)

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
    
    Nnnz = np.zeros(im.shape[2])
    Nsat = np.zeros(im.shape[2])
    data = np.zeros(im[...,0].shape, dtype = 'uint32')

    for i in range(im.shape[2]):
        
        Nnnz[i] = (im[:,:,i] != 0).sum()
        Nsat[i] = (im[:,:,i] >= 2**BITS-1).sum()
        
        if Nsat[i]/Nnnz[i] <= SATLIM:
            data += im[:,:,i]
            
    data = data.astype(float)

    popt, pcov = fitgaussian2D(data, 1)
       
        
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


'''
# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data.reshape(201, 201))
plt.colorbar()

#plot result
data_fitted = twoD_Gaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
plt.show()
'''

    