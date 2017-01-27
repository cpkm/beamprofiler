# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:31:57 2016

@author: cpkmanchee

Notes on beamwaist:

I ~ exp(-2*r**2/w0**2)
w0 is 1/e^2 waist radius
w0 = 2*sigma (sigma normal definition in gaussian)
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

import glob



def gaussian2D(xy_meshgrid,x0,y0,sigx,sigy,amp,const,theta=0):
    '''
    generates a 2D gaussian surface of size (n x m)
    
    Inputs:
    
        xy_meshgrid = [x,y]
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

    x = xy_meshgrid[0]
    y = xy_meshgrid[1]

    a = np.cos(theta)**2/(2*sigx**2) + np.sin(theta)**2/(2*sigy**2)
    b = -np.sin(2*theta)/(4*sigx**2) + np.sin(2*theta)/(4*sigy**2)
    c = np.sin(theta)**2/(2*sigx**2) + np.cos(theta)**2/(2*sigy**2)

    g = amp*np.exp(-(a*(x-x0)**2 -b*(x-x0)*(y-y0) + c*(y-y0)**2)) + const
       
    return g.ravel()


def fitgaussian2D(data, xy_tuple, rot=None):
    '''
    fits data to 2D gaussian function

    Inputs:
    	data = 2D array of image data
    	xy_tuple = x and y data as a tuple. each is a meshgrid
    	rot = enable rotation, default is None, if anything else rotation is allowed
    
    Outputs:
        popt, pcov = optimized parameters and covarience matrix
            popt = [x0,y0,sigx,sigy,amp,const,*theta], *theta not included if rot=None
    '''

    data = data.astype(float)

    x = xy_tuple[0]
    y = xy_tuple[1]
    
    ind0 = np.unravel_index(data.argmax(), data.shape)   
    x0 = x[ind0]
    y0 = y[ind0]
    sigx = x.max()/(2*4)
    sigy = y.max()/(2*4)
    
    p0 = [x0,y0,sigx,sigy,data.max(),0]

    if rot is not None:
        p0 += [0]
    
    popt,pcov = opt.curve_fit(gaussian2D,(x,y),data.ravel(),p0)
    
    return popt,pcov
    

def gaussianbeamwaist(z,z0,w0,M2=1,const=0,wl=1.030):
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
    z = 1000*np.asarray(z).astype(float)
    z0 = 1000*z0
 
    w = (w0**2 + M2**2*(wl/(np.pi*w0))**2*(z-z0)**2)**(1/2) +const

    return w

def fitM2(wz,z):
       
    w0 = np.min(wz)
    z0 = z[np.argmin(wz)]
    const = 0
    
    p0 = [z0,w0,1,const]
    
    limits = ([0,-np.inf,1,-np.inf], np.inf)
        
    popt,pcov = opt.curve_fit(gaussianbeamwaist,z,wz,p0,bounds=limits)
    
    return popt,pcov
    


def getroi(data,Nsig=4):
    '''
    Generates a region of interest for a 2D array, based on the varience of the data.
    Cropping box is defined by [left, bottom, width, height]
    
    Inputs:
        data = data array, 2D
        Nsig = number of std away from average to include in roi, default is 4 (99.994% inclusion)
        
    Outputs:
        data_roi = cropped data set, 2D array, size height x width    
    '''
    
    datax = np.sum(data,0)
    datay = np.sum(data,1)
    
    x = np.arange(datax.shape[0])
    y = np.arange(datay.shape[0])

    avgx = np.average(x, weights = datax)
    avgy = np.average(y, weights = datay)
    
    sigx = np.sqrt(np.sum(datax*(x-avgx)**2)/datax.sum())
    sigy = np.sqrt(np.sum(datay*(y-avgy)**2)/datay.sum())

    left = np.int(avgx - Nsig*sigx)
    bottom = np.int(avgy - Nsig*sigy)
    width = np.int(2*Nsig*sigx)
    height = np.int(2*Nsig*sigy)

    if left <= 0:
        width += left
        left = 0

    if bottom <= 0:
        height += bottom
        bottom = 0

    if left+width > data.shape[1]:
        width = data.shape[1] - left

    if bottom+height > data.shape[0]:
        height = data.shape[0] - bottom 

    return data[bottom:bottom+height,left:left+width]


def flattenrgb(im, bits=8, satlim=0.001):
    '''
    Flattens rbg array, excluding saturadted channels
    '''
    
    Nnnz = np.zeros(im.shape[2])
    Nsat = np.zeros(im.shape[2])
    data = np.zeros(im[...,0].shape, dtype = 'uint32')

    for i in range(im.shape[2]):
        
        Nnnz[i] = (im[:,:,i] != 0).sum()
        Nsat[i] = (im[:,:,i] >= 2**bits-1).sum()
        
        if Nsat[i]/Nnnz[i] <= satlim:
            data += im[:,:,i]

    return data
    

def d4sigma(data, xy):
    
    '''
    calculate D4sigma of beam
    '''
    
    x = xy[0]
    y = xy[1]
    
    dx,dy = np.meshgrid(np.gradient(x[0]),np.gradient(y[:,0]))
    
    A = np.sum(data*dx*dy)
    
    avgx = np.sum(data*x*dx*dy)/A
    avgy = np.sum(data*y*dx*dy)/A
    
    d4sigmax = 4*np.sqrt(np.sum(data*(x-avgx)**2*dx*dy)/A)
    d4sigmay = 4*np.sqrt(np.sum(data*(y-avgy)**2*dx*dy)/A)
    
    return np.array([avgx, avgy, d4sigmax, d4sigmay])
    

'''
End of definitions
'''    


BITS = 8       #image channel intensity resolution
SAT = [0,0,0]  #channel saturation detection
SATLIM = 0.001  #fraction of non-zero pixels allowed to be saturated
PIXSIZE = 1.4;  #pixel size in um, assumed square


filedir = '2016-07-29 testdata'
files = glob.glob(filedir+'/*.jpg')

beam_parameters = []
beam_stats = []

for f in files:
    
    im = plt.imread(f)
    data = flattenrgb(im, BITS, SATLIM)
    
    data = data.astype(float)

    data_full = data    
    data = getroi(data)

    x = np.arange(data.shape[1])*PIXSIZE
    y = np.arange(data.shape[0])*PIXSIZE
    x,y = np.meshgrid(x,y)

    popt, pcov = fitgaussian2D(data, (x,y),1)
    d4stats = d4sigma(data, (x,y))
               
    beam_parameters += [popt] 
    beam_stats += [d4stats]
    
beam_parameters = np.asarray(beam_parameters)
beam_stats = np.asarray(beam_stats)

wx = 2*beam_parameters[:,2]
wy = 2*beam_parameters[:,3]
d2x = (1/2)*beam_stats[:,2]
d2y = (1/2)*beam_stats[:,3]

z = np.loadtxt(filedir + '/position.txt', skiprows = 1)
z = 2*z

poptx, pcovx = fitM2(d2x,z)
popty, pcovy = fitM2(d2y,z)

plt.plot(z,d2x)
plt.plot(z,d2y)
plt.plot(z,gaussianbeamwaist(z,*poptx))
plt.plot(z,gaussianbeamwaist(z,*popty))


'''
    #plot orig
    #fig1, ax1 = plt.subplots(1,1)
    #ax1.imshow(im)
    
    #plot result
    data_fitted = gaussian2D((x,y), *popt)

    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    ax.imshow(data, cmap=plt.cm.jet, origin='bottom',
        extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, data_fitted.reshape(data.shape), 10, colors='w')
    plt.show()   
'''





    