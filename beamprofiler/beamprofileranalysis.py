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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import scipy.optimize as opt
import uncertainties as un

import glob
import time

def stop(s = 'error'): raise Exception(s)


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
    

def gaussianbeamwaist(z,z0,d0,M2=1,const=0,wl=1.030):
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
        d0 = beam waist in um (diameter)
        M2 = beam parameter M^2, unitless
        wl =  wavelenth in um, default to 1030nm (1.03um)

    Outputs:

        w = w(z) position dependent beam waist in um. Same size as 'z'
    '''
    z = 1000*np.asarray(z).astype(float)
    z0 = 1000*z0
    w0 = d0/2
    
    w = (w0**2 + M2**2*(wl/(np.pi*w0))**2*(z-z0)**2)**(1/2) + const

    return w

def fitM2(dz, z, wl=1.03E-6):
    '''
    z in mm
    dz, beamwidth in um

    '''

    dz = dz*1E-6
    z = z*1E-3
    
    di = np.min(dz)
    zi = z[np.argmin(dz)]
    
    ci = (wl/(np.pi*di))**2
    bi = -2*zi*ci
    ai = di**2 + ci*zi**2
    
    c_lim = np.array([0, np.inf])
    b_lim = np.array([-np.inf,np.inf])
    a_lim = np.array([0, np.inf])

    p0 = [ai, bi, ci]
    #limits = (-np.inf,np.inf)
    limits = ([i.min() for i in [a_lim,b_lim,c_lim]],  [i.max() for i in [a_lim,b_lim,c_lim]])

    f = lambda z,a,b,c: (a + b*z + c*z**2)**(1/2)
        
    popt,pcov = opt.curve_fit(f,z,dz,p0,bounds=limits)
    
    [a,b,c] = [un.ufloat(popt[i], np.sqrt(pcov[i,i])) for i in range(3)]

    z0 = (1E3)*(-b/(2*c))
    d0 = (1E6)*((4*a*c-b**2)**(1/2))*(1/(2*c**(1/2)))
    M2 = (np.pi/(8*wl))*((4*a*c-b**2)**(1/2))
    theta = c**(1/2)
    zR = (1E3)*((4*a*c-b**2)**(1/2))*(1/(2*c))

    value = [x.nominal_value for x in [z0,d0,M2,theta,zR]]
    std = [x.std_dev for x in [z0,d0,M2,theta,zR]]

    return value, std
    


def getroi(data,Nsig=3):
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
    Flattens rbg array, excluding saturated channels
    '''

    sat_det = np.zeros(im.shape[2])
    
    Nnnz = np.zeros(im.shape[2])
    Nsat = np.zeros(im.shape[2])
    data = np.zeros(im[...,0].shape, dtype = 'uint32')

    for i in range(im.shape[2]):
        
        Nnnz[i] = (im[:,:,i] != 0).sum()
        Nsat[i] = (im[:,:,i] >= 2**bits-1).sum()
        
        if Nsat[i]/Nnnz[i] <= satlim:
            data += im[:,:,i]
        else:
            sat_det[i] = 1
    
    output = normalize(data.astype(float))

    return output, sat_det
    

def d4sigma(data, xy):
    
    '''
    calculate D4sigma of beam
    x,y,data all same size
    A is normalization factor
    returns averages and d4sig in x and y
    x and y directions are orientation of image. no adjustment
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
    

def normalize(data, offset=0):
    '''
    normalize a dataset
    data is array or matrix to be normalized
    offset = (optional) constant offset
    '''
    
    return (data-data.min())/(data.max()-data.min()) + offset
    


def make_ticklabels_invisible(axes):
    for ax in axes:
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


def calculate_beamwidths(data):
    '''
    data = image matrix

    data,x,y all same dimensions
    '''
    error_limit = 0.0001
    it_limit = 5

    #x = pix2um(np.arange(data.shape[1]))
    #y = pix2um(np.arange(data.shape[0]))
    #x,y = np.meshgrid(x,y)

    errx = 1
    erry = 1
    itN = 0

    d0x = data.shape[1]
    d0y = data.shape[0]

    roi_new = [d0x/2,d0x,d0y/2,d0y]
    full_data = data

    while any([i>error_limit for i in [errx,erry]]):

        roi = roi_new
        data = get_roi(full_data,roi)
        moments = calculate_2D_moments(data)

        dx = 4*np.sqrt(moments[2])
        dy = 4*np.sqrt(moments[3])

        errx = np.abs(dx-d0x)/d0x
        erry = np.abs(dy-d0y)/d0y
        
        d0x = dx
        d0y = dy

        roi_new = [moments[0]+roi[0]-roi[1]/2,3*dx,moments[1]+roi[2]-roi[3]/2,3*dy]     #[centrex,width,centrey,height] 
        
        itN += 1
        if itN >= it_limit:
            print('exceeded iteration in calculating moments')
            break
   
    pixel_scale = [pix2um(1)]*2+ [pix2um(1)**2]*3
    moments = pixel_scale*moments
    [ax,ay,s2x,s2y,s2xy] = moments


    g = np.sign(s2x-s2y)
    dx = 2*np.sqrt(2)*((s2x+s2y) + g*((s2x-s2y)**2 + 4*s2xy**2)**(1/2))**(1/2)
    dy = 2*np.sqrt(2)*((s2x+s2y) - g*((s2x-s2y)**2 + 4*s2xy**2)**(1/2))**(1/2)

    if s2x == s2y:
        phi = (np.pi/4)*np.sign(s2xy)
    else:
        phi = (1/2)*np.arctan(2*s2xy/(s2x-s2y))

    beamwidths = [dx,dy,phi]

    return beamwidths,roi,moments


def calculate_2D_moments(data, axes_scale=[1,1], calc_2nd_moments = True):
    '''
    data = 2D data
    axes_scale = (optional) scaling factor for x and y

    returns first and second moments

    first moments are averages in each direction
    second moments are variences in x, y and diagonal
    '''
    x = axes_scale[0]*(np.arange(data.shape[1]))
    y = axes_scale[1]*(np.arange(data.shape[0]))
    dx,dy = np.meshgrid(np.gradient(x),np.gradient(y))
    x,y = np.meshgrid(x,y)

    A = np.sum(data*dx*dy)
    
    #first moments (averages)
    avgx = np.sum(data*x*dx*dy)/A
    avgy = np.sum(data*y*dx*dy)/A

    #calculate second moments if required
    if calc_2nd_moments:
        #second moments (~varience)
        sig2x = np.sum(data*(x-avgx)**2*dx*dy)/A
        sig2y = np.sum(data*(y-avgy)**2*dx*dy)/A
        sig2xy = np.sum(data*(x-avgx)*(y-avgy)*dx*dy)/A
        
        return np.array([avgx,avgy,sig2x,sig2y,sig2xy])

    else:
        return np.array([avgx, avgy])


def get_roi(data,roi):
    '''
    data = 2D data
    roi = [x0,width,y0,height]
    '''
    #need to fix!!!
    
    left = roi[0] - roi[1]/2
    bottom = roi[2] - roi[3]/2
    width = roi[1]
    height = roi[3]

    left = min(left,data.shape[1]-2)
    if left <= 0:
        width += left
        left = 0
    elif np.isnan(left):
        left=0
    else:
        left = np.int(left)

    bottom = min(bottom,data.shape[0]-2)
    if bottom <= 0:
        height += bottom
        bottom = 0
    elif np.isnan(bottom):
        bottom = 0
    else:
        bottom = np.int(bottom)

    if left+width > data.shape[1]:
        width = np.int(data.shape[1] - left)
    elif np.isnan(width):
        width = data.shape[1]
    else:
        width = np.int(width)

    if bottom+height > data.shape[0]:
        height = np.int(data.shape[0] - bottom)
    elif np.isnan(height):
        height = data.shape[0]
    else:
        height = np.int(height)

    return data[bottom:bottom+height,left:left+width]



def pix2um(input):

    if np.isscalar(input):
        output = input*PIXSIZE
    else:
        output =  [x*PIXSIZE for x in input]

    return output
    
'''
End of definitions
'''    


BITS = 8       #image channel intensity resolution
SATLIM = 0.001  #fraction of non-zero pixels allowed to be saturated
PIXSIZE = 1.745  #pixel size in um, measured


filedir = 'asymmetric scan'
files = glob.glob(filedir+'/*.jp*g')

# Consistency check, raises error if failure
if not files:
    stop('No files found... try again, but be better')

try:
    z = np.loadtxt(filedir + '/position.txt', skiprows = 1)
    z = 2*z
except (AttributeError, FileNotFoundError):
    stop('No position file found --> best be named "position.txt"')
    
if np.size(files) is not np.size(z):
    stop('# of images does not match positions - fix ASAP')
    

#output parameters
beam_stats = []
chl_sat = []
img_roi = []

for f in files:
    
    im = plt.imread(f)
    data, sat = flattenrgb(im, BITS, SATLIM)

    d4stats, roi , _ = calculate_beamwidths(data)

    beam_stats += [d4stats]
    chl_sat += [sat]
    img_roi += [roi]


    
beam_stats = np.asarray(beam_stats)
chl_sat = np.asarray(chl_sat)
img_roi = np.asarray(img_roi)

#x and y beam widths
d4x = beam_stats[:,0]
d4y = beam_stats[:,1]

#fit to gaussian mode curve
valx, stdx = fitM2(d4x,z)
valy, stdy = fitM2(d4y,z)

#obtain 'focus image'
focus_number = np.argmin(np.abs(valx[0]-z))
Z = z-valx[0]

im = plt.imread(files[focus_number])
data, SAT = flattenrgb(im, BITS, SATLIM)
data = normalize(get_roi(data.astype(float), img_roi[focus_number]))

x = pix2um(1)*(np.arange(data.shape[1]) - data.shape[1]/2)
y = pix2um(1)*(np.arange(data.shape[0]) - data.shape[0]/2)
X,Y = np.meshgrid(x,y)

##
#plotting for pretty pictures
plot_grid = [3,6]
plot_w = 10
asp_ratio = plot_grid[0]/plot_grid[1]
plot_h = plot_w*asp_ratio


fig = plt.figure(figsize=(plot_w,plot_h), facecolor='w')
gs = GridSpec(plot_grid[0], plot_grid[1])
ax1 = plt.subplot(gs[:2, 0])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0,0), colspan=3))
ax2 = plt.subplot(gs[:2,1:3])
ax3 = plt.subplot(gs[:2, 3:], projection='3d')
ax4 = plt.subplot(gs[2,1:3])
ax5 = plt.subplot(gs[2,3:])
ax6 = plt.subplot(gs[-1,0])

ax6.axis('off')
make_ticklabels_invisible([ax2])

ax2.imshow(data, cmap=plt.cm.plasma, origin='lower', interpolation='bilinear',
    extent=(x.min(), x.max(), y.min(), y.max()))
contours = data.max()*(np.exp(-np.array([2,1.5,1])**2/2))
widths = np.array([0.5,0.25,0.5])
CS = ax2.contour(data, contours, colors = 'k', linewidths = widths, origin='lower', interpolation='bilinear', extent=(x.min(), x.max(), y.min(), y.max()))

ax4.plot(x, normalize(np.sum(data,0)), 'k')
ax4.set_xlabel('x (um)')

ax1.plot(normalize(np.sum(data,1)), y, 'k')
ax1.set_ylabel('y (um)')

ax3.plot_surface(X,Y,data, cmap=plt.cm.plasma)
ax3.set_ylabel('y (um)')
ax3.set_xlabel('x (um)')

ax5.plot(Z,d4x,'bx',markerfacecolor='none')
ax5.plot(Z,d4y,'g+',markerfacecolor='none')
ax5.plot(Z,2*gaussianbeamwaist(z,*valx[:3]),'b', label='X')
ax5.plot(Z,2*gaussianbeamwaist(z,*valy[:3]),'g', label='Y')
ax5.set_xlabel('z (mm)')
ax5.set_ylabel('Beam Waist (um)')
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position('right')
ax5.legend(loc=9)

ax6.text(-0.1,0.2,'M2x = %.2f\nM2y = %.2f\nwx = %.2f\nwy = %.2f' %(valx[2],valy[2],valx[1]/2,valy[1]/2))

plt.show()
