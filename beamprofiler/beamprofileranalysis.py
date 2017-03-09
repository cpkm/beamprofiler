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
    
    w = (w0**2 + M2**2*(wl/(np.pi*w0))**2*(z-z0)**2)**(1/2) + const

    return w

def fitM2(wz,z):
       
    w0 = np.min(wz)
    z0 = z[np.argmin(wz)]
    const = 0
    
    p0 = [z0,w0,1,const]
    
    limits = ([0,-np.inf,1,-w0/2], [np.inf,np.inf,np.inf,w0/2])
        
    popt,pcov = opt.curve_fit(gaussianbeamwaist,z,wz,p0,bounds=limits, max_nfev=10000)
    
    return popt,pcov
    


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
    error_limit = 0.01

    x = pix2um(np.arange(data.shape[1]))
    y = pix2um(np.arange(data.shape[0]))
    x,y = np.meshgrid(x,y)

    errx = 1
    erry = 1
    d0x = x.max()
    d0y = y.max()
    roi_new = [d0x/2,d0x,d0y/2,d0y]
    full_data = data

    while any(errx,erry)>error_limit:

        roi = roi_new
        data = get_roi(full_data,roi)
        [ax,ay,s2x,s2y,s2xy] = calculate_2D_moments(data)
        
        g = np.sign(s2x-s2y)
        dx = 2*np.sqrt(2)*((s2x+s2y) + g*((s2x+s2y)**2 + 4*s2xy**2)**(1/2))**(1/2)
        dy = 2*np.sqrt(2)*((s2x+s2y) - g*((s2x+s2y)**2 + 4*s2xy**2)**(1/2))**(1/2)

        errx = np.abs(dx-d0x)/d0x
        erry = np.abs(dy-d0y)/d0y

        roi_new = [ax+roi[1]/2,3*dx,ay+roi[3]/2,3*dy]     #[centrex,width,centrey,height] 





    return


def calculate_2D_moments(data, axes_scale=[1,1]):
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

    #second moments (~varience)
    sig2x = np.sum(data*(x-avgx)**2*dx*dy)/A
    sig2y = np.sum(data*(y-avgy)**2*dx*dy)/A
    sig2xy = np.sum(data*(x-avgx)*(y-avgy)*dx*dy)/A

    return [avgx,avgy,sig2x,sig2y,sig2xy]


def get_roi(data,roi):
    '''
    data = 2D data
    roi = [x0,width,y0,height]
    '''
    left = np.int(roi[0] - roi[1]/2)
    bottom = np.int(roi[2]-roi[3]/2)
    width = np.int(roi[1])
    height = np.int(roi[3])

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



def pix2um(x):

    return x*PIXSIZE
    
'''
End of definitions
'''    


BITS = 8       #image channel intensity resolution
SATLIM = 0.001  #fraction of non-zero pixels allowed to be saturated
PIXSIZE = 1.74  #pixel size in um, measured


filedir = '2016-07-29 testdata'
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

for f in files:
    
    im = plt.imread(f)
    data, sat = flattenrgb(im, BITS, SATLIM)
    #SAT is 1x3 array (RGB), value True (1) means RGB channel was saturated

    data = getroi(data)

    x = pix2um(np.arange(data.shape[1]))
    y = pix2um(np.arange(data.shape[0]))
    x,y = np.meshgrid(x,y)

    d4stats = d4sigma(data, (x,y))

    beam_stats += [d4stats]
    chl_sat += [sat]
    
beam_stats = np.asarray(beam_stats)
chl_sat = np.asarray(chl_sat)

#x and y beam widths
d2x = (1/2)*beam_stats[:,2]
d2y = (1/2)*beam_stats[:,3]

#fit to gaussian mode curve
poptx, pcovx = fitM2(d2x,z)
popty, pcovy = fitM2(d2y,z)

#obtain 'focus image'
focus_number = np.argmin(np.abs(poptx[0]-z))

im = plt.imread(files[focus_number])
data, SAT = flattenrgb(im, BITS, SATLIM)
data = normalize(getroi(data.astype(float)))

x = np.arange(data.shape[1])*PIXSIZE - beam_stats[focus_number,0]
y = np.arange(data.shape[0])*PIXSIZE - beam_stats[focus_number,1]
X,Y = np.meshgrid(x,y)


'''
#plot profile fitresults
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(z,d2x,'bx')
ax1.plot(z,d2y,'go')
ax1.plot(z,gaussianbeamwaist(z,*poptx),'b')
ax1.plot(z,gaussianbeamwaist(z,*popty),'g')

#plot image of focussed beam profile
fig2, ax2 = plt.subplots(1, 1)
ax2.imshow(data, cmap=plt.cm.plasma, origin='lower', interpolation='bilinear',
    extent=(x.min(), x.max(), y.min(), y.max()))
contours = data.max()*(np.exp(-np.array([2,1.5,1,0.5])**2/2))
widths = np.array([1,0.5,1,0.5])
CS = ax2.contour(data, contours, colors = 'k', linewidths = widths, origin='lower', interpolation='bilinear', extent=(x.min(), x.max(), y.min(), y.max()))

ax2.plot(x, (1/4)*(y.max()-y.min())*np.sum(data,0)/np.sum(data,0).max() + y.min(), 'w')
ax2.plot((1/4)*(x.max()-x.min())*np.sum(data,1)/np.sum(data,1).max() + x.min(), y, 'w')
plt.show()   
'''
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

ax5.plot(z,d2x,'bx',markerfacecolor='none')
ax5.plot(z,d2y,'g+',markerfacecolor='none')
ax5.plot(z,gaussianbeamwaist(z,*poptx),'b', label='X')
ax5.plot(z,gaussianbeamwaist(z,*popty),'g', label='Y')
ax5.set_xlabel('z (mm)')
ax5.set_ylabel('Beam Radius (um)')
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position('right')
ax5.legend(loc=9)

ax6.text(-0.1,0.2,'M2x = %.2f\nM2y = %.2f\nwx = %.2f\nwy = %.2f' %(poptx[2],popty[2],poptx[1],popty[1]))

print(poptx)
plt.show()
