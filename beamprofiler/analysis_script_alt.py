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

#Override defaults if necessary (imported from *)
#BITS = 8       #image channel intensity resolution
#SATLIM = 0.001  #fraction of non-zero pixels allowed to be saturated
#PIXSIZE = 1.745E-6  #pixel size in m, measured 1.75um in x and y

cur_dir = False
alt_file_location = '/Users/cpkmanchee/Google Drive/PhD/Data/2017-09-20 Rod seed beam profile/2017-09-20 Unamplied seed'

if cur_dir:
    file_dir = os.getcwd()
else:
    file_dir = alt_file_location

files = glob.glob(file_dir+'/*.jp*g')
files.sort()

# Consistency check, raises error if failure
if not files:
    stop('No files found... try again, but be better')

try:
    z = read_position(file_dir)
    #z = np.loadtxt(file_dir + '/position.txt', skiprows = 2)
    #double pass configuration, covert mm to meters
    #z = 2*z*1E-3
except (AttributeError, FileNotFoundError):
    stop('No position file found --> best be named "position.txt"')

if np.size(files) is not np.size(z):
    stop('# of images does not match positions - fix ASAP')

order = np.argsort(z)
z = np.array(z)[order]
files = np.array(files)[order]

#output parameters
beam_stats = []
chl_sat = []
img_roi = []

for f in files:

    im = plt.imread(f)
    data, sat = flatten_rgb(im, BITS, SATLIM, sgl_chn=False)
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
valx, stdx = fit_M2(d4x,z)
valy, stdy = fit_M2(d4y,z)


##
#plotting for pretty pictures

#obtain 'focus image', chooses based on x-direction
focus_number = np.argmin(np.abs(valx[0]-z))

im = plt.imread(files[focus_number])
data, SAT = flatten_rgb(im, BITS, SATLIM)
data = normalize(get_roi(data.astype(float), img_roi[focus_number]))

#put x,y in um, z in mm
x = pix2len(1)*(np.arange(data.shape[1]) - data.shape[1]/2)*1E6
y = pix2len(1)*(np.arange(data.shape[0]) - data.shape[0]/2)*1E6
X,Y = np.meshgrid(x,y)
Z = (z-valx[0])*1E3

#set up plot grid
plot_grid = [2,2]
plot_w = 8
asp_ratio = plot_grid[0]/plot_grid[1]
plot_h = plot_w*asp_ratio
colormap = plt.cm.plasma

fig = plt.figure(figsize=(plot_w,plot_h), facecolor='w')
gs = GridSpec(plot_grid[0], plot_grid[1])

# identical to ax1 = plt.subplot(gs.new_subplotspec((0,0), colspan=3))
ax2 = plt.subplot(gs[0,0])
ax3 = plt.subplot(gs[0, 1], projection='3d')

ax7 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
#ax6 = plt.subplot(gs[-1,0])

#ax6.axis('off')
#make_ticklabels_invisible([ax2])

ax2.imshow(data, cmap=colormap, origin='lower', interpolation='bilinear',
    extent=(x.min(), x.max(), y.min(), y.max()))
#contours = data.max()*(np.exp(-np.array([2,1.5,1])**2/2))
#widths = np.array([0.5,0.25,0.5])
#CS = ax2.contour(data, contours, colors = 'k', linewidths = widths, origin='lower', interpolation='bilinear', extent=(x.min(), x.max(), y.min(), y.max()))

y_off = (y.max()-y.min())/2 
x_off = (x.max()-x.min())/2

ax2.plot(x, 0.5*y_off*normalize(np.sum(data,0)) - y_off, 'w')
ax2.set_xlabel('x (um)')

ax2.plot(0.5*x_off*normalize(np.sum(data,1)) - x_off, y, 'w')
ax2.set_ylabel('y (um)')

ax3.plot_surface(X,Y,data, cmap=colormap)
ax3.set_ylabel('y (um)')
ax3.set_xlabel('x (um)')

ax5.plot(Z,d4x*1E6,'x', c=colormap(0.1), markerfacecolor='none')
ax5.plot(Z,d4y*1E6,'+', c=colormap(0.9), markerfacecolor='none')
ax5.plot(Z,2*gaussianbeamwaist(z,*valx[:3])*1E6, c=colormap(0.3), label='X')
ax5.plot(Z,2*gaussianbeamwaist(z,*valy[:3])*1E6, c=colormap(0.7), label='Y')
ax6 = ax5.twinx()
ax6.plot(Z, beam_stats[:,2], c=colormap(0.5), label='phi')

ax6.yaxis.tick_left()
ax6.yaxis.set_label_position('left')
ax6.set_ylim([-np.pi/2,np.pi/2])
ax5.set_xlabel('z (mm)')
ax5.set_ylabel('Spot size, 2w (um)')
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position('right')
ax5.legend(loc=9)



#ax6.text(-0.2,0.2,'M2x = {:.2f}\nM2y = {:.2f}\nwx = {:.0f} um\nwy = {:.0f} um'.format(valx[2],valy[2],(1E6)*valx[1]/2,(1E6)*valy[1]/2))

rows = ['z0', 'd0', 'M2', 'theta', 'zR']
cols = ['val_x', 'std_x', 'val_y', 'std_y']
tabString = [['{:.3e}'.format(j) for j in i] for i in np.transpose([valx,stdx,valy,stdy])]


ax7.axis('tight')
ax7.axis('off')
ax7.table(cellText=tabString, 
    rowLabels = rows, colLabels = cols, loc='center')

plt.show()
