# beamprofiler
## Overview
[beamprofiler](beamprofiler) is a set of methods and tools used to collect and analyze optical beam profiles.


## knifeedge
### Description
[knifeedge](knifeedge/knifeedgeanalysis.py) is used to process a knife-edge profile and compute the corresponding beam profile. Final implementation is still a work-in-progress, but the method can still be used as-is. Currently, the local folder is scanned for .csv files (knife-edge measurement). The first column of the data file should be position; the second column should be power/intensity. This data is differentiated to return the beam profile to the normalized variable `prof`.

The knifeedge [knifeedge](knifeedge) folder contained sample data for testing.

### Usage
To use the analysis method, copy [knifeedgeanalysis.py](knifeedge/knifeedgeanalysis.py) into the same folder as your data. If using an IDE you can run the script, then use the output as necessary. Outside an IDE you can import the script - output variables with be preceeded by the script name, e.g `knifeedgeanalysis.prof`. You can also edit and run the script as necessary.


## m2scan
### Description
[m2scan](m2scan) contains GUI used to collect beam profile images. This can be used in conjuction with any usb camera (e.g. webcam, ccd) connected to the computer. The GUI also has the capability to monitor beampointing by calculating the central position of the beam on the sensor array. [m2gui](m2scan/m2gui.py) has other features such as video previewing, image averaging, sensor saturation detection, data logging, and image capture.

### Usage
Running the script [m2gui](m2scan/m2gui.py) will start the GUI.


## beamprofiler
### Description
[beamprofiler](beamprofiler) simply contains the analysis method for images collected with [m2gui](m2scan/m2gui.py).
### Usage
For the analysis, images must be accompanied by a text file containing the position elements of the collected images. The text file should have the following format:
```
<DATE> beam profile z-position
units = <UNITS>  #units of position data, default is mm
<pos1>
.
.
.
<posN>
```
The number of position elements MUST match the number of images.
