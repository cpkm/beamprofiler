# beamprofiler
## Overview
[beamprofiler](beamprofiler) is a set of methods and tools used to collect and analyze optical beam profiles.

   knifeedge - analysis for knifeedge measurements.  
   m2scan - gui for collecting beam profile images (allowing for m2 analysis).  
   beamprofiler - analysis of beam profile images, m2 fitting.  


## 1. knifeedge
### Description
[knifeedge](knifeedge/knifeedgeanalysis.py) is used to process a knife-edge profile and compute the corresponding beam profile. Final implementation is still a work-in-progress, but the method can still be used as-is. Currently, the local folder is scanned for .csv files (knife-edge measurement). The first column of the data file should be position; the second column should be power/intensity. This data is differentiated to return the beam profile to the normalized variable `prof`.

The knifeedge [knifeedge](knifeedge) folder contained sample data for testing.

### Usage
To use the analysis method, copy [knifeedgeanalysis.py](knifeedge/knifeedgeanalysis.py) into the same folder as your data. If using an IDE you can run the script, then use the output as necessary. Outside an IDE you can import the script - output variables with be preceeded by the script name, e.g `knifeedgeanalysis.prof`. You can also edit and run the script as necessary.


## 2. m2scan
### Description
[m2scan](m2scan) contains GUI used to collect beam profile images. This can be used in conjuction with any usb camera (e.g. webcam, ccd) connected to the computer. The GUI also has the capability to monitor beampointing by calculating the central position of the beam on the sensor array. [m2gui](m2scan/m2gui.py) has other features such as video previewing, image averaging, sensor saturation detection, data logging, and image capture.

### Usage
Running the script [m2gui](m2scan/m2gui.py) will start the GUI.
#### Beam profile images
##### Image capture
1. Select camera from drop-down menu. If multiple cameras are connected, they are listed as 0,1,2,... Pick the correct one.
2. Select the appropriate pixel bit depth of the camera. Default is 8-bits.
2. Press 'Start Preview' to begin a live stream from the camera. This allows for easy alignment/centering of the beam.
3. Choose a Save directory and filename. Use the 'Browse' button to select a directory.
4. Enter a numerical value of number of images to average per capture. Default is 10.
4. Press 'Capture' to save a beam profile. Images will be saved in the selected directory and numbered starting with 000.jpeg.
5. Take note (seperately) of the micrometer stage position for each image. Create a text file, position.txt (see details below), and input the stage positions. Note: these should be the real positions as read on the scale. The analysis accounts for the 2-pass geometry of the system.

##### Camera pixel saturation
The program will warn you if one or more of the camera channels (RGB) are saturated by the beam. Above the preview image there are three indicator boxes, one for each red, blue and green channels. If a specific channel is saturated the box will light up with it's specific colour. Unsaturated channel boxes remain grey. A channel is deamed saturated if 1% of the pixels are reading the maximum value.

#### Beam pointing
From the main screen press 'Beam pointing' to open the Beam Pointing window.


## 3. beamprofiler
### Description
[beamprofiler](beamprofiler) simply contains the analysis method for images collected with [m2gui](m2scan/m2gui.py). [beamprofileranalysis](beamprofiler/beamprofileranalysis.py) contain methods for analysis. It should be imported into your own analysis script using `from beamprofileranalysis import *`. This way, the methods can be used without prefix. A demo/template analysis script is included [here](beamprofiler/analysis_script.py), which can simply be modified for your analysis. Be sure to have both the method file and analysis script in the same directory. This does not necessarily need to be the same directory as the data you are analyzing, but that may simplify things for you.
### Usage
#### Position text file
For the analysis, images must be accompanied by a text file, named position.txt, containing the position elements of the collected images. The text file should have the following format:
```
<DATE> beam profile z-position
units = <UNITS>  #units of position data, default is mm
<pos1>
.
.
.
<posN>
```
The number of position elements MUST match the number of images. An example is shown [here](beamprofiler/asymmetric_scan/position.txt). Again, this file MUST be named position.txt, and must be in the same directory as your data.
#### Analysis script
You can use the analysis script found [here](beamprofiler/analysis_script.py). The easiest way to run the analysis is as follows:
1. Create position.txt as described above.
2. Copy [analysis_script.py](beamprofiler/analysis_script.py) and [beamprofileranalysis.py](beamprofiler/beamprofileranalysis.py) into your data folder.
3. Ensure [line 20](beamprofiler/analysis_script.py#L20) is set to `cur_dir = True` so data is searched in the current directory (location of method and analysis files).
4. Run analysis_script.
5. You can point the analysis_Script to a different data directory by changing Line 20 to `False` and specifying the data folder on Line 21, `alt_file_location = <data_folder>`. Note that position.txt must be in `<data_folder>`.
