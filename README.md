# beamprofiler
## Overview
[beamprofiler](http://github.com/cpkm/beamprofiler) is a set of methods and tools used to collect and analyze optical beam profiles.

## knifeedge
[knifeedge](knifeedge/knifeedgeanalysis.py) is used to process a knife-edge profile and compute the corresponding beam profile. Final implementation is still a work-in-progress, but the method can still be used as-is. Currently, the local folder is scanned for .csv files (knife-edge measurement). The first column of the data file should be position; the second column should be power/intensity. This data is differentiated to return the beam profile to the normalized variable `prof`.

The knifeedge [knifeedge](knifeedge) folder contained sample data for testing.

To use the analysis method, copy [knifeedgeanalysis.py](knifeedge/knifeedgeanalysis.py) into the same folder as your data. If using an IDE you can run the script, then use the output as necessary. Outside an IDE you can import the script - output variables with be preceeded by the script name, e.g `knifeedgeanalysis.prof`. You can also edit and run the script as necessary.

## m2scan

## beamprofiler
