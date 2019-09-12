Work still ongoing, not all features are working properly or are present at the moment.

## Installation/Usage:
nPDyn makes use of NAMDAnalyzer, which can be obtained at github.com/kpounot/NAMDAnalyzer

Extract the archive in the wanted directory


Prior to building, the path to Gnu Scientific Library (GSL) should be given in setup.cfg file
(required by libabsco)


### Unix and Windows
For installation within your python framework, use:

    make 
    make install

or

    python setup.py build
    python setup.py install


### Use with ipython
The package can be directly imported in python or a session can be started with IPython using the following:

    ipython -i <npdyn __main__ path> -- [kwargs]


### Keywords arguments on main call:
This can be used when python or ipython is called within the folder where __main__ is located.

- --QENS, -q            - import a list of QENS data files
- --FWS, -f             - import a list of FWS data files
- --TempRamp, -tr       - import a list of temperature remp data files
- --resolution, -res    - import a list of resolution function data files
- --empty-cell, -ec     - import an empty cell data file
- --D2O, -d             - import a D2O signal data file

Starting the API using this method creates a instance of Dataset class in a variable called 'data'.
