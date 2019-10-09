.. nPDyn documentation master file, created by
   sphinx-quickstart on Fri Sep 20 17:40:40 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nPDyn
=====

Python based API for analysis of neutron backscattering data.



Installation:
-------------
nPDyn can make use of NAMDAnalyzer, which can be obtained at:
http://github.com/kpounot/NAMDAnalyzer


Prior to building, the path to Gnu Scientific Library (GSL) should be given in setup.cfg file (required by libabsco)

If not, the package can still be installed but paalman-ping corrections won't work


Unix and Windows
^^^^^^^^^^^^^^^^

For installation within your python framework, use:

::

    make 
    make install

or

::

    python setup.py build
    python setup.py install


Use with ipython
----------------

The package can be directly imported in python or a session can be started with IPython using the following:

::

    ipython -i <npdyn __main__ path> -- [kwargs]


This can be used when python or ipython is called within the folder where __main__ is located.

Options:
    - -q, --QENS,                 - import a list of QENS data files
    - -f, --FWS,                  - import a list of FWS data files
    - -tr, --TempRamp,            - import a list of temperature remp elastic data files
    - -res, --resolution,         - import a list of resolution function data files
    - -ec, --empty-cell,          - import an empty cell data file
    - -fec, --fixed-empty-cell,   - import fixed-window empty cell measurement
    - -tec, --TempRamp-empty-cell - import temperature ramp elastic empty cell measurement
    - -d, --D2O,                  - import a D2O signal data file
    - -fd, --fixed-D2O,           - import a fixed-window D2O signal data file

Starting the API using this method creates a instance of Dataset class in a variable called 'data'.



Reference 
---------

.. toctree::
   :maxdepth: 2

   dataset
   MDParser
   dataManipulation/index
   dataParsers/index
   dataTypes/index
   fileFormatParser
   fit/index
   plot/index
   license
   help


Quick start
-----------

.. include:: ./quickstart.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
