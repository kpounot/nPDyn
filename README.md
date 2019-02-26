Work still ongoing, not all features are working properly or are present at the moment.

##Installation/Usage:
Just extract the archive in the wanted directory.
Then, the package can be directly imported in python or a session can be started with IPython using the following:

    ipython -i <npdyn __main__ path> -- [kwargs]

###Keywords arguments on main call:

--QENS, -q            - import a list of QENS data files
--FWS, -f             - import a list of FWS data files
--TempRamp, -tr       - import a list of temperature remp data files
--resolution, -res    - import a list of resolution function data files
--empty-cell, -ec     - import an empty cell data file
--D2O, -d             - import a D2O signal data file

Starting the API using this method creates a instance of Dataset class in a variable called 'data'.
