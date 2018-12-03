import sys
import numpy as np
from collections import namedtuple

def convert(datafile):
    """ This method take a single dataFile as argument and return the corresponding dataSet.

        The dataSet is a namedtuple containing the following entries:
            qVals       -> list of momentum transfer (q) values (in inverse Angström)
            X           -> list of energy transfers (in micro eV, for QENS) or other x-axis parameter
                           (e.g. temperature for elastic temperature ramp)
            intensities -> 2D numpy array with energy transfer dependant scattering intensities (axis 1),
                           for each q-value (axis 0)
            errors      -> same as intensities but this one contains the experimental errors
            temp        -> not used with .inx files, present here for consistency with hdf5 files 
            norm        -> boolean, wether data were normalized or not 
            qIdx        -> list of indices of q-value, used for fitting and plotting """


    datafile = datafile

    qData = namedtuple('qData', 'qVals X intensities errors temp norm qIdx')

    with open(datafile, 'r') as fileinput:
        data = fileinput.read().splitlines()

    #_Get the index of the data corresponding to each q-value
    indexList = []
    for i, value in enumerate(data):
        if value == data[0]:
            indexList.append(i)


    #_Create namedtuple containing lists of q-values, energies, intensities, errors and temperature
    data        = [val.split() for i, val in enumerate(data)]
    qVals       = []
    intensities = []
    X           = []
    errors      = []
    for i, value in enumerate(indexList):
        if i < len(indexList)-1: #_While we're not at the last angle entries
            qVals.append(data[value+2][0]) 
            X.append([val[0] for j, val in enumerate(data) if indexList[i] + 3 < j < indexList[i+1]])
            intensities.append([val[1] for j, val in enumerate(data) if indexList[i] + 3 < j < indexList[i+1]])
            errors.append([val[2] for j, val in enumerate(data) if indexList[i] + 3 < j < indexList[i+1]])

        else: #_Used for the last entries
            qVals.append(data[value+2][0]) 
            X.append([val[0] for j, val in enumerate(data) if j > indexList[i]+3])
            intensities.append([val[1] for j, val in enumerate(data) if j > indexList[i]+3])
            errors.append([val[2] for j, val in enumerate(data) if j > indexList[i]+3])

    #_Converting to floats
    qVals       = np.array(qVals).astype(float)
    X           = np.array(X[0]).astype(float)
    intensities = np.array(intensities).astype(float)
    errors      = np.array(errors).astype(float)

    #_Clean useless values from intensities and errors arrays
    np.place(errors, errors==0, np.inf)
    np.place(errors, errors==np.nan, np.inf)
    np.place(intensities, intensities / errors < 0.5, 0)
    np.place(errors, intensities / errors < 0.5, np.inf)
 
    #_Creating the named tuple (no temp with .inx)
    dataSet = qData(qVals, X, intensities, errors, None, False, np.arange(qVals.size)) 

    return dataSet    

