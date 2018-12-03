import sys
import numpy as np
import inxConvertQENS
import argParser
from collections import namedtuple


def inxBin(dataFile, binS):
    """ This method takes a single file as argument.
        Then, data are binned using the given binS parameter as bin size.
        
        Input:  dataFiles   -> data file or list of data files
                binS        -> bin size to reduce the dataset """

    #_Exctract data from the .inx file
    dataSet = inxConvertQENS.convert(dataFile)

    binS = int(binS)
    
    #_Compute the loop number needed for binning
    loopSize = np.ceil(dataSet.intensities.shape[1])
    for i in range(loopSize):
        dataSet._replace(   energies    = np.sum(dataSet.energies[i*binS:i*binS+binS]) / binS,
                            intensities = np.sum(dataSet.intensities[:,i*binS:i*binS+binS], axis=1) / binS, 
                            errors      = np.sum(dataSet.errors[:,i*binS:i*binS+binS], axis=1) / binS)

    # Remove unecessary data after binning
    dataSet = dataSet._replace(   energies    = dataSet.energies[:loopSize*binS + binS],
                        intensities = dataSet.intensities[:,:loopSize*binS + binS], 
                        errors      = dataSet.errors[:,:loopSize*binS + binS]) 

    return dataSet

