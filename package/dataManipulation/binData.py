import sys
import numpy as np

from collections import namedtuple


def binData(data, binS):
    """ This method takes a dataSet, and return the binned one.
        Attention, this cannot be used with FWS data processed from hdf5 files.

        Input:  data    -> dataSet to be binned
                binS    -> bin size to reduce the dataset """

    binS = int(binS)

    #_Compute the loop number needed for binning
    loopSize = int(np.ceil(data.intensities.shape[1] / binS))

    #_Get the arrays to be binned
    tempX           = data.X
    tempIntensities = data.intensities
    tempErrors      = data.errors

    #_Iterate binning, basically computes values mean inside a bin, then shift to the next bin 
    for i in range(loopSize):
            tempX[i]             = np.mean(tempX[i*binS:i*binS+binS]) 
            tempIntensities[:,i] = np.mean(tempIntensities[:,i*binS:i*binS+binS], axis=1)  
            tempErrors[:,i]      = np.mean(tempErrors[:,i*binS:i*binS+binS], axis=1) 


    # Remove unecessary data after binning
    data = data._replace(  X           = tempX[:loopSize],
                           intensities = tempIntensities[:,:loopSize], 
                           errors      = tempErrors[:,:loopSize]) 

    return data

