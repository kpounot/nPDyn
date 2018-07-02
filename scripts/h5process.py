"""
This script is meant to be used with IN16B data pre-processed (reduction, EC correction and vanadium centering)
with Mantid. It can handle both QENS and FW scans.

For QENS, data can be binned and averaged over all scans. Then the result is stored as a namedtuple
containing several members (all being numpy arrays):
    - qVals         -> list of q values
    - energies      -> list of energy channels
    - intensities   -> 2D array of counts values for each q-value (axis 0) and energy channels (axis 1)
    - errors        -> 2D array of errors values for each q-value (axis 0) and energy channels (axis 1) 
    - temp          -> temperature value  

For FWS, data are stored as list of namedtuple, each corresponding to one energy offset.
They containing several members (all being numpy arrays):
    - deltaE        -> energy offset
    - qVals         -> list of q values
    - intensities   -> 2D array of counts values for each q-value (axis 0) and scan number (axis 1)
    - errors        -> 2D array of errors values for each q-value (axis 0) and scan number (axis 1) 
    - temp          -> temperature value (for time-resolved FWS performed at fixed temperature)
"""

import sys, os
import numpy as np
import h5py as h5
import argParser 
from collections import namedtuple

def processData(dataFiles, binSize, averageFiles=False, FWS=False):

    h5List      = [] # Stores the h5 object containing the pre-processed (reduced with Mantid) data
    dataSetList = [] # Stores each dataSet as a dataBinList corresponding to each files given in dataFiles

    if type(dataFiles) == str:
        dataFiles = [dataFiles]

    for i, dataFile in enumerate(dataFiles):
        h5List.append(h5.File(dataFile))

        if FWS == True:
            FWSData = namedtuple('FWSData', 'deltaE qVals intensities errors temp') 

            wavelength  = h5List[i]['mantid_workspace_1/logs/wavelength/value'].value
            temp        = np.mean(h5List[i]['mantid_workspace_1/logs/sample.temperature/value'].value)

            twoThetaList = h5List[i]['mantid_workspace_1/workspace/axis2'].value
            listQ = 4*np.pi / wavelength * np.sin(np.pi  * twoThetaList / 360)

            dataBinList = [] # Stores the full dataset for a given data file
            for j, workspace in enumerate(h5List[i]):
                listI   = h5List[i][workspace + '/workspace/values'].value
                listErr = h5List[i][workspace + '/workspace/errors'].value

                #_Clean useless values from intensities and errors arrays
                np.place(listErr, listErr==0, np.inf)
                np.place(listErr, listErr==np.nan, np.inf)
                np.place(listI, listI / listErr < 1e-1, 0)
                np.place(listErr, listI / listErr < 1e-1, np.inf)

                deltaE = h5List[i][workspace + '/logs/Doppler.maximum_delta_energy/value'].value

                dataBinList.append( FWSData( deltaE, listQ, listI, listErr, temp ) )

            dataSetList.append(dataBinList)

        else: # Assumes QENS data
            QENSData = namedtuple('QENSData', 'qVals energies intensities errors temp')

            wavelength = h5List[i]['mantid_workspace_1/logs/wavelength/value'].value
            temp        = np.mean(h5List[i]['mantid_workspace_1/logs/sample.temperature/value'].value)

            twoThetaList = h5List[i]['mantid_workspace_1/workspace/axis2'].value
            listQ = 4*np.pi / wavelength * np.sin(np.pi  * twoThetaList / 360)

            listE = h5List[i]['mantid_workspace_1/workspace/axis1'].value[:-1] * 1e3

            listI   = h5List[i]['mantid_workspace_1/workspace/values'].value
            listErr = h5List[i]['mantid_workspace_1/workspace/errors'].value

            dataBinList = [] # Stores the full dataset for a given data file
            binSize = int(binSize)
            for i in range(listI.shape[1]):
                # Bins data using binSize parameter
                if binSize * i + binSize < listI.shape[1] - 1:
                    listE[i]     = np.sum( listE[binSize*i : binSize*i + binSize]     ) / binSize
                    listI[:,i]   = np.sum( listI[:,binSize*i : binSize*i + binSize]  , axis=1 ) / binSize
                    listErr[:,i] = np.sum( listErr[:,binSize*i : binSize*i + binSize], axis=1 ) / binSize

                elif binSize * i + binSize > listI.shape[1] - 1 and binSize * i < listI.shape[1] - 1:
                    listE[i]     = np.sum( listE[binSize*i:] ) / listE[binSize*i:].shape[0]
                    listI[:,i]   = np.sum( listI[:,binSize*i:] , axis=1 ) / listI[:,binSize*i:].shape[1]
                    listErr[:,i] = np.sum( listErr[:,binSize*i:], axis=1 ) / listErr[:,binSize*i:].shape[1]

                else:
                    break

            # Remove unecessary data after binning
            listE   = listE[:int( np.ceil(listE.shape[0] / binSize) )]
            listI   = listI[:,:int( np.ceil(listI.shape[1]   / binSize))] 
            listErr = listErr[:,:int( np.ceil(listErr.shape[1] / binSize))] 

            #_Clean useless values from intensities and errors arrays
            np.place(listErr, listErr==0, np.inf)
            np.place(listErr, listErr==np.nan, np.inf)
            np.place(listI, listI / listErr < 1e-1, 0)
            np.place(listErr, listI / listErr < 1e-1, np.inf)


            dataSetList.append( QENSData(listQ, listE, listI, listErr, temp) )

    if averageFiles and not FWS:
        # Average QENS spectra and remove all dataset but the first one in which average is stored
        dataSetSize = len(dataSetList)
        for i in range(1, len(dataSetList)):
            dataSetList[0] = dataSetList[0]._replace(intensities =  dataSetList[0].intensities 
                                                                    + dataSetList[-i].intensities)
            dataSetList[0] = dataSetList[0]._replace(errors =   dataSetList[0].intensities 
                                                                + dataSetList[-i].errors)
            dataSetList.pop()

        dataSetList[0]._replace(intensities = dataSetList[0].intensities / dataSetSize)
        dataSetList[0]._replace(errors = dataSetList[0].errors / dataSetSize)


    return dataSetList


if __name__ == '__main__':

    arg, karg = argParser.argParser(sys.argv)

    
    dataSetList = processData(arg[1:], int(karg['binS']), avgFiles, FWS)
