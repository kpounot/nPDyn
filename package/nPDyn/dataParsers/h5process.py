import sys, os
import numpy as np
import h5py as h5
from collections import namedtuple

def processData(dataFile, FWS=False, averageTemp=True):
    """ This script is meant to be used with IN16B data pre-processed 
        (reduction, (EC correction) and vanadium centering) with Mantid. 

        It can handle both QENS and fixed-window scans.

        For QENS, data can be binned and averaged over all scans. Then the result is stored as a namedtuple
        containing several members (all being numpy arrays):
            - qVals         - list of q values
            - X             - list of energy channels
            - intensities   - 2D array of counts values for each q-value (axis 0) and energy channels (axis 1)
            - errors        - 2D array of errors values for each q-value (axis 0) and energy channels (axis 1) 
            - temp          - temperature value  
            - norm          - boolean, wether data were normalized or not
            - qIdx          - list of indices of q-values, used for fitting and plotting

        For FWS, data are stored as list of namedtuple, each corresponding to one energy offset.
        They containing several members (all being numpy arrays):
            - qVals         - list of q values
            - deltaE        - energy offset
            - intensities   - 2D array of counts values for each q-value (axis 0) and scan number (axis 1)
            - errors        - 2D array of errors values for each q-value (axis 0) and scan number (axis 1) 
            - temp          - temperature value (for time-resolved FWS performed at fixed temperature) 
            - norm          - boolean, wether data were normalized or not 
            - qIdx          - list of indices of q-values, used for fitting and plotting 

    """

    h5File = h5.File(dataFile)

    #_Fixed window scan processing
    if FWS == True:
        FWSData = namedtuple('FWSData', 'qVals X intensities errors temp norm qIdx') 

        if averageTemp:
            temp    = np.mean(h5File['mantid_workspace_1/logs/sample.temperature/value'].value)
        else:
            temp    = h5File['mantid_workspace_1/logs/sample.temperature/value'].value

        wavelength  = h5File['mantid_workspace_1/logs/wavelength/value'].value

        twoThetaList = h5File['mantid_workspace_1/workspace/axis2'].value
        listQ = 4*np.pi / wavelength * np.sin(np.pi  * twoThetaList / 360)

        dataList = [] # Stores the full dataset for a given data file

        #_Initialize some lists and store energies, intensities and errors in them
        listI    = []
        listErr  = []
        deltaE   = []
        for j, workspace in enumerate(h5File):
            listI.append(   h5File[workspace + '/workspace/values'].value )
            listErr.append( h5File[workspace + '/workspace/errors'].value )
            deltaE.append(  h5File[workspace + '/logs/Doppler.maximum_delta_energy/value'].value )

        #_Converts intensities and errors to numpy and array and transpose to get 
        #_(# frames, # qVals, # energies) shaped array
        listI   = np.array(listI).T
        listErr = np.array(listErr).T
        deltaE  = np.array(deltaE)[:,0]

        #_Clean useless values from intensities and errors arrays
        np.place(listErr, listErr==0, np.inf)
        np.place(listErr, listErr==np.nan, np.inf)
        np.place(listI, listI / listErr < 0.5, 0)
        np.place(listErr, listI / listErr < 0.5, np.inf)


        dataList = FWSData( listQ, deltaE, listI, listErr, temp, False, np.arange(listQ.size) )

        return dataList

    #Þ_Assumes QENS data
    else: 
        QENSData = namedtuple('QENSData', 'qVals X intensities errors temp norm qIdx')

        wavelength  = h5File['mantid_workspace_1/logs/wavelength/value'].value
        temp        = np.mean(h5File['mantid_workspace_1/logs/sample.temperature/value'].value)

        twoThetaList = h5File['mantid_workspace_1/workspace/axis2'].value
        listQ = 4*np.pi / wavelength * np.sin(np.pi  * twoThetaList / 360)

        listE = h5File['mantid_workspace_1/workspace/axis1'].value[:-1] * 1e3

        listI   = h5File['mantid_workspace_1/workspace/values'].value
        listErr = h5File['mantid_workspace_1/workspace/errors'].value


        #_Clean useless values from intensities and errors arrays
        np.place(listErr, listErr==0, np.inf)
        np.place(listErr, listErr==np.nan, np.inf)
        np.place(listI, listI / listErr < 1e-1, 0)
        np.place(listErr, listI / listErr < 1e-1, np.inf)

        dataSet = QENSData(listQ, listE, listI, listErr, temp, False, np.arange(listQ.size))

        return dataSet



if __name__ == '__main__':

    arg, karg = argParser.argParser(sys.argv)

    
    dataSetList = processData(arg[1:], int(karg['binS']), avgFiles, FWS)
