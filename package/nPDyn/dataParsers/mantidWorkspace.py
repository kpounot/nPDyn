import sys, os
import numpy as np
import h5py as h5
from collections import namedtuple

from scipy.interpolate import interp1d

def processData(dataFile, FWS=False, averageTemp=True):
    """ This script is meant to be used with IN16B data pre-processed (reduction, (EC correction) 
            and vanadium centering) with Mantid. 

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
            - X             - energy offset
            - Y             - observable values (time, temperature or other varying parameter)
            - intensities   - 2D array of counts values for each q-value (axis 0) and scan number (axis 1)
            - errors        - 2D array of errors values for each q-value (axis 0) and scan number (axis 1) 
            - temp          - temperature value (for time-resolved FWS performed at fixed temperature) 
            - norm          - boolean, wether data were normalized or not 
            - qIdx          - list of indices of q-values, used for fitting and plotting 

    """

    h5File = h5.File(dataFile, 'r')

    #_Fixed window scan processing
    if FWS == True:
        FWSData = namedtuple('FWSData', 'qVals X Y intensities errors temp norm qIdx') 

        if averageTemp:
            temp    = np.mean(h5File['mantid_workspace_1/logs/sample.temperature/value'][()])
        else:
            temp    = h5File['mantid_workspace_1/logs/sample.temperature/value'][()]

        wavelength  = h5File['mantid_workspace_1/logs/wavelength/value'][()]

        twoThetaList = h5File['mantid_workspace_1/workspace/axis2'][()]
        listQ = 4*np.pi / wavelength * np.sin(np.pi  * twoThetaList / 360)

        dataList = [] # Stores the full dataset for a given data file

        #_Initialize some lists and store energies, intensities and errors in them
        listY    = []
        listI    = []
        listErr  = []
        deltaE   = []
        for j, workspace in enumerate(h5File):
            listY.append(   h5File[workspace + '/workspace/axis1'][()] )
            listI.append(   h5File[workspace + '/workspace/values'][()] )
            listErr.append( h5File[workspace + '/workspace/errors'][()] )
            deltaE.append(  h5File[workspace + '/logs/Doppler.maximum_delta_energy/value'][()] )


        for data in listY:
            if data.shape[0] != listY[0].shape[0]:
                interp = True

        if interp:
            listY, listI, listErr = _interpFWS(listY, listI, listErr)


        #_Converts intensities and errors to numpy and array and transpose to get 
        #_(# frames, # qVals, # energies) shaped array
        listY   = np.array(listY).T
        listI   = np.array(listI).T
        listErr = np.array(listErr).T
        deltaE  = np.array(deltaE)[:,0]

        #_Clean useless values from intensities and errors arrays
        np.place(listErr, listErr==0, np.inf)
        np.place(listErr, listErr==np.nan, np.inf)
        np.place(listI, listI / listErr < 0.5, 0)
        np.place(listErr, listI / listErr < 0.5, np.inf)


        dataList = FWSData( listQ, deltaE, listY, listI, listErr, temp, False, np.arange(listQ.size) )

        return dataList




    #_Assumes QENS data
    else: 
        QENSData = namedtuple('QENSData', 'qVals X intensities errors temp norm qIdx')

        wavelength  = h5File['mantid_workspace_1/logs/wavelength/value'][()]
        temp        = np.mean(h5File['mantid_workspace_1/logs/sample.temperature/value'][()])

        twoThetaList = h5File['mantid_workspace_1/workspace/axis2'][()]
        listQ = 4*np.pi / wavelength * np.sin(np.pi  * twoThetaList / 360)

        listE = h5File['mantid_workspace_1/workspace/axis1'][()][:-1] * 1e3

        listI   = h5File['mantid_workspace_1/workspace/values'][()]
        listErr = h5File['mantid_workspace_1/workspace/errors'][()]


        #_Clean useless values from intensities and errors arrays
        np.place(listErr, listErr==0, np.inf)
        np.place(listErr, listErr==np.nan, np.inf)
        np.place(listI, listI / listErr < 1e-1, 0)
        np.place(listErr, listI / listErr < 1e-1, np.inf)

        dataSet = QENSData(listQ, listE, listI, listErr, temp, False, np.arange(listQ.size))

        return dataSet



def _interpFWS(listX, listI, listErr):
    """ In the case of different sampling for the energy transfers used in FWS data, the
        function interpolates the smallest arrays to produce a unique numpy array of FWS data.

    """

    maxSize = 0
    maxX    = None

    #_Finds the maximum sampling in the list of dataset
    for k, data in enumerate(listX):
        if data.shape[0] >= maxSize:
            maxSize = data.shape[0]
            maxX    = data

    #_Performs an interpolation for each dataset with a sampling rate smaller than the maximum
    for k, data in enumerate(listX):
        if data.shape[0] != maxSize:
            interpI = interp1d( data, 
                                listI[k], 
                                kind='cubic', 
                                fill_value=(listI[k][:,0], listI[k][:,-1]),
                                bounds_error=False )

            interpErr = interp1d( data, 
                                  listErr[k], 
                                  kind='cubic', 
                                  fill_value=(listErr[k][:,0], listErr[k][:,-1]),
                                  bounds_error=False )

            listI[k]   = interpI(maxX)
            listErr[k] = interpErr(maxX)

            data = maxX


    return listX, listI, listErr




if __name__ == '__main__':

    arg, karg = argParser.argParser(sys.argv)

    
    dataSetList = processData(arg[1:], int(karg['binS']), avgFiles, FWS)
