"""

Classes
^^^^^^^

"""

import os

from dateutil.parser import parse

import h5py
import numpy as np

from collections import namedtuple

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from nPDyn.dataParsers.xml_detector_grouping import IN16B_XML



class IN16B_FWS:
    """
        This class can handle raw E/IFWS data from IN16B at the ILL in the hdf5 format.

        :arg scanList:       a string or a list of files to be read and parsed to extract the data.
                             It can be a path to a folder as well.
        :arg sumScans:       whether the scans should be summed or not
        :arg peakFindWindow: the size (in number of channels) of the window to find and align the peaks 
                             of the monitor to the peaks of the data.
        :arg detGroup:       detector grouping, i.e. the channels that are summed over along the
                             position-sensitive detector tubes. It can be an integer, then the same number
                             is used for all detectors, where the integer defines a region (middle of the
                             detector +/- detGroup). It can be a list of integers, then each integers of
                             the list should corresponds to a detector. Or it can be a string, defining
                             a path to an xml file as used in Mantid.
        :arg normalize:      whether the data should be normalized to the monitor

    """

    def __init__(self, scanList, 
                       sumScans=False, 
                       peakFindWindow=6,
                       detGroup=None,
                       normalize=True):


        self.FWSData = namedtuple('FWSData', 'qVals X Y intensities errors temp norm qIdx')
        
        #_Process the scanList argument in case a single string is given
        self.scanList = scanList
        if isinstance(scanList, str):
            if os.path.isdir(scanList):
                if scanList[-1] != '/':
                    scanList = scanList + '/'
                fList = os.listdir(scanList)
                fList.sort()
                self.scanList = [scanList + val for val in fList] 
            else:
                self.scanList = [scanList]
        else:
            self.scanList.sort()


        self.sumScans = sumScans
    
        self.peakFindWindow = peakFindWindow

        self.detGroup = detGroup

        self.normalize = normalize


        self.dataList       = []
        self.errList        = []
        self.monitor        = []
        self.energyList     = []
        self.qList          = []
        self.tempList       = []
        self.startTimeList  = []

        self.outTuple = None




    def process(self):
        """ Extract data from the provided files and reduce them using the given parameters. """

        self.dataList       = []
        self.errList        = []
        self.monitor        = []
        self.energyList     = []
        self.qList          = []
        self.tempList       = []
        self.startTimeList  = []

        for dataFile in self.scanList:

            print("Processing file %s...             " % dataFile, end='\r')

            dataset = h5py.File(dataFile, mode='r')

            data = dataset['entry0/data/PSD_data'][()]

            #_Gets the monitor data
            monitor = dataset['entry0/monitor/data'][()].squeeze().astype('float')
            self.monitor.append(np.copy(monitor))

            #_Sum along the selected region of the tubes
            data = self._detGrouping(data)

            nbrDet = data.shape[0]
            nbrChn = int(data.shape[1])

            maxDeltaE  = dataset['entry0/instrument/Doppler/maximum_delta_energy'][(0)]
            self.energyList.append(maxDeltaE)

            wavelength = dataset['entry0/wavelength'][()]
            angles     = [dataset['entry0/instrument/PSD/PSD angle %s' % int(val+1)][()] 
                                                                    for val in range(nbrDet)]
            angles     = 4*np.pi * np.sin(np.pi * np.array(angles).squeeze() / 360) / wavelength

            temp = dataset['entry0/sample/temperature'][()]

            time = parse(dataset['entry0/start_time'][0])

            self.dataList.append(np.copy(data))
            self.startTimeList.append(time)
            self.qList.append(np.copy(angles))
            self.tempList.append(np.copy(temp))

            dataset.close()




        if self.sumScans:
            sumData = np.zeros_like(self.dataList[0])
            sumMoni = np.zeros_like(self.monitor[0])
            for idx, data in enumerate(self.dataList):
                sumData += data
                sumMoni += self.monitor[idx]

            self.dataList = [sumData]
            self.monitor  = [sumMoni]



        for idx, data in enumerate(self.dataList):

            if self.normalize:
                normData, errData = self._normalizeToMonitor(data, self.monitor[idx]) 
            else:
                normData, errData = (data.sum(1), np.sqrt(data.sum(1))) 


            self.dataList[idx] = normData
            self.errList.append(errData)


        self._convertDataset()
    



    def _convertDataset(self):
        """ Converts the data lists of the class into namedtuple(s) that can be directly used by nPDyn. """

        energies, times, data, errors, temps = self._processEnergyOffsets()

        np.place(errors, errors==0.0, np.inf) 
        np.place(errors, errors==np.nan, np.inf) 
        np.place(data, data/errors < 0.5, 0) 
        np.place(errors, data/errors < 0.5, np.inf) 

        self.outTuple = self.FWSData(self.qList[0], 
                                     energies, 
                                     times,
                                     data,
                                     errors,
                                     temps,
                                     False,
                                     np.arange(self.qList[0].shape[0])) 




    def _normalizeToMonitor(self, data, monitor):
        """ The method finds the peak positions in data array. Then, for each peak, the closest
            peak in monitor array is found using the :arg peakFindWindow: class parameter to
            define the search region.

            The errors are computed by taking the square root of the selected data.
            Finally, both data and errors are divided by the monitor values corresponding to the
            peaks found and summed over the channel axis.
            
        """

        dataList = []
        errList  = []

        threshold = np.sort(monitor)[::-1][:self.peakFindWindow].mean() * 0.8
        peaks     = find_peaks(monitor, threshold=threshold)[0]
        monitor   = monitor[peaks]

        halfBin = int(self.peakFindWindow / 2)

        for idx, qData in enumerate(data):
            qDataPeaks = []
            for peak in peaks:
                qDataPeaks.append( qData[int(np.heaviside(peak-halfBin, 0)):peak+halfBin].max() ) 

            qDataPeaks = np.array(qDataPeaks)
            errors     = np.sqrt(qDataPeaks)

            qDataPeaks /= monitor
            errors     /= monitor

            dataList.append(qDataPeaks.sum())
            errList.append(errors.sum())

            

        return np.array(dataList), np.array(errList)





    def _detGrouping(self, data):
        """ The method performs a sum along detector tubes using the provided range to be kept.
            
            It makes use of the :arg detGroup: argument.

            If the argument is a scalar, it sums over all values that are in the range
            [center of the tube - detGroup : center of the tube + detGroup].

            If the argument is a list of integers, then each element of the list is assumed to correspond
            to a range for each corresponding detector in ascending order.

            If the argument is a mantid-related xml file (a python string), the xml_detector_grouping 
            module is then used to parse the xml file and the provided values are used to define the range. 

            :arg data:  PSD data for the file being processed

        """ 

        if isinstance(self.detGroup, int):
            midPos = int(data.shape[1] / 2)
            data = data[:,midPos - self.detGroup:midPos + self.detGroup,:]
            out  = np.sum(data, 1)
            

        elif isinstance(self.detGroup, (list, tuple, np.ndarray)):
            midPos = int(data.shape[1] / 2)

            out     = np.zeros( (data.shape[0], data.shape[2]) )
            for detId, detData in enumerate(data):
                detData = detData[midPos - self.detGroup[detId]:midPos + self.detGroup[detId]]
                out[detId]     = np.sum(detData, 0)


        elif isinstance(self.detGroup, str):
            numTubes = data.shape[0]
            xmlData  = IN16B_XML(self.detGroup, numTubes)

            detRanges = xmlData.getPSDValues()

            out = np.zeros( (data.shape[0], data.shape[2]) )
            for detId, vals in enumerate(detRanges):
                out[detId]     = data[detId, vals[0]:vals[1]].sum(0)


        elif self.detGroup is None:
            out = np.sum(data, 1)

        return out.astype('float')





    def _processEnergyOffsets(self):
        """ In the case of different sampling for the energy transfers used in FWS data, the
            function interpolates the smallest arrays to produce a unique numpy array of FWS data.

            The method return the unique energy offsets, the time deltas from the first scan,
            the interpolated data and errors, and the temperatures for each energy offset.

        """

        energies = np.unique(self.energyList)
        nbrE = energies.size

        data   = []
        errors = []
        temps  = []
    
        #_Computes the time deltas for each energy offset
        starts    = np.array(self.startTimeList)
        initTime  = self.startTimeList[0]
        deltaTime = []
        idxList   = []
        for dE in energies:
            tmpList   = []
            indices = np.argwhere(np.array(self.energyList) == dE)[:,0]
            idxList.append(indices.astype(int))

            starts[indices] = starts[indices] - initTime
            for sTime in starts[indices]:
                tmpList.append(float(sTime.total_seconds() / 3600))

            deltaTime.append(np.array(tmpList))


        #_Finds the maximum sampling in the list of dataset
        maxSize = 0
        maxX    = None
        for k, time in enumerate(deltaTime):
            if time.shape[0] >= maxSize:
                maxSize = time.shape[0]
                maxX    = time


        #_Performs an interpolation for each dataset with a sampling rate smaller than the maximum
        for idx, dEidx in enumerate(idxList):
            dataAtdE = np.array(self.dataList)[dEidx]
            errAtdE  = np.array(self.errList)[dEidx]
            if dataAtdE.shape[0] < maxSize:
                interpI = interp1d( deltaTime[idx], 
                                    dataAtdE, 
                                    axis=0,
                                    kind='linear', 
                                    fill_value=(dataAtdE[0], dataAtdE[-1]),
                                    bounds_error=False)

                interpErr = interp1d( deltaTime[idx], 
                                      errAtdE, 
                                      axis=0,
                                      kind='linear', 
                                      fill_value=(errAtdE[0], errAtdE[-1]),
                                      bounds_error=False )


                dataAtdE = interpI(maxX)
                errAtdE  = interpErr(maxX)

            data.append(dataAtdE)
            errors.append(errAtdE)
            temps.append(np.array(self.tempList)[dEidx])
            deltaTime[idx] = maxX





        return (energies, 
                np.array(deltaTime), 
                np.array(data).transpose(1,2,0), 
                np.array(errors).transpose(1,2,0), 
                temps)



