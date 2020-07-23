"""

Classes
^^^^^^^

"""

import os

import h5py
import numpy as np

from collections import namedtuple

from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit, basinhopping

from nPDyn.dataParsers.xml_detector_grouping import IN16B_XML



class IN16B_QENS:
    """
        This class can handle raw QENS data from IN16B at the ILL in the hdf5 format.

        :arg scanList:          a string or a list of files to be read and parsed to extract the data.
                                It can be a path to a folder as well.
        :arg sumScans:          whether the scans should be summed or not.
        :arg unmirroring:       whether the data should be unmirrored or not.
        :arg vanadiumRef:       if :arg unmirroring: is True, then the peaks positions are identified 
                                using the data provided with this argument. If it is None, then the peaks
                                positions are identified using the data in scanList.
        :arg detGroup:          detector grouping, i.e. the channels that are summed over along the
                                position-sensitive detector tubes. It can be an integer, then the same number
                                is used for all detectors, where the integer defines a region (middle of the
                                detector +/- detGroup). It can be a list of integers, then each integers of
                                the list should corresponds to a detector. Or it can be a string, defining
                                a path to an xml file as used in Mantid.
        :arg normalize:         whether the data should be normalized to the monitor
        :arg strip:             an integer defining the number of points that are ignored at each 
                                extremity of the spectrum.

    """

    def __init__(self, scanList, 
                       sumScans=True, 
                       unmirroring=True, 
                       vanadiumRef=None,
                       peakFindingMask=None,
                       detGroup=None,
                       normalize=True,
                       strip=10):


        self.QENSData = namedtuple('QENSData', 'qVals X intensities errors temp norm qIdx')
        
        #_Process the scanList argument in case a single string is given
        self.scanList = scanList
        if isinstance(scanList, str):
            if os.path.isdir(scanList):
                fList    = os.listdir(scanList)
                self.scanList = [scanList + val for val in fList] 
            else:
                self.scanList = [scanList]


        self.sumScans = sumScans
    
        self.peaks = None
        self.unmirroring = unmirroring

        #_Process the vanadiumRef argument in case a single string is given
        self.vanadiumRef = vanadiumRef
        if isinstance(scanList, str):
            if os.path.isdir(scanList):
                fList    = os.listdir(scanList)
                scanList = [scanList.strip('/') + '/' + val for val in fList] 
            else:
                scanList = [scanList]


        self.peakFindingMask   = peakFindingMask

        self.detGroup = detGroup

        self.normalize = normalize

        self.strip = strip

        self.dataList   = []
        self.errList    = []
        self.energyList = []
        self.qList      = []
        self.tempList   = []

        self.outTuple = None




    def process(self):
        """ Extract data from the provided files and reduce them using the given parameters. """

        self.dataList   = []
        self.energyList = []
        self.qList      = []
        self.tempList   = []


        for dataFile in self.scanList:
            dataset = h5py.File(dataFile, mode='r')

            data = dataset['entry0/data/PSD_data'][()]
            self.monitor = dataset['entry0/monitor/data'][()].squeeze().astype('float')
            np.place(self.monitor, self.monitor==0, np.inf)

            #_Sum along the selected region of the tubes
            data    = self._detGrouping(data)

            nbrDet = data.shape[0]
            nbrChn = int(data.shape[1] / 2)

            maxDeltaE  = dataset['entry0/instrument/Doppler/maximum_delta_energy'][()]
            energies   = 2 * np.arange(nbrChn) * (maxDeltaE / nbrChn) - maxDeltaE

            wavelength = dataset['entry0/wavelength'][()]
            angles     = [dataset['entry0/instrument/PSD/PSD angle %s' % int(val+1)][()] 
                                                                    for val in range(nbrDet)]
            angles     = 4*np.pi * np.sin(np.pi * np.array(angles).squeeze() / 360) / wavelength

            temp = dataset['entry0/sample/temperature'][()]

            self.dataList.append(np.copy(data))
            self.energyList.append(np.copy(energies))
            self.qList.append(np.copy(angles))
            self.tempList.append(np.copy(temp))

            dataset.close()
        

        if self.sumScans:
            self.monitor  = len(self.dataList) * self.monitor
            self.dataList = [np.sum(np.array(self.dataList), 0)]

        for idx, data in enumerate(self.dataList):
            if self.unmirroring:
                if self.vanadiumRef is not None:
                    vana = IN16B_QENS(self.vanadiumRef,
                                      detGroup=self.detGroup,
                                      peakFindingMask=self.peakFindingMask) 
                    vana.process()
                    self.leftPeak  = vana.leftPeak
                    self.rightPeak = vana.rightPeak
                    data = self._unmirrorData(data)
                    self.monitor = self._unmirrorData(self.monitor.reshape(1, 2*nbrChn))
                else:
                    self._findPeaks(data)
                    data = self._unmirrorData(data)
                    self.monitor = self._unmirrorData(self.monitor.reshape(1, 2*nbrChn))

                errData = np.sqrt(data)

            else:
                errData = np.sqrt(data)


            if self.normalize:
                data    = data / self.monitor
                errData = errData / self.monitor

            np.place(errData, errData==0.0, np.inf)
            np.place(errData, errData==np.nan, np.inf)
            np.place(data, data / errData < 0.1, 0)
            np.place(errData, data / errData < 0.1, np.inf)

            self.dataList[idx] = data
            self.errList.append(errData)


        self._convertDataset()



    def _convertDataset(self):
        """ Converts the data lists of the class into namedtuple(s) that can be directly used by nPDyn. """

        self.outTuple = []
        for idx, data in enumerate(self.dataList):
            self.outTuple.append(self.QENSData(self.qList[idx], 
                                 self.energyList[idx][self.strip:-self.strip], 
                                 data[:,self.strip:-self.strip], 
                                 self.errList[idx][:,self.strip:-self.strip],
                                 self.tempList[idx],
                                 False,
                                 np.arange(self.qList[idx].shape[0])))





    def _findPeaks(self, data):
        """ Find the peaks in each subset of mirrored data using the selected method. """

        nbrChannels = data.shape[1]
        midChannel  = int(nbrChannels / 2)

        if self.peakFindingMask is None:
            mask = np.zeros_like(data)
            mask[:,int(midChannel/4):int(3*midChannel/4)]   = 1
            mask[:,int(5*midChannel/4):int(7*midChannel/4)] = 1

        maskedData = data * mask

        #_Finds the peaks using a Savitsky-Golay filter to smooth the data, followed by extracting 
        #_the position of the maximum
        filters  = np.array([savgol_filter(maskedData, 5, 4), 
                             savgol_filter(maskedData, 11, 4),
                             savgol_filter(maskedData, 19, 3),
                             savgol_filter(maskedData, 25, 3)])


        savGol_leftPeak = np.mean( np.argmax(filters[:,:,:midChannel], 2), 0)
        savGol_rightPeak = np.mean( np.argmax(filters[:,:,midChannel:], 2), 0)


        #_Finds the peaks by using a Gaussian function to fit the data
        Gaussian = lambda x, normF, gauW, shift, bkgd: (
                        normF * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi)) + bkgd )

        leftPeaks  = []
        rightPeaks = []
        for qIdx, qData in enumerate(maskedData):
            errors = np.sqrt(qData)
            np.place(errors, errors==0, np.inf)
            params = curve_fit( Gaussian, 
                                np.arange(midChannel), 
                                qData[:midChannel],
                                sigma=errors[:midChannel],
                                p0=[qData[:midChannel].max(),1, midChannel/2, 0] )
            leftPeaks.append(params[0][2])
            
            params = curve_fit( Gaussian, 
                                np.arange(midChannel), 
                                qData[midChannel:],
                                sigma=errors[midChannel:],
                                p0=[qData[midChannel:].max(), 1, midChannel/2, 0] )
            rightPeaks.append(params[0][2])


        gauss_leftPeak  = np.array(leftPeaks)
        gauss_rightPeak = np.array(rightPeaks)


        self.leftPeak  = (0.85*gauss_leftPeak + 0.15*savGol_leftPeak).astype(int)
        self.rightPeak = (0.85*gauss_rightPeak + 0.15*savGol_rightPeak).astype(int)








    def _unmirrorData(self, data):
        """ If unmirroring is required, unmirror data using the peaks found. Basically, the 
            two mirror dataset are aligned on their respective peaks and summed.

            It should be called after the summation on vertical positions of the PSD was performed.

        """

        nbrChannels = data.shape[-1]
        midChannel  = int(nbrChannels / 2)

        out = np.zeros_like(data)
        for qIdx, qOut in enumerate(out):
            leftPos  = midChannel - self.leftPeak[qIdx]
            rightPos = midChannel - self.rightPeak[qIdx]

            qOut[leftPos:leftPos+midChannel]   += data[qIdx,:midChannel] 
            qOut[rightPos:rightPos+midChannel] += data[qIdx,midChannel:] 


        data = out[:,int(midChannel / 2):int(midChannel / 2) + midChannel]

        return data



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
