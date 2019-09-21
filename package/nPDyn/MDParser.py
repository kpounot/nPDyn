import sys, os

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from nPDyn.dataTypes import *
from nPDyn.dataTypes.models import *
from nPDyn.plot.plotMD_MSD import plotMSDSeries


#_Try to import NAMDAnalyzer, print a warning message in case it cannot be found
try:
    from NAMDAnalyzer.Dataset import Dataset as MDDataset
    from NAMDAnalyzer.dataAnalysis.backscatteringDataConvert import BackScatData
except ImportError:
    class MDDataset: pass
    class BackScatData: pass

    print("\nNAMDAnalyzer (github.com/kpounot/NAMDAnalyzer) was not installed "
           + "within your python framework.\n" 
           + "MD simulations related methods won't work.\n")


class MDData(MDDataset, BackScatData):
    """ This class wraps the NAMDAnalyzer Dataset class. It's initialized with the __init__ method of
        the class and the given file list.

        Several methods are available to convert Elastic Incoherent Neutron Scattering or scattering function
        from NAMDAnalyzer to a namedtuple that can directly be used by nPDyn fitting and plotting methods. 
        
        The getTempRampEISF methods need several .dcd files that will treated sequentially to extract
        mean-squared displacements. """

    def __init__(self, expData, fileList, stride=1):
        MDDataset.__init__(self, fileList, stride=stride)
        BackScatData.__init__(self, self)

        self.expData = expData
        self.msdSeriesList = []

        self.MDDataT = namedtuple('MDDataT', 'qVals X intensities errors temp norm qIdx')





    def initMD(self, psfFile, stride=1):
        """ Initialize a NAMDAnalyzer instance with the given psf file. """


        self.importFile(psfFile)
        self.stride = stride



#--------------------------------------------------
#_Methods to obtain nPDyn like data types that are append to experimental data list
#--------------------------------------------------
    def getTempRampEISF(self, dcdFiles, tempList, dataSetIdx=0, resBkgdIdx=None, bkgdIdx=None, 
                        converter_kwargs={}):
        """ Calls ``compScatteringFunc`` from NAMDAnalyzer for all given dcdFiles, extracts the EISF and
            stores values in a data tuple that can be used directly in nPDyn.

            :arg dcdFiles:         list of dcd files corresponding to each temperature
            :arg tempList:         list of temperatures used, should be in the same order as dcd files
            :arg dataSetIdx:       experimental dataset index to be used as reference for q-values and indices
                                    (optional, default 0)
            :arg resBkgdIdx:       index of experimental resolution data from which background parameter
                                    should be extracted
            :arg converter_kwargs: arguments to be passed to scattering function computation methods
                                    in NAMDAnalyzer package. (optional, some defaults value but better use
                                    it explicitly. 
                                

            The result is added to *dataSetList* in :class:`Dataset` class. 


            For NAMDAnalyzer BackScatData module, see:
            https://namdanalyzer.readthedocs.io/en/latest/dataAnalysis/BackscatteringDataConvert

        """

        qIdx  = self.expData.dataSetList[dataSetIdx].data.qIdx
        qVals = self.expData.dataSetList[dataSetIdx].data.qVals[qIdx]

        tempDataSeries  = []

        #_Defining some defaults arguments
        kwargs = { 'qValList': qVals }

        #_Modifies default arguments with given ones, if any
        for key, value in converter_kwargs.items():
            kwargs[key] = value

        for dcdFile in dcdFiles:

            self.importFile(dcdFile)
            self.compScatteringFunc(**kwargs)

            tempDataSeries.append( self.convertScatFunctoEISF().real )

        intensities = np.array(tempDataSeries).T
        errors      = 1e-6 * np.ones_like(intensities)

        dataTuple = self.MDDataT( qVals, np.array(tempList).astype('f'), intensities, errors, 0, False,
                                                                                np.arange(qVals.size) )



        #_Gets fitted instrumental background and add it to simulated data
        if resBkgdIdx is not None:
            bkgd = np.array( [data[0][-1] for data 
                             in self.expData.resData[resBkgdIdx].params] )[qIdx, np.newaxis]
            
            dataTuple = dataTuple._replace(intensities = dataTuple.intensities + bkgd)

        #_Gets fitted dataset background and add it to simulated data
        if bkgdIdx is not None:
            bkgd = np.array( [self.expData.dataSetList[bkgdIdx].getBackground(qIdx) 
                for qIdx, val in enumerate(self.expData.dataSetList[bkgdIdx].data.qIdx)] )[:,np.newaxis]

            MDDataT = MDDataT._replace(intensities = MDDataT.intensities + bkgd)


        #_Appends computed temp ramp to dataSetList 
        self.expData.dataSetList.append( TempRampType.TempRampType( self.psfFile, 
                                                            dataTuple,
                                                            dataTuple ) )





    def getQENS(self, dcdFile, dataSetIdx=0, resBkgdIdx=None, bkgdIdx=None,
                converter_kwargs={}):
        """ This method calls the ``compScatteringFunc`` from NAMDAnalyzer package.

            :arg dcdFile:          list of file path to be used to compute QENS spectra
            :arg dataSetIdx:       index of experimental dataset to be used to extract q-values
            :arg resBkgdIdx:       index of resolution data to be used for background
            :arg bkgdIdx:          similar to resBkgdIdx, but using sample data in dataSetList
            :arg converter_kwargs: arguments to be given to NAMDAnalyzer compScatFunc method 

            The result is added to nPDyn *dataSetList* attribute in :class:`Dataset` class.

            For NAMDAnalyzer BackScatData module, see:
            https://namdanalyzer.readthedocs.io/en/latest/dataAnalysis/BackscatteringDataConvert

        """
        



        qIdx  = self.expData.dataSetList[dataSetIdx].data.qIdx
        qVals = self.expData.dataSetList[dataSetIdx].data.qVals[qIdx]

        #_Defining some defaults arguments
        kwargs = { 'qValList': qVals }

        #_Modifies default arguments with given ones, if any
        for key, value in converter_kwargs.items():
            kwargs[key] = value


        self.importFile(dcdFile)
        self.compScatteringFunc(**kwargs)

        scatF = self.scatFunc
        errors = np.zeros_like(scatF[0])
        MDDataT = self.MDDataT( qVals, scatF[1], scatF[0], errors, 0, False, np.arange(qVals.size) )



        #_Gets fitted instrumental background and add it to simulated data
        if resBkgdIdx is not None:
            bkgd = np.array( [data[0][-1] for data 
                             in self.expData.resData[resBkgdIdx].params] )[qIdx, np.newaxis]
            
            MDDataT = MDDataT._replace(intensities = MDDataT.intensities + bkgd)

        #_Gets fitted dataset background and add it to simulated data
        if bkgdIdx is not None:
            bkgd = np.array( [self.expData.dataSetList[bkgdIdx].getBackground(qIdx) 
                for qIdx, val in enumerate(self.expData.dataSetList[bkgdIdx].data.qIdx)] )[:,np.newaxis]

            MDDataT = MDDataT._replace(intensities = MDDataT.intensities + bkgd)

            

        self.expData.dataSetList.append( QENSType.QENSType( self.psfFile, 
                                                            MDDataT,
                                                            MDDataT ) )





    def getMSDfromMD(self, dcdFiles, converter_kwargs={}):
        """ For each dcd file in dcdFiles, import it, compute the MSD directly from the trajectories. 

            This makes use of the ``compMSD`` method from NAMDAnalyzer BackScatData module

            For NAMDAnalyzer BackScatData module, see:
            https://namdanalyzer.readthedocs.io/en/latest/dataAnalysis/BackscatteringDataConvert

        """

        msdSeries = []

        #_Defining some defaults arguments
        kwargs = {  'frameNbr'    : 100,
                    'selection'   : 'waterH', 
                    'frames'      : slice(0, None, 1) }

        #_Modifies default arguments with given ones, if any
        for key, value in converter_kwargs.items():
            kwargs[key] = value


        for dcdFile in dcdFiles:

            self.importFile(dcdFile)
            self.compMSD(**kwargs) 
            msdSeries.append( self.MSD )


        self.msdSeriesList.append( np.array(msdSeries) )



#--------------------------------------------------
#_Plotting method 
#--------------------------------------------------
    def plotMSDfromMD(self, tempList, msdIdxList, *fileIdxList):
        """ Plot the given computed MSD along with given files in fileIdxList.

            :arg tempList:    temperature list, same order as in the msd list
            :arg msdIdxList:  indices for msd series in self.MDData.msdSeriesList
            :arg fileIdxList: indices for files in self.fileIdxList to be plotted (optional, default all)
                    
        """

        datasetList     = [dataset for i, dataset in enumerate(self.expData.dataSetList) if i in fileIdxList] 
        msdSeriesList   = [msd for i, msd in enumerate(self.msdSeriesList) if i in msdIdxList] 

        plotW = plotMSDSeries(msdSeriesList, tempList, datasetList)
        plotW.show()

