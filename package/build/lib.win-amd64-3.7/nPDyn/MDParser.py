import sys, os

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from .dataTypes import *
from .dataTypes.models import *
from .plot.plotMD_MSD import plotMSDSeries


#_Try to import NAMDAnalyzer, print a warning message in case it cannot be found
try:
    from NAMDAnalyzer.Dataset import Dataset as MDDataset
    from NAMDAnalyzer.dataAnalysis.backscatteringDataConvert import BackScatData
except ImportError:
    raise ImportError("\nNAMDAnalyzer (github.com/kpounot/NAMDAnalyzer) was not installed "
                        + "within your python framework.\n" 
                        + "MD simulations related methods won't work.\n")


class MDData(MDDataset, BackScatData):
    """ This class wraps the NAMDAnalyzer class. It's initialized with the __init__ method of
        NAMDAnalyzer's Dataset class and the given file list.

        Several methods are available to convert Elastic Incoherent Neutron Scattering or scattering function
        from NAMDAnalyzer to a namedtuple that can directly be used by nPDyn fitting and plotting methods. 
        
        The getTempRampEISF methods need several .dcd files that will be loaded and unloaded one after 
        the other for each temperature. """

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
    def getTempRampEISF(self, dcdFiles, tempList, dataSetIdx=0, resBkgdIdx=None, converter_kwargs={}):
        """ Calls compScatteringFunc from NAMDAnalyzer for all given dcdFiles, extracts the EISF and
            stores values in a data tuple that can be used directly in nPDyn.

            Input:  dcdFiles    -> list of dcd files corresponding to each temperature
                    tempList    -> list of temperatures used, should be in the same order as dcd files
                    dataSetIdx  -> experimental dataset index to be used as reference for q-values and indices
                                   (optional, default 0)
                    resBkgdIdx  -> index of experimental resolution data from which background parameter
                                    should be extracted
                    converter_kwargs -> arguments to be passed to scattering function computation methods
                                        in NAMDAnalyzer package. (optional, some defaults value but better use
                                        it explicitly. 
                                        

            This namedtuple can be added to the dataSetList of nPDyn and used as any experimetal file. """

        qVals = self.expData.dataSetList[dataSetIdx].data.qVals

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
            bkgd = np.array( [data[0][-1] for data in self.expData.resData[resBkgdIdx].params] )[:, np.newaxis]
            
            dataTuple = dataTuple._replace(intensities = dataTuple.intensities + bkgd)



        #_Appends computed temp ramp to dataSetList 
        self.expData.dataSetList.append( TempRampType.TempRampType( self.psfFile, 
                                                            dataTuple,
                                                            dataTuple ) )





    def getQENS(self, dcdFile, dataSetIdx=0, resBkgdIdx=None, converter_kwargs={}):
        """ This method calls the convertScatFunctoEISF for energy space.

            This procedure is done for each dcd file in 'dcdFiles'. It might take some time depending
            on the size of the dcd files. Due to memory limitation, this cannot be done in parallel on most
            computers. This is why the method is implemented like that.

            Finally, all the EISF for each temperature are gathered into a single namedtuple similar to the
            one used in nPDyn.

            This namedtuple can be added to the dataSetList of nPDyn and used as any experimental file. 
            
            Input:  dcdFile             -> list of file path to be used to compute QENS spectra
                    dataSetIdx          -> index of experimental dataset to be used to extract q-values
                    resBkgdIdx          -> index of resolution data to be used for background
                    converter_kwargs    -> arguments to be given to NAMDAnalyzer compScatFunc method """
        



        qVals = self.expData.dataSetList[dataSetIdx].data.qVals

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
            bkgd = np.array( [data[0][-1] for data in self.expData.resData[resBkgdIdx].params] )[:, np.newaxis]
            
            MDDataT = MDDataT._replace(intensities = MDDataT.intensities + bkgd)

            

        self.expData.dataSetList.append( QENSType.QENSType( self.psfFile, 
                                                            MDDataT,
                                                            MDDataT ) )





    def getMSDfromMD(self, dcdFiles, converter_kwargs={}):
        """ For each dcd file in dcdFiles, import it, compute the MSD directly from the trajectories """

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
        """ Plot the given computed MSD alongwith given files in fileIdxList.

            Input:  tempList    -> temperature list, same order as in the msd list
                    msdIdxList  -> indices for msd series in self.MDData.msdSeriesList
                    fileIdxList -> indices for files in self.fileIdxList to be plotted (optional, default all)
                    """

        datasetList     = [dataset for i, dataset in enumerate(self.expData.dataSetList) if i in fileIdxList] 
        msdSeriesList   = [msd for i, msd in enumerate(self.msdSeriesList) if i in msdIdxList] 

        plotW = plotMSDSeries(msdSeriesList, tempList, datasetList)
        plotW.show()

