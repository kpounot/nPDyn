import sys, os

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

from NAMDAnalyzer.NAMDAnalyzer import NAMDAnalyzer


class MDData(NAMDAnalyzer):
    """ This class wraps the NAMDAnalyzer class. It's initialized with the __init__ method of
        NAMDAnalyzer's Dataset class and the given file list.

        Several methods are available to convert Elastic Incoherent Neutron Scattering or scattering function
        from NAMDAnalyzer to a namedtuple that can directly be used by nPDyn fitting and plotting methods. 
        
        The getTempRampEISF methods need several .dcd files that will be loaded and unloaded one after 
        the other for each temperature. """

    def __init__(self, fileList=None, stride=1):
        super().__init__(fileList, stride)

        self.QENSdataList       = []
        self.FWSdataList        = []
        self.tempRampdataList   = []

        self.MDDataT = namedtuple('MDDataT', 'qVals X intensities errors temp norm qIdx')


    def getTempRampEISF(self, dcdFiles, tempList, qVals, qIdx, binSize=2, converter_kwargs={}):
        """ This method calls the convertScatFunctoEISF for energy space or simply extract the column
            corresponding to the given timeInterval if 'fromTimeSpace' argument is set to True.

            This procedure is done for each dcd file in 'dcdFiles'. It might take some time depending
            on the size of the dcd files. Due to memory limitation, this cannot be done in parallel on most
            computers. This is why the method is implemented like that.

            Finally, all the EISF for each temperature are gathered into a single namedtuple similar to the
            one used in nPDyn.

            This namedtuple can be added to the dataSetList of nPDyn and used as any experimetal file. """


        tempDataSeries  = []

        #_Defining some defaults arguments
        kwargs = {  'qValList'    : qVals,
                    'minFrames'   : 1, 
                    'maxFrames'   : 600, 
                    'nbrBins'     : 80, 
                    'resFunc'     : None, 
                    'selection'   : 'waterH', 
                    'begin'       : 0, 
                    'end'         : None      } 

        #_Modifies default arguments with given ones, if any
        for key, value in converter_kwargs.items():
            kwargs[key] = value

        for dcdFile in dcdFiles:

            self.importFile(dcdFile)
            self.dcdData.binDCD(binSize)
            self.dcdData.compScatteringFunc(**kwargs)

            if fromTimeSpace:
                tempDataSeries.append( self.EISF[0][:,frame].real )

            else:
                tempDataSeries.append( self.dcdData.convertScatFunctoEISF().real )

        intensities = np.array(tempDataSeries).T
        errors      = 1e-6 * np.ones_like(intensities)

        return self.MDDataT( qVals, np.array(tempList), intensities, errors, 0, False, qIdx )



    def getQENS(self, dcdFile, qVals, qIdx, binSize=2, converter_kwargs={}):
        """ This method calls the convertScatFunctoEISF for energy space or simply extract the column
            corresponding to the given timeInterval if 'fromTimeSpace' argument is set to True.

            This procedure is done for each dcd file in 'dcdFiles'. It might take some time depending
            on the size of the dcd files. Due to memory limitation, this cannot be done in parallel on most
            computers. This is why the method is implemented like that.

            Finally, all the EISF for each temperature are gathered into a single namedtuple similar to the
            one used in nPDyn.

            This namedtuple can be added to the dataSetList of nPDyn and used as any experimetal file. """



        #_Defining some defaults arguments
        kwargs = {  'qValList'    : qVals,
                    'minFrames'   : 1, 
                    'maxFrames'   : 600, 
                    'nbrBins'     : 80, 
                    'resFunc'     : None, 
                    'selection'   : 'waterH', 
                    'begin'       : 0, 
                    'end'         : None      } 

        #_Modifies default arguments with given ones, if any
        for key, value in converter_kwargs.items():
            kwargs[key] = value


        self.importFile(dcdFile)
        self.dcdData.binDCD(binSize)
        self.dcdData.compScatteringFunc(**kwargs)

        scatF = self.dcdData.getScatFunc()

        errors      = np.zeros_like(scatF[0])

        return self.MDDataT( qVals, scatF[1], scatF[0], errors, 0, False, qIdx )




    def getMSDfromMD(self, dcdFiles, binSize=1, converter_kwargs={}):
        """ For each dcd file in dcdFiles, import it, compute the MSD directly from the trajectories """

        msdSeries = []

        #_Defining some defaults arguments
        kwargs = {  'frameNbr'    : 100,
                    'selection'   : 'waterH', 
                    'begin'       : 0, 
                    'end'         : None      } 

        #_Modifies default arguments with given ones, if any
        for key, value in converter_kwargs.items():
            kwargs[key] = value


        for dcdFile in dcdFiles:

            self.importFile(dcdFile)
            self.dcdData.binDCD(binSize)
            msdSeries.append( self.dcdData.compMSD(**kwargs) )


        return np.array(msdSeries)


        
