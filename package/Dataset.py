import sys, os
import numpy as np
import h5py as h5
import argparse
import re

from collections import namedtuple

from . import fileFormatParser
from .dataTypes import *
from .dataTypes.models import *
from .fit import *
from .plot import *

from .MDParser import MDData
from .plot.plotMD_MSD import plotMSDSeries


class Dataset:
    """ Master class of nPDyn, contains a list of dataFiles, which can be sample, resolution, D2O data or
        anything else as long as the format can be recognized. 

        For call with ipython, the dataSet can be initialized directly from command line using the 
        following: 'ipython -i dataSet.py -- [dataSet related optional arguments]'
        
        Input:  QENSFiles   ->  list of Quasi-Elastic Neutron Scattering data files to be loaded (optional)
                FWSFiles    ->  list of Fixed Window Scans data files to be loaded (optional)
                TempRampFiles   ->  list of temperature ramps data files to be loaded (optional)
                ECFile      -> Empty cell data to be loaded (optional)
                resFiles    -> list of resolution function related data to be loaded
                D2OFiles    -> list of D2O data to be loaded

        """

    def __init__(self, QENSFiles=None, FWSFiles=None, TempRampFiles=None, ECFile=None, resFiles=None, 
                                                                                            D2OFile=None):


        #_Declaring attributes related to samples dataset
        self.dataSetList    = []

        #_Declaring resolution function(s) related variables
        self.resData    = []

        #_Declaring D2O lineshape related variables
        self.D2OData    = None

        #_Declaring empty cell signal related variables
        self.ECData     = None

        self.msdSeriesList      = []    #_Used to store msd from trajectories, present here to allow for
                                        #_different psf files loading


        self.modelList = {  1:resFunc_pseudoVoigt.Model,
                            2:resFunc_gaussian.Model,
                            3:D2OFunc_singleLorentzian_BH.Model,
                            4:QENS_prot_powder_doubleLorentzian_BH.Model,
                            5:QENS_water_powder_BH.Model,
                            6:TempRamp_gaussian.Model                     }


        
        
        #_Processing files arguments
        if resFiles: 
            for f in resFiles:
                data = resType.ResType(f)
                data.importData()
                data = resFunc_pseudoVoigt.Model(data)
                data.fit()

                self.resData.append( data )


        if D2OFile:
            data = D2OType.D2OType(D2OFile)
            data.importData()
            data = D2OFunc_singleLorentzian_BH.Model(data)

            if self.resData != []:
                data.assignResData(self.resData[0])
                data.normalize()
                data.qWisefit()

            self.D2OData = data


        if ECFile: 
            data = ECType.ECType(ECFile)
            data.importData()
            data = resFunc_pseudoVoigt.Model(data)
            data.fit()

            self.ECData = data


        if QENSFiles: 
            for f in QENSFiles:
                data = QENSType.QENSType(f) 
                data.importData()
                data.assignD2OData( self.D2OData )
                data.assignECData( self.ECData )
                self.dataSetList.append( data )

        if FWSFiles: 
            for f in FWSFiles:
                data = FWSType.FWSType(f)
                data.importData()
                data.assignD2OData( self.D2OData )
                data.assignECData( self.ECData )
                self.dataSetList.append( data )

        if TempRampFiles: 
            for f in TempRampFiles:
                data = TempRampType.TempRampType(f)
                data.importData()
                data.assignD2OData( self.D2OData )
                data.assignECData( self.ECData )
                self.dataSetList.append( data )



        self.resFuncAssign()



    def resFuncAssign(self):
        """ This method is used during initialization, and can be used each time resolution functions
            need to be re-assigned to data (after a model change for instance).
            If only one resolution function was provided, it will assume that the same one has to be used
            for all QENS and FWS data loaded. Therefore, the same resolution function data will be assigned
            to all experimental data.

            If the number of resolution data is the same as the number of experimental data (QENS or FWS), then
            they are assigned in an order-wise manner. 

            If none of the above conditions are fulfilled, nothing is done and resolution data should be 
            assigned manually. """

        lenResData = len(self.resData)

        if lenResData == 1:
            for data in self.dataSetList:
                data.assignResData(self.resData[0])


        if lenResData == len(self.dataSetList):
            for idx, data in enumerate(self.dataSetList):
                data.assignResData(self.resData[idx])





#--------------------------------------------------
#_Importation, reset and deletion methods
#--------------------------------------------------
    def importFile(self, fileFormat=None, fileType='QENS'):
        """ Read and import data from experimental data file.
            If no file format is given, this method tries to identify the file's type automatically, 
            and send an error message in case the file could not be imported. 
            It can be used to import .inx file or QENS/FWS from hdf5 files for now.

            Input:  dataFile    -> path of the data file to be imported
                    fileFormat  -> format of the file to be imported (inx, hdf5,...)
                    fileType    -> string that specifies the type of data to be imported, possible values are 
                                   QENS, FWS, TempRamp, res, EC, D2O (optional, default QENS)

            
            The files are imported without binning. """

        #_Use given file format. If None, try to guess
        if fileFormat:
            data = fileFormatParser.readFile(fileFormat, dataFile, FWS)
        else:
            data = fileFormatParser.guessFileFormat(dataFile, FWS)

        data = ResType(dataFile, data)
        self.resData.append(data)



    def removeDataSet(self, fileIdx):
        """ This method takes either a single integer as argument, which corresponds
            to the index of the file to be removed from self.dataFiles, self.dataSetList and 
            self.rawDataList. """

        self.dataSetList.pop(fileIdx)


    def resetDataSet(self, *fileIdxList):
        """ This method takes either a single integer or a list of integer as argument. They correspond
            to the indices of the file to be reset to their initial state using self.rawDataList. """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        for idx in fileIdxList:
            self.dataSetList[idx].resetData()
 

#--------------------------------------------------
#_Molecular dynamics simulation data related methods
#--------------------------------------------------

    def initMD(self, psfFile, stride=1):
        """ Initialize a NAMDAnalyzer instance with the given psf file. """

        try:
            self.MDData.dcdData.dcdData = None #_Free memory
            self.MDData.psfData         = None #_Free memory
        except AttributeError:
            pass


        self.MDData = MDData(psfFile, stride)



    def importTempRampMD(self, dcdFiles, tempList, dataSetIdx=0, binSize=1, converter_kwargs={}):
        """ Imports data from MD simulation using the getTempEISF method in MDParser class. 
        
            Input:  dcdFiles    -> list of dcd files corresponding to each temperature
                    tempList    -> list of temperatures used, should be in the same order as dcd files
                    dataSetIdx  -> experimental dataset index to be used as reference for q-values and indices
                                   (optional, default 0)
                    converter_kwargs -> arguments to be passed to scattering function computation methods
                                        in NAMDAnalyzer package. (optional, some defaults value but better use
                                        it explicitly. 
                                        
            Stores the resulting namedtuple in self.dataSetList as any other experimental data. """

        qVals = self.dataSetList[0].data.qVals
        qIdx  = self.dataSetList[0].data.qIdx

        MDTuple = namedtuple('MDDataT', 'qVals X intensities errors temp norm qIdx')
        MDDataT = self.MDData.getTempRampEISF(dcdFiles, tempList, qVals, qIdx, binSize, converter_kwargs)
            
        self.dataSetList.append( TempRampType.TempRampType( self.MDData.psfFile, 
                                                            MDDataT,
                                                            MDDataT,
                                                            self.D2OData,
                                                            self.ECData ) )


        self.dataSetList[-1].assignResData(self.resData[0]) #_Assign first resolution function by default

        #_Free up memory
        self.MDData.dcdData.dcdData = None


    def importQENS_MD(self, dcdFile, dataSetIdx=0, binSize=1, resBkgdIdx=None, converter_kwargs={}):
        """ Imports data from MD simulation using the getTempEISF method in MDParser class. 
        
            Input:  dcdFiles    -> list of dcd files corresponding to each temperature
                    tempList    -> list of temperatures used, should be in the same order as dcd files
                    dataSetIdx  -> experimental dataset index to be used as reference for q-values and indices
                                   (optional, default 0)
                    resBkgdIdx  -> if not None, should be an integer giving the index of resolution data
                                    in self.resData, from which background signal intensity should be
                                    extracted (optional, default None - no experimental background added)
                    converter_kwargs -> arguments to be passed to scattering function computation methods
                                        in NAMDAnalyzer package. (optional, some defaults value but better use
                                        it explicitly. 
                                        
            Stores the resulting namedtuple in self.dataSetList as any other experimental data. """

        qVals = self.dataSetList[0].data.qVals
        qIdx  = self.dataSetList[0].data.qIdx

        MDTuple = namedtuple('MDDataT', 'qVals X intensities errors temp norm qIdx')
        MDDataT = self.MDData.getQENS(dcdFile, qVals, qIdx, binSize, converter_kwargs)


        #_Gets fitted instrumental background and add it to simulated data
        if resBkgdIdx is not None:
            bkgd = np.array( [data[0][-1] for data in self.resData[resBkgdIdx].params] )[:, np.newaxis]
            
            MDDataT = MDDataT._replace(intensities = MDDataT.intensities + bkgd)

            
        self.dataSetList.append( QENSType.QENSType( self.MDData.psfFile, 
                                                            MDDataT,
                                                            MDDataT,
                                                            self.D2OData,
                                                            self.ECData ) )


        self.dataSetList[-1].assignResData(self.resData[0]) #_Assign first resolution function by default

        #_Free up memory
        self.MDData.dcdData.dcdData = None



    def plotMSDfromMD(self, tempList, msdIdxList, *fileIdxList):
        """ Plot the given computed MSD alongwith given files in fileIdxList.

            Input:  tempList    -> temperature list, same order as in the msd list
                    msdIdxList  -> indices for msd series in self.MDData.msdSeriesList
                    fileIdxList -> indices for files in self.fileIdxList to be plotted (optional, default all)
                    """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        datasetList     = [dataset for i, dataset in enumerate(self.dataSetList) if i in fileIdxList] 
        msdSeriesList   = [msd for i, msd in enumerate(self.msdSeriesList) if i in msdIdxList] 

        plotW = plotMSDSeries(msdSeriesList, tempList, datasetList)
        plotW.show()



    def getMSDfromMD(self, dcdFiles, binSize=1, converter_kwargs={}):
        """ For each dcd file in dcdFiles, import it, compute the MSD directly from the trajectories """

        self.msdSeriesList.append(self.MDData.getMSDfromMD(dcdFiles, binSize, converter_kwargs))


#--------------------------------------------------
#_Data manipulation methods
#--------------------------------------------------
    def normalize_usingResFunc(self, *fileIdxList):
        """ This method uses the fitted normalization factor (normF) to normalize each dataSet in
            fileIdxList.

            If only one resolution data file is loaded, use this one for all dataset, else, use them in
            the same order as dataset in self.dataSetList

            Input: fileIdxList -> list of indices of dataset in self.dataSetList to be normalized """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        for i in fileIdxList:
            self.dataSetList[i].normalize()



    
    def normalize_ENS_usingLowTemp(self, *fileIdxList, nbrBins=5):
        """ This method is meant to be used only with elastic temperature ramp. For which the given
            number of first bins (low temperature) are used to compute an average signal at low 
            temperature. The average is then used to normalize the whole dataset. This for each q-value.

            Input:  fileIdxList ->  can be "all", then every dataSet in self.dataSetList is normalized
                                    can also be a single integer or a list of integer (optional, default "all")
                    nbrBins     ->  number of low temperature bins used to compute the normalization factor
                                    (optional, default 5) """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        for i in fileIdxList:
            self.dataSetList[i].normalize_usingLowTemp(nbrBins)





    def substract_EC(self, subFactor, *fileIdxList):
        """ This method uses the fitted empty cell function to substract the signal for the selected
            dataSet. 

            Input: subFactor    -> pre-multiplying factor for empty cell data prior to substraction
                   fileIdxList  -> can be "all", then every dataSet in self.dataSetList is normalized
                                can also be a single integer or a list of integer (optional, default "all") """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        #_Apply corrections for samples data
        for i in fileIdxList:
            self.dataSetList[i].substractEC(subFactor)

 



    def discardDetectors(self, decIdxList, *fileIdxList):
        """ Remove data corresponding to given detectors/q-values
            The process modifies dataset.qIdx attributes, that is used for sample QENS fitting and plotting. """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        for idx in fileIdxList:
            self.dataSetList[idx].discardDetectors(*decIdxList)




    def resetDetectors(self, *fileIdxList):
        """ Reset qIdx entry to its original state, with all q values taken into account. """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        for idx in fileIdxList:
            self.dataSetList[idx].data = self.dataSetList[idx]._replace(qIdx = 
                                            [idx for idx, val in enumerate(self.dataSetList[idx].qVals)])

            


    def assignModeltoData(self, model, *fileIdxList):
        """ Helper function to quickly assign the given model to all dataset with given indices in
            self.dataSetList. If model is not None, the decorator pattern is used to modify the dataType
            class behavior. """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))


        for idx in fileIdxList:
            self.dataSetList[idx] = model( self.dataSetList[idx] )





    def fitData(self, *fileIdxList, p0=None, bounds=None, qWise=False):
        """ Helper function to quickly call fit method in all class instances present in self.dataSetList
            for the given indices in fileIdxList.
            Check first for the presence of a fit method and print a warning message if none is found. """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))


        for idx in fileIdxList:
            try:
                if qWise:
                    self.dataSetList[idx].qWiseFit(p0=p0, bounds=bounds)
                else:
                    self.dataSetList[idx].fit(p0=p0, bounds=bounds)

            except AttributeError:
                print("No fitting function was found for dataType at index %i in dataSetList.\n" % idx
                        + "Please assign a model to it prior to call a fitting method.\n")




#--------------------------------------------------
#_Binning methods
#--------------------------------------------------
    def binDataSet(self, binS, *fileIdxList):
        """ The method find the index corresponding to the file, perform the binning process,
            then replace the value in self.dataSetList by the binned one. 
            
            Input: binS         -> bin size
                   fileIdxList  -> indices of the dataSet to be binned, can be a single int or a list of int 
                                   (optional, default "all") """ 


        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        #_Calling binData for each dataSet in fileIdxList
        for idx in fileIdxList:
            self.dataSetList[idx].binData(binS)




    def binResData(self, binS):
        """ Same as binDataSet but for resolution function data. """ 
        
        for data in self.resData:
            data.binData(binS)
 


    def binAll(self, binS):
        """ Bin all dataSet in dataSetList as well as resolutio, empty cell and D2O data if present. """ 
        
        self.binDataSet(binS) #_Bin the dataset list

        #_For other types of data, check if something was loaded, and if so, perform the binning
        if self.resData:
            self.binResData(binS)

        if self.ECData:
            self.ECData.binData(binS)

        if self.D2OData:
            self.D2OData.binData(binS)





#--------------------------------------------------
#_Resolution function related methods
#--------------------------------------------------
    def plotResFunc(self):
        """ This method plots the resolution function.
            A PyQt window is showed with different data representation possibilities. """

        plotW = resPlot.ResPlot(self.resData)
        plotW.show()



#--------------------------------------------------
#_Empty cell data related methods
#--------------------------------------------------
    def plotECFunc(self):
        """ This method plots the empty cell lineshape fitted function.
            A PyQt window is showed with different data representation possibilities. """

        plotW = ECPlot.ECPlot([self.ECData])
        plotW.show()
        


#--------------------------------------------------
#_D2O signal related methods (liquid state)
#--------------------------------------------------
    def plotD2OFunc(self):
        """ This method plots the resolution function.
            A PyQt window is showed with different data representation possibilities. """

        plotW = D2OPlot.D2OPlot([self.D2OData])
        plotW.show()
 


#--------------------------------------------------
#_Plotting methods
#--------------------------------------------------
    def plotQENS(self, *fileIdxList):
        """ This methods plot the sample data in a PyQt5 widget allowing the user to show different
            types of plots. 

            The resolution function and other parameters are automatically obtained from the current
            dataSet class instance. 
            
            Input:  fileIdxList -> indices of dataset to be plotted (optional, default "all") """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        datasetList = [dataset for i, dataset in enumerate(self.dataSetList) if i in fileIdxList]

        plotW = QENSPlot.QENSPlot(datasetList)
        
        plotW.show()




    def plotTempRampENS(self, *fileIdxList):
        """ This methods plot the sample data in a PyQt5 widget allowing the user to show different
            types of plots. 

            The resolution function and other parameters are automatically obtained from the current
            dataSet class instance. 
            
            Input:  fileIdxList -> indices of dataset to be plotted (optional, default "all")
                    powder      -> whether the sample is a powder or in liquid state (optional, default True) 
                    qDiscardList-> integer or list if indices corresponding to q-values to discard """

        #_If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataSetList))

        datasetList = [dataset for i, dataset in enumerate(self.dataSetList) if i in fileIdxList] 

        plotW = TempRampPlot.TempRampPlot(datasetList)
        plotW.show()





if __name__ == '__main__':

    #_Defining options for nPDyn call
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--QENS", nargs='*',  
                    help="List of files corresponding to sample Quasi-Elastic Neutron Scattering (QENS) data")
    parser.add_argument("-f", "--FWS", nargs='*', 
                    help="List of files corresponding to sample Fixed-Window Scan (FWS) data")
    parser.add_argument("-tr", "--TempRamp", nargs='*', 
                help="List of files corresponding to temperature ramp elastic data")
    parser.add_argument("-res", "--resolution", nargs='*', 
                                    help="Specify the file(s) to be used for resolution function fitting.")
    parser.add_argument("-ec", "--empty-cell", nargs='?',
                                    help="Specify the file containing QENS empty cell data")
    parser.add_argument("-d", "--D2O", nargs='?', help="Specify the file containing QENS D2O data")


    args = parser.parse_args()

    data = dataSet(args.QENS, args.FWS, args.TempRamp, args.empty_cell, args.resolution, args.D2O)
