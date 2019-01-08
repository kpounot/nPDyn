import sys, os
import numpy as np
import h5py as h5
import argparse
import re

from collections import namedtuple

from . import fileFormatParser
from .dataTypes import *
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


        #_Processing files arguments
        if QENSFiles: 
            for f in QENSFiles:
                data = QENSType.QENSType(f) 
                data.importData()
                self.dataSetList.append( data )

        if FWSFiles: 
            for f in FWSFiles:
                data = FWSType.FWSType(f)
                data.importData()
                self.dataSetList.append( data )

        if TempRampFiles: 
            for f in TempRampFiles:
                data = TempRampType.TempRampType(f)
                data.importData()
                self.dataSetList.append( data )

        if ECFile: 
            pass

        if resFiles: 
            for f in resFiles:
                data = resType.ResType(f)
                data.importData()
                data = resFunc_pseudoVoigt.ResFunc_pseudoVoigt(data)
                data.fit()

                self.resData.append( data )

        if D2OFile:
            pass




#--------------------------------------------------
#_Importation, reset and deletion methods
#--------------------------------------------------
    def importQENS(self, dataFile, fileFormat=None):
        """ Read and import data from experimental data file.
            If no file format is given, this method tries to identify the file's type automatically.

            Input:  dataFile    -> path of the data file to be imported
                    fileFormat  -> format of the file to be imported (inx, hdf5,...)
            
            The files are imported without binning. """

        #_Use given file format. If None, try to guess
        if fileFormat:
            data = fileFormatParser.readFile(fileFormat, dataFile, False)
        else:
            data = fileFormatParser.guessFileFormat(dataFile, False)

        #_Add data file name to the list
        self.dataFiles.append(dataFile)

        data = QENSType(dataFile, data)
        self.dataSetList.append(data)



    def importFWS(self, dataFile, fileFormat=None):
        """ Read and import data from experimental data file.
            If no file format is given, this method tries to identify the file's type automatically.

            Input:  dataFile    -> path of the data file to be imported
                    fileFormat  -> format of the file to be imported (inx, hdf5,...)
            
            The files are imported without binning. """

        #_Use given file format. If None, try to guess
        if fileFormat:
            data = fileFormatParser.readFile(fileFormat, dataFile, True)
        else:
            data = fileFormatParser.guessFileFormat(dataFile, True)

        #_Add data file name to the list
        self.dataFiles.append(dataFile)

        data = FWSType(dataFile, data)
        self.dataSetList.append(data)



    def importTempRamp(self, dataFile, fileFormat=None):
        """ Read and import data from experimental data file.
            If no file format is given, this method tries to identify the file's type automatically.

            Input:  dataFile    -> path of the data file to be imported
                    fileFormat  -> format of the file to be imported (inx, hdf5,...)
            
            The files are imported without binning. """

        #_Use given file format. If None, try to guess
        if fileFormat:
            data = fileFormatParser.readFile(fileFormat, dataFile, True)
        else:
            data = fileFormatParser.guessFileFormat(dataFile, True)

        #_Add data file name to the list
        self.dataFiles.append(dataFile)

        data = TempRampType(dataFile, data)
        self.dataSetList.append(data)




    def removeDataSet(self, fileIdx):
        """ This method takes either a single integer as argument, which corresponds
            to the index of the file to be removed from self.dataFiles, self.dataSetList and 
            self.rawDataList. """

        self.dataFiles.pop(fileIdx)
        self.dataSetList.pop(fileIdx)
        self.rawDataList.pop(fileIdx)
        self.paramsList.pop(fileIdx)
        self.modelList.pop(fileIdx)


    def resetDataSet(self, *fileIdxList):
        """ This method takes either a single integer or a list of integer as argument. They correspond
            to the indices of the file to be reset to their initial state using self.rawDataList. """

        for idx in fileIdxList:
            self.dataSetList[idx]   = self.dataSetList[idx]._replace(
                                                qVals       = np.copy(self.rawDataList[idx].qVals),
                                                X           = np.copy(self.rawDataList[idx].X),
                                                intensities = np.copy(self.rawDataList[idx].intensities),
                                                errors      = np.copy(self.rawDataList[idx].errors),
                                                temp        = np.copy(self.rawDataList[idx].temp),
                                                norm        = False,
                                                qIdx        = np.copy(self.rawDataList[idx].qIdx) )
            self.paramsList[idx]    = []
            self.modelList[idx]     = []
 

#--------------------------------------------------
#_Molecular dynamics simulation data related methods
#--------------------------------------------------

    def initMD(self, psfFile):
        """ Initialize a NA.MDAnalyzer instance with the given psf file. """

        self.MDData = None #_Free memory
        self.MDData = MDData(psfFile)



    def importTempRampMD(self, dcdFiles, tempList, dataSetIdx=0, binSize=1, fromTimeSpace=False, frame=100, 
                                                                                    converter_kwargs={}):
        """ Imports data from MD simulation using the getTempEISF method in MDParser class. 
        
            Input:  dcdFiles    -> list of dcd files corresponding to each temperature
                    tempList    -> list of temperatures used, should be in the same order as dcd files
                    dataSetIdx  -> experimental dataset index to be used as reference for q-values and indices
                                   (optional, default 0)
                    fromTimeSpace    -> if true, use directly the EISF at given frame instead of scattering func
                                     (optional, default False)
                    frame            -> frame to be used with fromTimeSpace (optional, default 100)
                    converter_kwargs -> arguments to be passed to scattering function computation methods
                                        in NAMDAnalyzer package. (optional, some defaults value but better use
                                        it explicitly. 
                                        
            Stores the resulting namedtuple in self.dataSetList as any other experimental data. """

        qVals = self.dataSetList[0].qVals
        qIdx  = self.dataSetList[0].qIdx

        MDTuple = namedtuple('MDDataT', 'qVals X intensities errors temp norm qIdx')
        MDDataT = self.MDData.getTempRampEISF(dcdFiles, tempList, qVals, qIdx, binSize, fromTimeSpace, frame,
                                                                                        converter_kwargs)
            
        self.dataSetList.append( MDDataT )
        self.rawDataList.append( MDTuple(**MDDataT._asdict() ))
        self.modelList.append([])
        self.paramsList.append([])
        self.paramsNames.append([])
        self.dataFiles.append('MD data')

        #_Free up memory
        self.MDData.dcdData.dcdData = None


    def plotMSDfromMD(self, tempList, msdIdxList, fileIdxList=[]):
        """ Plot the given computed MSD alongwith given files in fileIdxList.

            Input:  tempList    -> temperature list, same order as in the msd list
                    msdIdxList  -> indices for msd series in self.MDData.msdSeriesList
                    fileIdxList -> indices for files in self.fileIdxList to be plotted (optional, default all)
                    """

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        #_Conversion to a list, if fileIdxList is a single integer
        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        if type(msdIdxList) == int:
            msdIdxList = [msdIdxList]


        plotW = plotMSDSeries(self, msdIdxList, tempList, fileIdxList)
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

        for i in fileIdxList:
            #_Check for q-values to be the same between data and resolution
            normFList = []
            if len(self.resData) == 1:
                for qIdx, qVal in enumerate(self.resData[0].qVals):
                    if qVal in self.dataSetList[i].qVals:
                        normFList.append(self.resParams[0][qIdx][0][0])
            else:
                for qIdx, qVal in enumerate(self.resData[i].qVals):
                    if qVal in self.dataSetList[i].qVals:
                        normFList.append(self.resParams[i][qIdx][0][0])


    
    def normalize_ENS_usingLowTemp(self, fileIdxList="all", nbrBins=5):
        """ This method is meant to be used only with elastic temperature ramp. For which the given
            number of first bins (low temperature) are used to compute an average signal at low 
            temperature. The average is then used to normalize the whole dataset. This for each q-value.

            Input:  fileIdxList ->  can be "all", then every dataSet in self.dataSetList is normalized
                                    can also be a single integer or a list of integer (optional, default "all")
                    nbrBins     ->  number of low temperature bins used to compute the normalization factor
                                    (optional, default 5) """

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        #_Conversion to a list, if fileIdxList is a single integer
        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        for i in fileIdxList:
            #_Computing the normalization factors
            normFList = np.mean(self.dataSetList[i].data.intensities[:,:nbrBins], axis=1)





    def substract_EC(self, subFactor=0.8, fileIdxList="all", D2OData=False, resData=False):
        """ This method uses the fitted empty cell function to substract the signal for the selected
            dataSet. 

            Input: subFactor    -> pre-multiplying factor for empty cell data prior to substraction
                                   (optional, default 0.9)
                   fileIdxList  -> can be "all", then every dataSet in self.dataSetList is normalized
                                  can also be a single integer or a list of integer (optional, default "all")
                   D2OData      -> if True, performs the substraction in D2O data as well 
                                  (optional, default False)
                   resData      -> it True, performs the substraction for resolution data as well
                                  (optional, default False)"""

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        #_Conversion to a list, if fileIdxList is a single integer
        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        #_Apply corrections for D2O data and resolution data if arguments set to True
        if resData:
            for idx, data in enumerate(self.resData):
                ECFunc    = np.row_stack((self.ECFunc(data.X, *params[0]) 
                                                                for params in self.ECParams)) 
                data = data._replace(intensities = data.intensities - ECFunc)

                #_Clean useless values from intensities and errors arrays
                S       = data.intensities
                errors  = data.errors
                np.place(errors, S < 0, np.inf)
                np.place(S, S < 0, 0)
                data = data._replace(intensities = S)
                data = data._replace(errors = errors)
 
        if D2OData:
            ECFunc    = np.row_stack((self.ECFunc(self.D2OData.X, *params[0]) 
                                                            for params in self.ECParams)) 
            self.D2OData = self.D2OData._replace(intensities = self.D2OData.intensities - ECFunc)

            #_Clean useless values from intensities and errors arrays
            S       = self.D2OData.intensities
            errors  = self.D2OData.errors
            np.place(errors, S < 0, np.inf)
            np.place(S, S < 0, 0)
            self.D2OData = self.D2OData._replace(intensities = S)
            self.D2OData = self.D2OData._replace(errors = errors)
 
        #_Apply corrections for samples data
        for i in fileIdxList:
            X = self.dataSetList[i].X

            ECFunc    = np.row_stack((self.ECFunc(X, *params[0]) 
                                                            for params in self.ECParams)) 

            if self.dataSetList[i].norm: #_If data are normalized, use same normalization on EC data as well
                if len(self.resData) == 1:
                    normFList = [params[0][0] for params in self.resParams[0]]
                else:
                    normFList = [params[0][0] for params in self.resParams[i]]
                ECFunc = np.apply_along_axis(lambda arr: arr / normFList, 0, ECFunc)

            #_Substracting the empty cell intensities from experimental data
            self.dataSetList[i] = self.dataSetList[i]._replace(intensities = 
                    self.dataSetList[i].intensities - ECFunc)

            #_Clean useless values from intensities and errors arrays
            S       = self.dataSetList[i].intensities
            errors  = self.dataSetList[i].errors
            np.place(errors, S < 0, np.inf)
            np.place(S, S < 0, 0)
            self.dataSetList[i] = self.dataSetList[i]._replace(intensities = S)
            self.dataSetList[i] = self.dataSetList[i]._replace(errors = errors)
 

    def discardDetectors(self, qIdx, fileIdxList='all'):
        """ Remove data corresponding to given detectors/q-values
            The process modifies dataset.qIdx attributes, that is used for sample QENS fitting and plotting. """

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        if isinstance(fileIdxList, int):
            fileIdxList = [fileIdxList]

        if isinstance(qIdx, int):
            qIdx = [qIdx]


        for idx in fileIdxList:
            self.dataSetList[idx] = self.dataSetList[idx]._replace(qIdx = 
                                            [val for val in self.dataSetList[idx].qIdx if val not in qIdx])


    def resetDetectors(self, fileIdxList='all'):
        """ Reset qIdx entry to its original state, with all q values taken into account. """

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        if isinstance(fileIdxList, int):
            fileIdxList = [fileIdxList]

        for idx in fileIdxList:
            self.dataSetList[idx] = self.dataSetList[idx]._replace(qIdx = 
                                            [idx for idx, val in enumerate(self.dataSetList[idx].qVals)])

            

#--------------------------------------------------
#_Binning methods
#--------------------------------------------------
    def binDataSet(self, binS, fileIdxList="all"):
        """ The method find the index corresponding to the file, perform the binning process,
            then replace the value in self.dataSetList by the binned one. 
            
            Input: binS         -> bin size
                   fileIdxList  -> indices of the dataSet to be binned, can be a single int or a list of int 
                                   (optional, default "all") """ 

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        #_Conversion to a list, if fileIdxList is a single integer
        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        #_Calling binData for each dataSet in fileIdxList
        for idx in fileIdxList:
            self.dataSetList[idx] = binData.binData(self.dataSetList[idx], binS)

    def binResData(self, binS, fileIdxList="all"):
        """ Same as binDataSet but for resolution function data. """ 
        
        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.resData))

        #_Conversion to a list, if fileIdxList is a single integer
        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        for idx in fileIdxList:
            self.resData[idx] = binData.binData(self.resData[idx], binS)
 
    def binD2OData(self, binS):
        """ Same as binDataSet but for D2O data. """ 
        
        self.D2OData = binData.binData(self.D2OData, binS)
 
    def binECData(self, binS):
        """ Same as binDataSet but for empty cell data. """ 
        
        self.ECData = binDatabinData(self.ECData, binS)

    def binAll(self, binS):
        """ Bin all dataSet in dataSetList as well as resolutio, empty cell and D2O data if present. """ 
        
        self.binDataSet(binS) #_Bin the dataset list

        #_For other types of data, check if something was loaded, and if so, perform the binning
        if self.resData:
            self.binResData(binS)

        if self.ECData:
            self.ECData = binData(self.ECData, binS)

        if self.D2OData:
            self.D2OData = binData(self.D2OData, binS)





#--------------------------------------------------
#_Resolution function related methods
#--------------------------------------------------
    def importResData(self, resFile):
        """ Read and import data from experimental data file.
            If no file format is given, this method tries to identify the file's type automatically, 
            and send an error message in case the file could not be imported. 
            It can be used to import .inx file or QENS from hdf5 files for now.

            Input:  dataFile    -> path of the data file to be imported
                    fileFormat  -> format of the file to be imported (inx, hdf5,...)
                    fileType    -> string that specifies the type of data to be imported 
                                   (QENS, FWS, TempRamp - optional, default QENS)

            
            The files are imported without binning. """

        #_Use given file format. If None, try to guess
        if fileFormat:
            data = fileFormatParser.readFile(fileFormat, dataFile, FWS)
        else:
            data = fileFormatParser.guessFileFormat(dataFile, FWS)

        data = ResType(dataFile, data)
        self.resData.append(data)


    def fitRes_gaussianModel(self):
        """ Fits the self.resData using a two gaussians model and the scipy's curve_fit procedure.
            Then store the result in self.resParams variable. """

        for val in self.resData:
            self.resParams.append(resFit_gaussianModel.resFit(val))
            self.resFunc.append(resFit_gaussianModel.resFunc)
            self.resPNames.append(["normF", "S", "g0", "g1", "shift", "bkgd"])

    def fitRes_pseudoVoigtModel(self):
        """ Fits the self.resData using a pseudo-Voigt profile and the scipy's curve_fit procedure.
            Then store the result in self.resParams variable. """

        for val in self.resData:
            self.resParams.append(resFit_pseudoVoigtModel.resFit(val))
            self.resFunc.append(resFit_pseudoVoigtModel.resFunc)
            self.resPNames.append(["normF", "S", "lorW", "gauW", "shift", "bkgd"])

    def plotResFunc(self):
        """ This method plots the resolution function.
            A PyQt window is showed with different data representation possibilities. """

        plotW = resPlot.ResPlot(self)
        plotW.show()






#--------------------------------------------------
#_Empty cell data related methods
#--------------------------------------------------
    def importECData(self, ECFile):
        """ This method tries to identify the file's type automatically, and send an error message 
            in case the file could not be imported. 
            It can be used to import empty cell QENS .inx or hdf5 file.

            For FWS sample data, the QENS empty cell fitted function's value at the given energy
            offset is simply used for substraction.
            
            The files are imported without binning. """

        try:
            if re.search('.inx', ECFile):
                self.ECData    = inxConvert.convert(ECFile) 
            elif re.search('.nxs', ECFile):
                self.ECData    = h5process.processData(ECFile) 
            else: #_Try a generic hdf5 file (.h5 extension or else)
                self.ECData    = h5process.processData(ECFile) 

        except Exception as err:
            print("Error of type %s, at line %s in %s" % ( sys.exc_info()[0], 
                                                          sys.exc_info()[2].tb_lineno, 
                                                          sys.exc_info()[2].tb_frame.f_code.co_filename))


    def fitEC_pseudoVoigtModel(self):
        """ Fits the self.ECData using a pseudo-Voigt profile and the scipy's curve_fit procedure.
            Then store the result in self.ECParams variable. 
            
            Pseudo-Voigt profile appear to be quite good in fitting empty cell signal, but in case
            the user wants to use another function, a method dedicated to that will be provided. """

        self.ECParams = ECFit_pseudoVoigtModel.ECFit(self.ECData)
        self.ECFunc   = ECFit_pseudoVoigtModel.fitFunc


    def plotECFunc(self):
        """ This method plots the empty cell lineshape fitted function.
            A PyQt window is showed with different data representation possibilities. """

        plotW = ECPlot.ECPlot(self.ECData, self.ECFunc, self.ECParams)
        plotW.show()
        





#--------------------------------------------------
#_D2O signal related methods (liquid state)
#--------------------------------------------------
    def importD2O_QENS_Data(self, D2OFile):
        """ This method tries to identify the file's type automatically, and send an error message 
            in case the file could not be imported. 
            It can be used to import resolution QENS .inx or hdf5 file.
            
            The files are imported without binning. """

        try:
            if re.search('.inx', D2OFile):
                self.D2OData    = inxConvert.convert(D2OFile) 
            elif re.search('.nxs', D2OFile):
                self.D2OData    = h5process.processData(D2OFile) 
            else: #_Try a generic hdf5 file (.h5 extension or else)
                self.D2OData    = h5process.processData(D2OFile) 

        except Exception as err:
            print("Error of type %s, at line %s in %s" % ( sys.exc_info()[0], 
                                                           sys.exc_info()[2].tb_lineno, 
                                                           sys.exc_info()[2].tb_frame.f_code.co_filename))

    def fitD2O_QENS_Data(self, BH_iter=300, p0=[0.5, 0.3], bounds=[(0., 1), (0., 1)], disp=False):
        """ This method calls the D2OFit module for QENS data.
            The resulting parameters are stores as a list containing the scipy's curve_fit output for each
            scattering angle / q value. 
            
            Input:  BH_iter -> maximum number of basinhopping iteration to perform (optional)
                    p0      -> starting parameters (optional)
                    disp    -> if True, print the basinhopping's state info (optional, default False) """

        try:
            if self.resParams == []: #_Check if resolution function in available
                raise Exception("No resolution function parameters were found.\n"
                                + "Please load resolution data and fit a model to it.\n")
        except Exception as e:
            print(e)
            return

        self.D2OParams = D2OFit.basinHopping_D2OFit(self.D2OData, self, BH_iter, p0, bounds, disp)


    def plotD2OFunc(self):
        """ This method plots the resolution function.
            A PyQt window is showed with different data representation possibilities. """

        plotW = D2OPlot.D2OPlot(self)
        plotW.show()
 





#--------------------------------------------------
#_Fitting methods
#--------------------------------------------------
    def fitQENS(self, model=fitQENS_models.Protein_powder_2Lorentzians, fileIdxList="all", qWise=False, 
                                                                                    BH_iter=100, disp=True):
        """ This method calls fitQENS funtion with the given model.
            It makes use of scipy's basinhopping procedure to find the global minimum. 
            
            Input:  modelFunc   -> model function to be used for fitting
                    fileIdxList -> indices of dataset to be used in dataSetList (optional, default "all")
                    qWise       -> if true, fits the model for each q-value separatly
                                   if false, performs a global fit (optional, default False)
                    qDiscardList-> integer or list of indices corresponding to q-values to discard for analysis
                    p0          -> python list with starting parameters for basinhopping (optional)
                    bounds      -> python list of tuples (min, max), bounds for parameters during fitting
                                   (optional) 
                    paramsNames -> list of python strings for parameters names/labels
                                   (used for plotting, optional)
                    BH_iter     -> max number of basinhopping iterations (optional, default 100)
                    disp        -> if True, prints result of each basinhopping iteration 
                                   (optional, default False) """

                

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        #_Conversion to a list, if fileIdxList is a single integer
        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        for idx in fileIdxList:
            print("\nFitting data for file: %s\n" % self.dataFiles[idx] + 50*"-")
            self.paramsList[idx]  = fitQENS.basinHopping_fit(self, model, idx, qWise, BH_iter, disp)
            self.modelList[idx]   = model
            self.paramsNames[idx] = paramsNames



    def fitTempRampENS(self, fileIdxList="all", model=fitENS_models.gaussian, 
            paramsNames=['Scale Factor', 'MSD'], p0=[1, 0.1], bounds=(0., np.inf)):
        """ This method uses the given model to fit the q-dependence of elastic scattering intensity
            for each temperature in data. 

            Input:  fileIdxList -> indices of dataset to be used in dataSetList (optional, default "all")
                    modelFunc   -> model function to use, should be compatible with scipy's curve_fit
                                                                                (optional, default gaussian)
                    paramsNames -> names of the parameters used in the model (optional)
                    p0          -> starting values for model parameters (optional)
                    bounds      -> bounds on model parameters (optional)

            Returns a list of curve_fit output for each temperature. """

        #_Getting all index within the range of dataSetList size
        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        #_Conversion to a list, if fileIdxList is a single integer
        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        for idx in fileIdxList:
            self.paramsList[idx]    = fitENS.tempRampCurveFit(self, model, idx, p0, bounds)
            self.modelList[idx]     = model
            self.paramsNames[idx]   = paramsNames




#--------------------------------------------------
#_Plotting methods
#--------------------------------------------------
    def plotQENS(self, fileIdxList="all", powder=True):
        """ This methods plot the sample data in a PyQt5 widget allowing the user to show different
            types of plots. 

            The resolution function and other parameters are automatically obtained from the current
            dataSet class instance. 
            
            Input:  fileIdxList -> indices of dataset to be plotted (optional, default "all")
                    powder      -> whether the sample is a powder or in liquid state (optional, default True) 
                    qDiscardList-> integer or list if indices corresponding to q-values to discard """

        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        if powder:
            plotW = QENSPlot_powder.QENSPlot_powder(self, fileIdxList)
        
        plotW.show()


    def plotTempRampENS(self, fileIdxList="all"):
        """ This methods plot the sample data in a PyQt5 widget allowing the user to show different
            types of plots. 

            The resolution function and other parameters are automatically obtained from the current
            dataSet class instance. 
            
            Input:  fileIdxList -> indices of dataset to be plotted (optional, default "all")
                    powder      -> whether the sample is a powder or in liquid state (optional, default True) 
                    qDiscardList-> integer or list if indices corresponding to q-values to discard """

        if fileIdxList == "all":
            fileIdxList = range(len(self.dataSetList))

        if type(fileIdxList) == int:
            fileIdxList = [fileIdxList]

        plotW = TempRampPlot.TempRampPlot(self, fileIdxList)
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
