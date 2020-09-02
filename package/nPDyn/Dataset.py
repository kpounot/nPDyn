""" The module contains the central class of nPDyn, the :class:`Dataset`
    class, which handles the various dataset loaded and provides
    methods to quicly perform processing and fitting operations in the
    experimental data.

"""

import argparse

from collections import namedtuple

from nPDyn.dataTypes import (ECType, fECType, resType,
                             D2OType, fD2OType, FWSType, QENSType,
                             TempRampType)

from nPDyn.plot import (QENSPlot, FWSPlot, TempRampPlot, D2OPlot,
                        ECPlot, resPlot)


class Dataset:
    """ Master class of nPDyn, contains a list of dataFiles, which can be
        sample, resolution, D2O data or anything else as long as the format
        can be recognized.

        For call with ipython, the dataSet can be initialized directly from
        command line using the following:

        .. code-block:: bash

            $ ipython -i Dataset.py -- [dataSet related optional arguments]

        It goes like this, when a file is imported, data are loaded into
        a class, depending on the given data type. This class will inherit
        from :class:`baseType`, and might have specific methods or can
        redefine methods if needed by the data type.

        Then, using a decorator pattern, a model can be assigned to this
        class by using the provided :py:func:`assignModeltoData` method
        in :class:`Dataset` or simply by using the following:
        
            >>> myClass = Builtin_ModelClass(myClass)

        Each builtin model have a *fit* method, with a *qWise* argument
        that allow to perform either a global or a q-wis fit. They contains
        several methods to easily access fitted parameters and curves.

        Finally, various plotting methods are available, each corresponding
        to a given data type.

        Parameters
        ----------
        QENSFiles : list(str), optional
            list of Quasi-Elastic Neutron Scattering data
            files to be loaded
        FWSFiles : list(str), optional
            list of Fixed Window Scans data files to be loaded
        TempRampFiles : list(str), optional 
            list of temperature ramps data files to be loaded
        ECFile : str, optional        
            Empty cell data to be loaded
        resFiles : str, optional      
            list of resolution function related data to be loaded
        D2OFile : str, optional      
            D2O data to be loaded


    """


    def __init__(self, QENSFiles=None, FWSFiles=None, TempRampFiles=None,
                 ECFile=None, fECFile=None, tECFile=None, resFiles=None,
                 D2OFile=None, fD2OFile=None):



        # Declaring attributes related to samples dataset
        self.datasetList    = []

        # Declaring resolution function(s) related variables
        self.resData    = []

        # Declaring D2O lineshape related variables
        self.D2OData    = None
        self.fD2OData   = None

        # Declaring empty cell signal related variables
        self.ECData     = None
        self.fECData    = None
        self.tECData    = None

        # Used to store msd from trajectories, present here to allow for
        # different psf files loading
        self.msdSeriesList      = []


        self.importFiles(None, **{'QENSFiles': QENSFiles,
                                  'FWSFiles': FWSFiles,
                                  'TempRampFiles': TempRampFiles,
                                  'ECFile': ECFile,
                                  'fECFile': fECFile,
                                  'tECFile': tECFile,
                                  'resFiles': resFiles,
                                  'D2OFile': D2OFile,
                                  'fD2OFile': fD2OFile})


        self.resFuncAssign()



    def importFiles(self, fileFormat=None, QENSFiles=None, FWSFiles=None,
                    TempRampFiles=None, ECFile=None, fECFile=None,
                    tECFile=None, resFiles=None, D2OFile=None, fD2OFile=None):
        """ Read and import pre-processed data from experimental data file.

            If no file format is given, this method tries to identify the
            file's type automatically, and send an error message in case
            the file could not be imported.
            It can be used to import .inx file or QENS/FWS from hdf5 files
            for now.

            :arg dataFile:      path of the data file to be imported
            :arg fileFormat:    format of the file to be imported (inx, hdf5)
                                (optional, default None)
            :arg QENSFiles,...: named parameters containing a list of files
                                paths for each file type given


            The files are imported without binning and stored
            in *datasetList* attribute.

        """


        # Processing files arguments
        if ECFile:
            data = ECType.ECType(ECFile)
            data.importData(fileFormat=fileFormat)
            data = ECFunc_pseudoVoigt.Model(data)
            try:
                data.fit()
            except RuntimeError as e:
                print('Error while fitting empty cell data\n')
                print(e)

            self.ECData = data


        if fECFile:
            data = fECType.fECType(fECFile)
            data.importData(fileFormat=fileFormat)

            self.fECData = data


        if tECFile:
            data = TempRampType.TempRampType(tECFile)
            data.importData(fileFormat=fileFormat)

            self.tECData = data


        if resFiles:
            for f in resFiles:
                data = resType.ResType(f)
                data.importData(fileFormat=fileFormat)
                data = resFunc_pseudoVoigt.Model(data)
                data.assignECData(self.ECData)
                data.fit()

                self.resData.append(data)



        if D2OFile:
            data = D2OType.D2OType(D2OFile)
            data.importData(fileFormat=fileFormat)
            data = D2OFunc_singleLorentzian_CF.Model(data)
            data.assignECData(self.ECData)

            if self.resData != []:
                data.assignResData(self.resData[0])
                data.normalize()
                data.fit()

            self.D2OData = data




        if fD2OFile:
            data = fD2OType.fD2OType(fD2OFile)
            data.importData(fileFormat=fileFormat)

            if self.fECData is not None:
                data.assignECData(self.fECData)
            else:
                data.assignECData(self.ECData)

            if self.resData != []:
                data.assignResData(self.resData[0])


            self.fD2OData = data




        if QENSFiles:
            for f in QENSFiles:
                data = QENSType.QENSType(f)
                data.importData(fileFormat=fileFormat)
                data.assignD2OData(self.D2OData)
                data.assignECData(self.ECData)
                self.datasetList.append(data)


        if FWSFiles:
            for f in FWSFiles:
                data = FWSType.FWSType(f)
                data.importData()

                if self.fECData is not None:
                    data.assignECData(self.fECData)
                else:
                    data.assignECData(self.ECData)


                if self.fD2OData is not None:
                    data.assignD2OData(self.fD2OData)
                else:
                    data.assignD2OData(self.D2OData)


                self.datasetList.append(data)



        if TempRampFiles:
            for f in TempRampFiles:
                data = TempRampType.TempRampType(f)
                data.importData(fileFormat=fileFormat)
                data.assignD2OData(self.D2OData)

                if self.tECData is not None:
                    data.assignECData(self.tECData)
                else:
                    data.assignECData(self.ECData)

                self.datasetList.append(data)




    def importRawData(self, dataList, instrument, dataType='QENS', kwargs={}):
        """ This method uses instrument-specific algorithm to import raw data.

            :arg dataList:      a list of data files to be imported
            :arg instrument:    the instrument used to record data
                                (only 'IN16B' possible for now)
            :arg dataType:      type of data recorded (can be 'QENS', 'FWS',
                                'res', 'ec', 'D2O', 'fec' or 'fD2O')
            :arg kwargs:        keyword arguments to be passed to the algorithm
                                (see algorithm in dataParsers for details)

        """

        if dataType == 'QENS':
            data = QENSType.QENSType(dataList)

            data.importRawData(dataList, instrument, dataType, kwargs)

            data.assignD2OData(self.D2OData)
            data.assignECData(self.ECData)

            self.datasetList.append(data)



        elif dataType == 'FWS':
            data = FWSType.FWSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)

            if self.fECData is not None:
                data.assignECData(self.fECData)
            else:
                data.assignECData(self.ECData)


            if self.fD2OData is not None:
                data.assignD2OData(self.fD2OData)
            else:
                data.assignD2OData(self.D2OData)


            self.datasetList.append(data)



        elif dataType == 'res':
            data = resType.ResType(dataList)

            data.importRawData(dataList, instrument, dataType, kwargs)

            tmpData = data.data
            tmpRaw  = data.rawData
            for idx, val in enumerate(data.data):
                tmp = resType.ResType(dataList[idx])

                tmp.data = tmpData[idx]
                tmp.rawData = tmpRaw[idx]

                tmp = resFunc_pseudoVoigt.Model(tmp)
                tmp.assignECData(self.ECData)
                tmp.fit()

                self.resData.append(tmp)


        elif dataType == 'ec':
            data = ECType.ECType(dataList)

            data.importRawData(dataList, instrument, dataType, kwargs)
            data.data    = data.data[0]
            data.rawData = data.rawData[0]

            data = ECFunc_pseudoVoigt.Model(data)
            try:
                data.fit()
            except RuntimeError as e:
                print('Error while fitting empty cell data\n')
                print(e)

            self.ECData = data

        elif dataType == 'fec':
            data = fECType.fECType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)

            self.fECData = data


        elif dataType == 'D2O':
            data = D2OType.D2OType(dataList)

            data.importRawData(dataList, instrument, dataType, kwargs)
            data.rawData = data.rawData[0]
            data.data    = data.data[0]

            data = D2OFunc_singleLorentzian_CF.Model(data)
            data.assignECData(self.ECData)

            if self.resData != []:
                data.assignResData(self.resData[0])
                data.fit()

            self.D2OData = data

        elif dataType == 'fD2O':
            data = fD2OType.fD2OType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)

            if self.fECData is not None:
                data.assignECData(self.fECData)
            else:
                data.assignECData(self.ECData)

            if self.resData != []:
                data.assignResData(self.resData[0])


            self.fD2OData = data


        elif dataType == 'TempRamp':
            data = TempRampType.TempRampType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)

            if self.fECData is not None:
                data.assignECData(self.fECData)
            else:
                data.assignECData(self.ECData)


            if self.fD2OData is not None:
                data.assignD2OData(self.fD2OData)
            else:
                data.assignD2OData(self.D2OData)


            self.datasetList.append(data)





        self.resFuncAssign()



    def resFuncAssign(self):
        """ This method is used during initialization, and can be used
            each time resolution functions need to be re-assigned to data
            (after a model change for instance).
            If only one resolution function was provided, it will assume
            that the same one has to be used for all QENS and FWS data loaded.
            Therefore, the same resolution function data will be assigned
            to all experimental data.

            If the number of resolution data is the same as the number of
            experimental data (QENS or FWS), then they are assigned in
            an order-wise manner.

            If none of the above conditions are fulfilled, nothing is done
            and resolution data should be assigned manually.

        """

        lenResData = len(self.resData)

        if lenResData == 1:
            for data in self.datasetList:
                data.assignResData(self.resData[0])


        if lenResData == len(self.datasetList):
            for idx, data in enumerate(self.datasetList):
                data.assignResData(self.resData[idx])





# -------------------------------------------------
# Importation, reset and deletion methods
# -------------------------------------------------
    def removeDataset(self, fileIdx):
        """ This method takes either a single integer as argument,
            which corresponds to the index of the file to be removed
            from self.dataFiles, self.datasetList and self.rawDataList.

        """

        self.datasetList.pop(fileIdx)


    def resetDataset(self, *fileIdxList):
        """ This method takes either a single integer or a list of
            integer as argument. They correspond to the indices of the
            file to be reset to their initial state using self.rawDataList.

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        for idx in fileIdxList:
            self.datasetList[idx].resetData()



    def resetAll(self):
        """ Reset all dataset, as well as resolution D2O and empty
            cell data to their initial state.

        """

        for dataset in self.datasetList:
            dataset.resetData()

        for resData in self.resData:
            resData.resetData()

        if self.D2OData:
            self.D2OData.resetData()

        if self.fD2OData:
            self.fD2OData.resetData()

        if self.ECData:
            self.ECData.resetData()

        if self.fECData:
            self.fECData.resetData()

        if self.tECData:
            self.tECData.resetData()


# -------------------------------------------------
# Data manipulation methods
# -------------------------------------------------
    def normalize_usingResFunc(self, *fileIdxList):
        """ This method uses the fitted normalization factor
            (normF) to normalize each dataSet in fileIdxList.

            If only one resolution data file is loaded, use this
            one for all dataset, else, use them in the same order
            as dataset in self.datasetList

            :arg fileIdxList: list of indices of dataset in
            self.datasetList to be normalized

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        for i in fileIdxList:
            self.datasetList[i].normalize()




    def normalize_ENS_usingLowTemp(self, *fileIdxList, nbrBins=8):
        """ This method is meant to be used only with elastic temperature
            ramp. For which the given number of first bins (low temperature)
            are used to compute an average signal at low temperature.
            The average is then used to normalize the whole dataset.
            This for each q-value.

            :arg fileIdxList:   can be "all", then every dataSet in
                                self.datasetList is normalized can also
                                be a single integer or a list of integer
                                (optional, default "all")
            :arg nbrBins:       number of low temperature bins used to
                                compute the normalization factor
                                (optional, default 8)

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        for i in fileIdxList:
            self.datasetList[i].normalize_usingLowTemp(nbrBins)






    def absorptionCorrection(self, *fileIdxList, canType='tube',
                             canScaling=0.95, neutron_wavelength=6.27,
                             absco_kwargs={}, D2O=True, res=True):
        """ Method for quick absorption correction on all selected
            dataset in fileIdxList.

            Same arguments as in :class:`baseType` class, except D2O, which,
            if True, involves that corrections are performed on D2O data too.

            Also, if *res* argument is set to True, correction are done for
            resolution function data too.

        """


        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        # Apply corrections for samples data
        for i in fileIdxList:
            self.datasetList[i].absorptionCorrection(canType, canScaling,
                                                     neutron_wavelength,
                                                     absco_kwargs)


        if res:
            try:
                absco_kwargs['thickness_S'] = 0.05
                absco_kwargs['mu_i_S'] = 1.673
                absco_kwargs['mu_f_S'] = 1.673
                for resData in self.resData:
                    resData.absorptionCorrection(canType, canScaling,
                                                 neutron_wavelength,
                                                 absco_kwargs)
                    resData.fit()
            except AttributeError:
                print("No resolution data were loaded, corrections on "
                      "these cannot be applied\n")
                return


        if D2O:
            if self.D2OData is not None:
                try:
                    self.D2OData.absorptionCorrection(canType, canScaling,
                                                      neutron_wavelength,
                                                      absco_kwargs)
                    self.D2OData.fit()
                except AttributeError as e:
                    print("No D2O data were loaded, corrections on these "
                          "cannot be applied\n")
                    print(e)
                    return

            if self.fD2OData is not None:
                try:
                    self.fD2OData.absorptionCorrection(canType, canScaling,
                                                       neutron_wavelength,
                                                       absco_kwargs)
                except AttributeError as e:
                    print("No fD2O data were loaded, corrections on these "
                          "cannot be applied\n")
                    print(e)
                    return






    def subtract_EC(self, *fileIdxList, subFactor=0.95, subD2O=True,
                    subRes=False, useModel=True):
        """ This method uses the fitted empty cell function to subtract
            the signal for the selected dataset.

            :arg subFactor:   pre-multiplying factor for empty cell
                              data prior to substraction
            :arg fileIdxList: can be "all", then every dataSet in
                              self.datasetList is normalized
                              can also be a single integer or a list
                              of integer (optional, default "all")
            :arg subD2O:      if True, tries to substract empty cell
                              signal from D2O data too
            :arg subRes:      if True, substract empty cell signal from
                              resolutions data too

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        # Apply corrections for samples data
        for i in fileIdxList:
            self.datasetList[i].subtractEC(subFactor, useModel)



        if subRes:
            for idx, resData in enumerate(self.resData):
                resData.subtractEC(subFactor, useModel)
                resData.fit()


        if subD2O:
            try:
                self.datasetList[i].D2OData.subtractEC(subFactor, useModel)
                self.datasetList[i].D2OData.fit()
            except AttributeError:
                pass

            try:
                self.datasetList[i].fD2OData.subtractEC(subFactor, useModel)
                self.datasetList[i].fD2OData.fit()
            except AttributeError:
                pass




    def discardOutliers(self, meanScale, *fileIdxList, D2O=False,
                        EC=False, res=False):
        """ Discard outliers by setting errors to infinite for each
            data points having a signal over noise ration under a given
            threshold (determined by meanScale * mean(S/N ratios)).

            :arg meanScale:   factor by which mean of signal over noise
                              ratio will be multiplied. Then, this scaled
                              mean is used as a threshold under which
                              data errors will be set to infinite so that
                              they won't weigh in the fitting procedure.
            :arg fileIdxList: list of data files to be used
                              for outliers discard
            :arg D2O:         if True, discard outliers in D2O data too,
                              and refit
            :arg EC:          if True, discard outliers in empty cell data,
                              and refit
            :arg res:         if True, discard outliers in resolution
                              and refit

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        for idx in fileIdxList:
            self.datasetList[idx].discardOutliers(meanScale)



        if res:
            for resData in self.resData:
                try:
                    resData.discardOutliers(meanScale)
                    resData.fit()
                except AttributeError:
                    print("No resolution data were loaded, outliers "
                          "couldn't be discarded")
                    pass


        if EC:
            try:
                self.ECData.discardOutliers(meanScale)
                self.ECData.fit()
            except AttributeError:
                print("No empty cell data were loaded, outliers couldn't "
                      "be discarded")
                pass



        if D2O:
            try:
                self.D2OData.discardOutliers(meanScale)
                self.D2OData.qWiseFit()
            except AttributeError:
                print("No D2O data were loaded, outliers couldn't be "
                      "discarded")
                pass







    def discardDetectors(self, decIdxList, *fileIdxList):
        """ Remove data corresponding to given detectors/q-values
            The process modifies dataset.qIdx attributes, that is
            used for sample QENS fitting and plotting.

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        for idx in fileIdxList:
            self.datasetList[idx].discardDetectors(*decIdxList)




    def resetDetectors(self, *fileIdxList):
        """ Reset qIdx entry to its original state, with all q values
            taken into account. """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        for idx in fileIdxList:
            self.datasetList[idx].resetDetectors()


    def setQRange(self, minQ, maxQ, *fileIdxList):
        """ Defines a q-range within which detectors are not discarded.

            :arg minQ: minimum q-value to keep
            :arg maxQ: maximum q-value to keep

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        for idx in fileIdxList:
            self.datasetList[idx].setQRange(minQ, maxQ)




    def assignModeltoData(self, model, *fileIdxList):
        """ Helper function to quickly assign the given model to all
            dataset with given indices in self.datasetList. If model is
            not None, the decorator pattern is used to modify the dataType
            class behavior.

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))


        for idx in fileIdxList:
            self.datasetList[idx] = model(self.datasetList[idx])





    def fitData(self, *fileIdxList, p0=None, bounds=None,
                qWise=False, kwargs={}):
        """ Helper function to quickly call fit method in all class instances
            present in self.datasetList for the given indices in fileIdxList.
            Check first for the presence of a fit method and print a warning
            message if none is found.

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))


        for idx in fileIdxList:
            if qWise:
                self.datasetList[idx].qWiseFit(p0, bounds, kwargs)
            else:
                self.datasetList[idx].fit(p0, bounds, kwargs)





# -------------------------------------------------
# Binning methods
# -------------------------------------------------
    def binDataset(self, binS, *fileIdxList):
        """ The method find the index corresponding to the file, perform
            the binning process, then replace the value in self.datasetList
            by the binned one.

            :arg binS:        bin size
            :arg fileIdxList: indices of the dataSet to be binned, can be
                              a single int or a list of int
                              (optional, default "all")

        """


        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        # Calling binData for each dataSet in fileIdxList
        for idx in fileIdxList:
            self.datasetList[idx].binData(binS)




    def binResData(self, binS, *fileIdxList):
        """ Same as binDataset but for resolution function data.

            :arg binS:        bin size
            :arg fileIdxList: indices of the dataSet to be binned, can be
                              a single int or a list of int
                              (optional, default "all")

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.resData))

        # Calling binData for each dataSet in fileIdxList
        for idx in fileIdxList:
            self.resData[idx].binData(binS)




    def binAll(self, binS):
        """ Bin all dataSet in datasetList as well as resolutio,
            empty cell and D2O data if present.

        """

        self.binDataset(binS)  # Bin the dataset list

        # For other types of data, check if something was loaded, and if so,
        # perform the binning
        if self.resData:
            self.binResData(binS)

        if self.ECData:
            self.ECData.binData(binS)

        if self.tECData:
            self.tECData.binData(binS)

        if self.D2OData:
            self.D2OData.binData(binS)





# -------------------------------------------------
# Resolution function related methods
# -------------------------------------------------
    def plotResFunc(self):
        """ This method plots the resolution function.
            A PyQt window is showed with different data
            representation possibilities.

        """

        plotW = resPlot.ResPlot(self.resData)
        plotW.show()


# -------------------------------------------------
# Empty cell data related methods
# -------------------------------------------------
    def plotECFunc(self):
        """ This method plots the empty cell lineshape fitted function.
            A PyQt window is showed with different data representation
            possibilities.

        """

        plotW = ECPlot.ECPlot([self.ECData])
        plotW.show()


# -------------------------------------------------
# D2O signal related methods (liquid state)
# -------------------------------------------------
    def plotD2OFunc(self):
        """ This method plots the resolution function.
            A PyQt window is showed with different data representation
            possibilities.

        """

        plotW = D2OPlot.D2OPlot([self.D2OData])
        plotW.show()


# -------------------------------------------------
# Plotting methods
# -------------------------------------------------
    def plotQENS(self, *fileIdxList):
        """ This methods plot the sample data in a PyQt5 widget allowing
            the user to show different types of plots.

            The resolution function and other parameters are automatically
            obtained from the current dataSet class instance.

            :arg fileIdxList: indices of dataset to be plotted
                              (optional, default "all")

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        datasetList = [self.datasetList[i] for i in fileIdxList]

        plotW = QENSPlot.QENSPlot(datasetList)

        plotW.show()

    def plotFWS(self, fileIdx=0):
        """ This methods plot the sample data in a PyQt5 widget
            allowing the user to show different types of plots.

            The resolution function and other parameters are automatically
            obtained from the current dataSet class instance.

            :arg fileIdx: index of dataset to plot in self.datasetList

        """

        plotW = FWSPlot.FWSPlot(self.datasetList[fileIdx])

        plotW.show()

    def plotTempRampENS(self, *fileIdxList):
        """ This methods plot the sample data in a PyQt5 widget allowing
            the user to show different types of plots.

            The resolution function and other parameters are automatically
            obtained from the current dataSet class instance.

            :arg fileIdxList:  indices of dataset to be plotted
                               (optional, default "all")
            :arg powder:       whether the sample is a powder or in liquid
                               state (optional, default True)
            :arg qDiscardList: integer or list if indices corresponding to
                               q-values to discard

        """

        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.datasetList))

        datasetList = [self.datasetList[i] for i in fileIdxList]

        plotW = TempRampPlot.TempRampPlot(datasetList)
        plotW.show()


if __name__ == '__main__':

    # Defining options for nPDyn call
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--QENS", nargs='*',
                        help="List of files corresponding to \
                              Quasi-Elastic Neutron Scattering (QENS) data")
    parser.add_argument("-f", "--FWS", nargs='*',
                        help="List of files corresponding to \
                              Fixed-Window Scan (FWS) data")
    parser.add_argument("-tr", "--TempRamp", nargs='*',
                        help="List of files corresponding to \
                              temperature ramp elastic data")
    parser.add_argument("-res", "--resolution", nargs='*',
                        help="Specify the file(s) to be used for \
                              resolution function fitting.")
    parser.add_argument("-ec", "--empty-cell", nargs='?',
                        help="Specify the file containing \
                              QENS empty cell data")
    parser.add_argument("-fec", "--fixed-empty-cell", nargs='?',
                        help="Specify the file containing \
                              FWS empty cell data")
    parser.add_argument("-tec", "--TempRamp-empty-cell", nargs='?',
                        help="Specify the file containing \
                              temperature ramp empty cell data")
    parser.add_argument("-d", "--D2O", nargs='?',
                        help="Specify the file containing QENS D2O data")
    parser.add_argument("-fd", "--fixed-D2O", nargs='?',
                        help="Specify the file containing FWS D2O data")

    args = parser.parse_args()

    data = Dataset(args.QENS, args.FWS,
                   args.TempRamp,
                   args.empty_cell,
                   args.fixed_empty_cell,
                   args.TempRamp_empty_cell,
                   args.resolution, args.D2O,
                   args.fixed_D2O)
