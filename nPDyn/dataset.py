"""The module contains the central class of nPDyn.

The :class:`Dataset`
class, which handles the various dataset loaded and provides
methods to quickly perform processing and fitting operations in the
experimental data.

"""
import argparse

from collections import namedtuple

from nPDyn.dataTypes import fwsType, qensType

from nPDyn.plot import makeWindow, QENSPlot, FWSPlot


class Dataset:
    """Master class of nPDyn.

    It contains a list of data files, which can be
    sample, resolution, D2O data or anything else as long as the format
    can be recognized.

    For call with ipython, the :class:`Dataset` class can be initialized
    directly from command line using the following:

    .. code-block:: bash

        $ ipython -i Dataset.py -- [dataset related optional arguments]

    It goes like this, when a file is imported, data are loaded into
    a data type class, depending on the given data type. This class will
    inherit from :class:`baseType`, and might have specific methods or can
    redefine methods if needed by the data type.

    All QENS and FWS data are listed in the `dataList` attribute.
    The other data type are assigned to the other attributes as described
    below in the parameters.

    The modelling rely on the *lmfit* library. Some built-in models
    are available to the user in the :module:`models.builtins` module.
    Also, simple lineshapes as well as a 2D model builder for *lmfit*
    are available in the :module:`models.presets` module.

    Finally, various plotting methods are available, each corresponding
    to a given data type.

    Parameters
    ----------
    QENSFiles : list(str), optional
        list of Quasi-Elastic Neutron Scattering data
        files to be loaded
    FWSFiles : list(str), optional
        list of Fixed Window Scans data files to be loaded
    ECFile : str, optional
        Empty cell data to be loaded in `ECData` attribute.
    ECFile : str, optional
        Empty cell data to be loaded
    resFiles : str, optional
        list of resolution function related data to be loaded
    D2OFile : str, optional
        D2O data to be loaded

    """

    def __init__(
        self,
        QENSFiles=None,
        FWSFiles=None,
        ECFile=None,
        fECFile=None,
        resFiles=None,
        D2OFile=None,
        fD2OFile=None,
    ):
        # Declaring attributes related to samples dataset
        self.dataList = []

        # Declaring resolution function(s) related variables
        self.resData = []

        # Declaring D2O lineshape related variables
        self.D2OData = None
        self.fD2OData = None

        # Declaring empty cell signal related variables
        self.ECData = None
        self.fECData = None

        # Used to store msd from trajectories, present here to allow for
        # different psf files loading
        self.msdSeriesList = []

        self.importFiles(
            None,
            **{
                "QENSFiles": QENSFiles,
                "FWSFiles": FWSFiles,
                "ECFile": ECFile,
                "fECFile": fECFile,
                "resFiles": resFiles,
                "D2OFile": D2OFile,
                "fD2OFile": fD2OFile,
            }
        )
        self.resFuncAssign()

    def importFiles(
        self,
        fileFormat=None,
        QENSFiles=None,
        FWSFiles=None,
        ECFile=None,
        fECFile=None,
        resFiles=None,
        D2OFile=None,
        fD2OFile=None,
    ):
        """Read and import pre-processed data from experimental data file.

        If no file format is given, this method tries to identify the
        file's type automatically, and send an error message in case
        the file could not be imported.
        It can be used to import .inx file or QENS/FWS from hdf5 files
        for now.

        Parameters
        ----------
        fileFormat : str, optional
            format of the file to be imported (inx, hdf5)
            (default None)
        QENSFiles, FWSFiles,... : str or list of str, optional
            named parameters containing a list of files paths
            (for QENS, FWS, res) or a single file paths
            for each file to be imported.

        Notes
        -----
        The files are imported without binning and stored
        in *dataList* attribute.

        """
        # Processing files arguments
        if ECFile:
            data = qensType.QENSType(ECFile)
            data.importData(fileFormat=fileFormat)
            self.ECData = data

        if fECFile:
            data = fwsType.FWSType(fECFile)
            data.importData(fileFormat=fileFormat)
            self.fECData = data

        if resFiles:
            for f in resFiles:
                data = qensType.QENSType(f)
                data.importData(fileFormat=fileFormat)
                data.ECData = self.ECData

                self.resData.append(data)

        if D2OFile:
            data = qensType.QENSType(D2OFile)
            data.importData(fileFormat=fileFormat)
            data.ECData = self.ECData

            if self.resData != []:
                data.resData = self.resData[0]

            self.D2OData = data

        if fD2OFile:
            data = fwsType.FWSType(fD2OFile)
            data.importData(fileFormat=fileFormat)

            if self.fECData is not None:
                data.ECData = self.fECData
            else:
                data.ECData = self.ECData

            if self.resData != []:
                data.resData = self.resData[0]

            self.fD2OData = data

        if QENSFiles:
            for f in QENSFiles:
                data = qensType.QENSType(f)
                data.importData(fileFormat=fileFormat)
                data.D2OData = self.D2OData
                data.ECData = self.ECData
                self.dataList.append(data)

        if FWSFiles:
            for f in FWSFiles:
                data = fwsType.FWSType(f)
                data.importData(fileFormat=fileFormat)

                if self.fECData is not None:
                    data.ECData = self.fECData
                else:
                    data.ECData = self.ECData

                if self.fD2OData is not None:
                    data.D2OData = self.fD2OData
                else:
                    data.D2OData = self.D2OData

                self.dataList.append(data)

        self.resFuncAssign()

    def importRawData(self, dataList, instrument, dataType="QENS", kwargs={}):
        """This method uses instrument-specific algorithm to import raw data.

        :arg dataList:      a list of data files to be imported
        :arg instrument:    the instrument used to record data
                            (only 'IN16B' possible for now)
        :arg dataType:      type of data recorded (can be 'QENS', 'FWS',
                            'res', 'ec', 'D2O', 'fec' or 'fD2O')
        :arg kwargs:        keyword arguments to be passed to the algorithm
                            (see algorithm in dataParsers for details)

        """

        if dataType == "QENS":
            data = qensType.QENSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)
            data.D2OData = self.D2OData
            data.ECData = self.ECData
            self.dataList.append(data)

        elif dataType == "FWS":
            data = fwsType.FWSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)

            if self.fECData is not None:
                data.ECData = self.fECData
            else:
                data.ECData = self.ECData

            if self.fD2OData is not None:
                data.D2OData = self.fD2OData
            else:
                data.D2OData = self.D2OData

            self.dataList.append(data)

        elif dataType == "res":
            data = qensType.QENSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)
            data.ECData = self.ECData
            self.resData.append(data)

        elif dataType == "ec":
            data = qensType.QENSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)
            self.ECData = data

        elif dataType == "fec":
            data = fwsType.FWSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)
            self.fECData = data

        elif dataType == "D2O":
            data = qensType.QENSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)
            data.ECData = self.ECData

            if self.resData != []:
                data.resData = self.resData[0]

            self.D2OData = data

        elif dataType == "fD2O":
            data = fwsType.FWSType(dataList)
            data.importRawData(dataList, instrument, dataType, kwargs)

            if self.fECData is not None:
                data.ECData = self.fECData
            else:
                data.ECData = self.ECData

            if self.resData != []:
                data.resData = self.resData[0]

            self.fD2OData = data

        self.resFuncAssign()

    def resFuncAssign(self):
        """Assign data for resolution function to each dataset.

        This method is used during initialization, and can be used
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
            for data in self.dataList:
                data.resData = self.resData[0]

        if lenResData == len(self.dataList):
            for idx, data in enumerate(self.dataList):
                data.resData = self.resData[idx]

    # -------------------------------------------------
    # Importation, reset and deletion methods
    # -------------------------------------------------
    def removeDataset(self, fileIdx):
        """Remove a dataset from 'dataList' attribute.

        This method takes either a single integer as argument,
        which corresponds to the index of the file to be removed
        from self.dataFiles, self.dataList and self.rawDataList.

        """
        self.dataList.pop(fileIdx)

    def resetData(self, *fileIdxList):
        """Reset data from 'dataList' attribute.

        This method takes either a single integer or a list of
        integer as argument. They correspond to the indices of the
        file to be reset to their initial state using self.rawDataList.

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        for idx in fileIdxList:
            self.dataList[idx].resetData()

    def resetAll(self):
        """Reset all datasets, as well as resolution, D2O and empty
        cell data to their initial state.

        """
        for dataset in self.dataList:
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

    # -------------------------------------------------
    # Data manipulation methods
    # -------------------------------------------------
    def scaleData(self, scale, *fileIdxList):
        """Scale the selected data in `self.dataList`.

        Parameters
        ----------
        scale : float or array of floats
            scale factor(s) to apply on the intensities and errors.
        fileIdxList : int, optional
            list of indices of dataset in self.dataList to be normalized

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        for i in fileIdxList:
            self.dataList[i].scaleData(scale)

    def normalize_usingResFunc(self, *fileIdxList):
        """This method uses the fitted model for resolution function
        to normalize the data.

        Each dataset is normalized using the `resData` attribute associated
        to it.

        Parameters
        ----------
        fileIdxList : int, optional
            list of indices of dataset in self.dataList to be normalized

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        for i in fileIdxList:
            self.dataList[i].normalize_usingResFunc()

    def normalize_usingSelf(self, *fileIdxList):
        """This method uses the fitted model for resolution function
        to normalize the data.

        Each dataset is normalized using the `resData` attribute associated
        to it.

        Parameters
        ----------
        fileIdxList : int, optional
            list of indices of dataset in self.dataList to be normalized

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        for i in fileIdxList:
            self.dataList[i].normalize_usingSelf()

    def normalize_usingLowTemp(self, *fileIdxList, nbrBins=8):
        """Normalize data using low temperature measurements.

        For temperature based data, the given number of first bins
        are used to compute an average signal at low temperature.
        The average is then used to normalize the whole dataset.
        This for each q-value.

        Parameters
        ----------
        fileIdxList : int, optional
            List of indices of dataset in self.dataList to be normalized.
        nbrBins : int
            Number of low temperature bins used to compute the
            normalization factor. (optional, default 8)

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        for i in fileIdxList:
            self.dataList[i].normalize_usingLowTemp(nbrBins)

    def absorptionCorrection(
        self,
        *fileIdxList,
        canType="tube",
        canScaling=0.95,
        neutron_wavelength=6.27,
        absco_kwargs={},
        D2O=True,
        res=True
    ):
        """Method for quick absorption correction on all selected
        dataset in fileIdxList.

        Same arguments as in :class:`baseType` class, except D2O, which,
        if True, involves that corrections are performed on D2O data too.

        Also, if *res* argument is set to True, correction are done for
        resolution function data too.

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        # Apply corrections for samples data
        for i in fileIdxList:
            self.dataList[i].absorptionCorrection(
                canType, canScaling, neutron_wavelength, absco_kwargs
            )

        if res:
            try:
                absco_kwargs["thickness_S"] = 0.05
                absco_kwargs["mu_i_S"] = 1.673
                absco_kwargs["mu_f_S"] = 1.673
                for resData in self.resData:
                    resData.absorptionCorrection(
                        canType, canScaling, neutron_wavelength, absco_kwargs
                    )
            except AttributeError:
                print(
                    "No resolution data were loaded, corrections on "
                    "these cannot be applied\n"
                )
                return

        if D2O:
            if self.D2OData is not None:
                try:
                    self.D2OData.absorptionCorrection(
                        canType, canScaling, neutron_wavelength, absco_kwargs
                    )
                except AttributeError as e:
                    print(
                        "No D2O data were loaded, corrections on these "
                        "cannot be applied\n"
                    )
                    print(e)
                    return

            if self.fD2OData is not None:
                try:
                    self.fD2OData.absorptionCorrection(
                        canType, canScaling, neutron_wavelength, absco_kwargs
                    )
                except AttributeError as e:
                    print(
                        "No fD2O data were loaded, corrections on these "
                        "cannot be applied\n"
                    )
                    print(e)
                    return

    def subtract_EC(
        self,
        *fileIdxList,
        subFactor=0.95,
        subD2O=True,
        subRes=False,
        useModel=True
    ):
        """This method uses the fitted empty cell function to subtract
        the signal for the selected dataset.

        :arg subFactor:   pre-multiplying factor for empty cell
                          data prior to subtraction
        :arg fileIdxList: can be "all", then every dataset in
                          self.dataList is normalized
                          can also be a single integer or a list
                          of integer (optional, default "all")
        :arg subD2O:      if True, tries to subtract empty cell
                          signal from D2O data too
        :arg subRes:      if True, subtract empty cell signal from
                          resolutions data too
        :arg useModel:    if True, will use the fitted model of empty cell
                          for subtraction.

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        # Apply corrections for samples data
        for i in fileIdxList:
            self.dataList[i].subtractEC(subFactor, useModel)

        if subRes:
            for idx, resData in enumerate(self.resData):
                resData.subtractEC(subFactor, useModel)
                resData.fit()

        if subD2O:
            try:
                self.dataList[i].D2OData.subtractEC(subFactor, useModel)
            except AttributeError:
                pass

            try:
                self.dataList[i].fD2OData.subtractEC(subFactor, useModel)
            except AttributeError:
                pass

    def discardDetectors(self, *qIdx):
        """Remove data corresponding to given detectors/q-values indices."""
        for data in self.dataList:
            data.discardDetectors(*qIdx)

        for data in self.resData:
            data.discardDetectors(*qIdx)

        for data in (self.ECData, self.fECData, self.D2OData, self.fD2OData):
            if data is not None:
                data.discardDetectors(*qIdx)

    def setQRange(self, minQ, maxQ):
        """Defines a q-range within which detectors are not discarded.

        Parameters
        ----------
        minQ : float
            Minimum q-value to keep.
        maxQ : float
            Maximum q-value to keep.

        """
        for data in self.dataList:
            data.setQRange(minQ, maxQ)

        for data in self.resData:
            data.setQRange(minQ, maxQ)

        for data in (self.ECData, self.fECData, self.D2OData, self.fD2OData):
            if data is not None:
                data.setQRange(minQ, maxQ)

    def fitData(self, *fileIdxList, **kwargs):
        """Helper function to quickly call fit method in all class instances
        present in self.dataList for the given indices in fileIdxList.
        Check first for the presence of a fit method and print a warning
        message if none is found.

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        for idx in fileIdxList:
            self.dataList[idx].fit(**kwargs)

    def fitRes(self, *fileIdxList, **kwargs):
        """Helper function to quickly call fit method in all class instances
        present in self.resData for the given indices in fileIdxList.
        Check first for the presence of a fit method and print a warning
        message if none is found.

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.resData))

        for idx in fileIdxList:
            self.resData[idx].fit(**kwargs)

    # -------------------------------------------------
    # Binning methods
    # -------------------------------------------------
    def binDataset(self, binS, *fileIdxList, axis="energies"):
        """The method find the index corresponding to the file, perform
        the binning process, then replace the value in self.dataList
        by the binned one.

        :arg binS:        bin size
        :arg fileIdxList: indices of the dataSet to be binned, can be
                          a single int or a list of int
                          (optional, default "all")

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        # Calling binData for each dataSet in fileIdxList
        for idx in fileIdxList:
            self.dataList[idx].binData(binS, axis=axis)

    def binResData(self, binS, *fileIdxList, axis="energies"):
        """Same as binDataset but for resolution function data.

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
            self.resData[idx].binData(binS, axis=axis)

    def binAll(self, binS, axis="energies"):
        """Bin all datasets in dataList as well as resolution,
        empty cell and D2O data if present.

        """
        self.binDataset(binS, axis=axis)  # Bin the dataset list

        # For other types of data, check if something was loaded, and if so,
        # perform the binning
        if self.resData:
            self.binResData(binS, axis=axis)

        if self.ECData:
            self.ECData.binData(binS, axis=axis)

        if self.D2OData:
            self.D2OData.binData(binS, axis=axis)

    # -------------------------------------------------
    # Plotting methods
    # -------------------------------------------------
    def plotResFunc(self):
        """This method plots the resolution function.

        A PyQt window is showed with different data
        representation possibilities.

        """
        makeWindow(QENSPlot, self.resData)

    def plotECFunc(self):
        """This method plots the empty cell lineshape fitted function.

        A PyQt window is showed with different data representation
        possibilities.

        """
        makeWindow(QENSPlot, [self.ECData])

    def plotD2OFunc(self):
        """This method plots the resolution function.

        A PyQt window is showed with different data representation
        possibilities.

        """
        makeWindow(QENSPlot, [self.D2OData])

    def plotQENS(self, *fileIdxList):
        """This methods plot the sample data in a PyQt5 widget allowing
        the user to show different types of plots.

        The resolution function and other parameters are automatically
        obtained from the current dataset class instance.

        Parameters
        ----------
        fileIdxList : int, optional
            List of indices of dataset in self.dataList to be normalized.

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        dataList = [self.dataList[i] for i in fileIdxList]

        makeWindow(QENSPlot, dataList)

    def plotFWS(self, *fileIdxList):
        """This methods plot the sample data in a PyQt5 widget
        allowing the user to show different types of plots.

        The resolution function and other parameters are automatically
        obtained from the current dataset class instance.

        :arg fileIdx: index of dataset to plot in self.dataList

        """
        # If not file indices were given, assumes that all should be use
        if not fileIdxList:
            fileIdxList = range(len(self.dataList))

        dataList = [self.dataList[i] for i in fileIdxList]

        makeWindow(FWSPlot, dataList)
