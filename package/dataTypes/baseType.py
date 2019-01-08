import numpy as np

from collections import namedtuple

from itertools import count

from ..dataManipulation.binData import binData
from ..fileFormatParser import guessFileFormat, readFile, fileImporters


class BaseType:

    def __init__(self, fileName=None, data=None, rawData=None):
        """ Initialize a base type that will be inherited by all other specialized types through its decorator. 
        
            Input:  fileName    -> name of the file being read
                    data        -> resulting namedtuple from data parsers 
                    rawData     -> used by the decorator """

        self.fileName   = fileName
        self.data       = data 
        self.rawData    = rawData #_Used to reset data to its initial state




    def importData(self, fileFormat=None):
        """ Extract data from file and store them in self.data and self.rawData attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. """

        if fileFormat:
            data = readFile(fileFormat, self.fileName)
        else:
            data = guessFileFormat(self.fileName)

        self.data       = data
        self.rawData    = data



    def binData(self, binSize):
        """ Bin self.data attribute using the given binSize. """

        self.data = binData(self.data, binSize)



    def normalize(self, normFList):
        """ Normalizes data using a list of normalization factors.
            There should be as many normalization factors as q-values in data. """

        #_Applying normalization
        self.data = self.data._replace(intensities = 
                        np.apply_along_axis(lambda arr: arr / normFList, 0, self.data.intensities))
        self.data = self.data._replace(errors = 
                        np.apply_along_axis(lambda arr: arr / normFList, 0, self.data.errors))
        self.data = self.data._replace(norm = True)



    def resetData(self):
        """ Reset self.data to its initial state by copying rawData attribute. """

        self.data = self.data._replace( qVals       = np.copy(self.rawData.qVals),
                                        X           = np.copy(self.rawData.X),
                                        intensities = np.copy(self.rawData.intensities),
                                        errors      = np.copy(self.rawData.errors),
                                        temp        = np.copy(self.rawData.temp),
                                        norm        = False,
                                        qIdx        = np.copy(self.rawData.qIdx) )



    def discardDetectors(self, *qIdx):
        """ Remove detectors indices.
            The process modifies self.data.qIdx attribute that is used for fitting and plotting. """

        self.data = self.data._replace(qIdx = [val for val in self.data.qIdx if val not in qIdx])



    def resetDetectors(self):
        """ Reset qIdx entry to its original state, with all q values taken into account. """

        self.data = self.data._replace(qIdx = [idx for idx, val in enumerate(self.data.qVals)])

 



class BaseTypeDecorator(BaseType):
    """ Decorator for BaseType. Should be inherited by all child types. """

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData)

