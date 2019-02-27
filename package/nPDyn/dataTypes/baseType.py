import numpy as np

from collections import namedtuple

from itertools import count

from ..dataManipulation.binData import binData
from ..fileFormatParser import guessFileFormat, readFile, fileImporters


class BaseType:

    def __init__(self, fileName=None, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        """ Initialize a base type that will be inherited by all other specialized types through its decorator. 
        
            Input:  fileName    -> name of the file being read
                    data        -> resulting namedtuple from data parsers 
                    rawData     -> used by the decorator """

        self.fileName   = fileName
        self.data       = data 
        self.rawData    = rawData #_Used to reset data to its initial state

        self.resData    = resData #_For use with sample data types
        self.D2OData    = D2OData #_For use with sample data types
        self.ECData     = ECData 



    def importData(self, fileFormat=None):
        """ Extract data from file and store them in self.data and self.rawData attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. """

        if fileFormat:
            data = readFile(fileFormat, self.fileName)
        else:
            data = guessFileFormat(self.fileName)

        self.data       = data
        self.rawData    = self.data._replace(   qVals       = np.copy(self.data.qVals),
                                                X           = np.copy(self.data.X),
                                                intensities = np.copy(self.data.intensities),
                                                errors      = np.copy(self.data.errors),
                                                temp        = np.copy(self.data.temp),
                                                norm        = False,
                                                qIdx        = np.copy(self.data.qIdx) )



    def binData(self, binSize):
        """ Bin self.data attribute using the given binSize. """

        self.data = binData(self.data, binSize)



    def normalize(self):
        """ Normalizes data using a list of scaling factors from resolution function fit.
            It assumes that the scaling factor is in first position in the model parameters.
            There should be as many normalization factors as q-values in data. """

        normFList = np.array( [params[0][0] for params in self.resData.params] )

        #_Applying normalization
        self.data = self.data._replace( intensities = self.data.intensities / normFList[:,np.newaxis],
                                        errors      = self.data.errors / normFList[:,np.newaxis],
                                        norm        = True )





    def substractEC(self, scaleFactor=0.8):
        """ Use the assigned empty cell data for substraction to loaded data.
            
            Empty cell data are scaled using the given scaleFactor prior to substraction. """

        #_Compute the fitted Empty Cell function
        ECFunc = []
        for qIdx, qVal in enumerate(self.data.qVals):
            ECFunc.append( self.ECData.model( self.data.X, *self.ECData.params[qIdx][0] ) )

        ECFunc = np.array( ECFunc )

        #_If data are normalized, uses the same normalization factor for empty cell data
        if self.data.norm:
            normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
            ECFunc /= normFList

        self.data = self.data._replace( intensities = self.data.intensities - scaleFactor * ECFunc )


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)




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




    def assignResData(self, resData):
        """ Sets self.resData attribute to the given one, a ResType instance that can be used by fitting 
            functions in QENS or FWS types. """

        self.resData = resData



    def assignD2OData(self, D2OData):
        """ Sets self.D2OData attribute to the given one, a D2OType instance that can be used by fitting
            functions in QENS or FWS types. """

        self.D2OData = D2OData
 

    def assignECData(self, ECData):
        """ Sets self.ECData attribute to the given one, a ECType instance that can be used by fitting
            functions in QENS or FWS types. """

        self.ECData = ECData
 




class DataTypeDecorator(BaseType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)

