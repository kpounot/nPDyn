import numpy as np

from collections import namedtuple

from .baseType import BaseType
from ..fileFormatParser import guessFileFormat, readFile, fileImporters


class FWSType(BaseType):

    def __init__(self, fileName, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


    def importData(self, fileFormat=None):
        """ Extract data from file and store them in self.data and self.rawData attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. """

        if fileFormat:
            data = readFile(fileFormat, self.fileName, True)
        else:
            data = guessFileFormat(self.fileName, True)

        self.data       = data
        self.rawData    = self.data._replace(   qVals       = np.copy(self.data.qVals),
                                                X           = np.copy(self.data.X),
                                                intensities = np.copy(self.data.intensities),
                                                errors      = np.copy(self.data.errors),
                                                temp        = np.copy(self.data.temp),
                                                norm        = False,
                                                qIdx        = np.copy(self.data.qIdx) )




    def binData(self, binSize):
        """ Binning in energy dimension does nothing as it make no sense for FWS data type. """

        return



    def normalize(self):
        """ Normalizes data using a list of scaling factors from resolution function fit.
            It assumes that the scaling factor is in first position in the model parameters.
            There should be as many normalization factors as q-values in data. """

        normFList = np.array( [params[0][0] for params in self.resData.params] )

        #_Applying normalization
        self.data = self.data._replace( 
                                    intensities = self.data.intensities / normFList[np.newaxis,:,np.newaxis],
                                    errors      = self.data.errors / normFList[np.newaxis,:,np.newaxis],
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

        self.data = self.data._replace( intensities = self.data.intensities 
                                                                    - scaleFactor * ECFunc[np.newaxis,:,:] )


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)







class DataTypeDecorator(FWSType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)



