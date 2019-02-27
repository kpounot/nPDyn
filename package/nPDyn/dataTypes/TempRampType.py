import numpy as np

from collections import namedtuple

from .baseType import BaseType
from ..fileFormatParser import guessFileFormat, readFile, fileImporters


class TempRampType(BaseType):

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



    def normalize_usingLowTemp(self, nbrBins):
        """ Normalizes data using low temperature signal. An average is performed over the given
            number of bins for each q value and data are divided by the result. """

        normFList = np.mean(self.data.intensities[:,:nbrBins], axis=1)[:,np.newaxis]

        self.data = self.data._replace( intensities = self.data.intensities / normFList,
                                        errors      = self.data.errors / normFList,
                                        norm=True )




    def substractEC(self, scaleFactor=0.8):
        """ This method can be used to substract empty cell data from QENS data.
            Empty cell signal is rescaled using the given 'subFactor' prior to substraction. """

        #_Compute the fitted Empty Cell function
        ECFunc = []
        for qIdx, qVal in enumerate(self.data.qVals):
            ECFunc.append( self.ECData.model( 0.0, *self.ECData.params[qIdx][0] ) )

        ECFunc = np.array( ECFunc )[:,np.newaxis]

        #_If data are normalized, uses the same normalization factor for empty cell data
        if self.data.norm:
            normFList = np.array([params[0][0] for params in self.ECData.params])[:,np.newaxis]
            ECFunc /= normFList


        #_Substracting the empty cell intensities from experimental data
        self.data = self.data._replace(intensities = self.data.intensities - scaleFactor * ECFunc )


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)




class DataTypeDecorator(TempRampType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)




 
