"""

Classes
^^^^^^^

"""

import numpy as np

from collections import namedtuple

from nPDyn.dataTypes.baseType import BaseType
from nPDyn.fileFormatParser import guessFileFormat, readFile, fileImporters


class fECType(BaseType):
    """ This class inherits from :class:`baseType` class.

    """

    def __init__(self, fileName=None, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)



    def importData(self, fileFormat=None):
        """ Extract data from file and store them in *data* and *rawData* attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. 

        """

        if fileFormat:
            data = readFile(fileFormat, self.fileName, True)
        else:
            data = guessFileFormat(self.fileName, True)

        self.data    = data._replace(   qVals       = data.qVals,
                                        X           = data.X,
                                        intensities = data.intensities[0],
                                        errors      = data.errors[0],
                                        temp        = data.temp,
                                        norm        = False,
                                        qIdx        = data.qIdx )

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



    def discardNonElastic(self):
        """ Can be used to set to zero in intensity the region that does not pertain to the elastic
            peak. This can be useful for proper empty cell signal substraction. 

        """


        toDiscard = np.where( self.data.X != 0.0)

        newIntensities = self.data.intensities
        newIntensities[toDiscard] = 0

        self.data = self.data._replace(intensities = newIntensities)





class DataTypeDecorator(fECType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)
