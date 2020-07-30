"""

Classes
^^^^^^^

"""

import numpy as np

from nPDyn.dataTypes.FWSType import FWSType
from nPDyn.fileFormatParser import guessFileFormat, readFile


class fD2OType(FWSType):
    """ This class inherits from :class:`baseType` class.

    """

    def __init__(self, fileName=None, data=None, rawData=None,
                 resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


        self.volFraction = 0.95


    def importData(self, fileFormat=None):
        """ Extract data from file and store them in
            *data* and *rawData* attributes.

            If no fileFormat is given, tries to guess it, try
            hdf5 format if format cannot be guessed.

        """

        if fileFormat:
            data = readFile(fileFormat, self.fileName, True)
        else:
            data = guessFileFormat(self.fileName, True)

        self.data    = data._replace(
            qVals       = data.qVals,
            X           = data.X,
            intensities = data.intensities[0],
            errors      = data.errors[0],
            temp        = data.temp,
            norm        = False,
            qIdx        = data.qIdx)

        self.rawData    = self.data._replace(
            qVals       = np.copy(self.data.qVals),
            X           = np.copy(self.data.X),
            intensities = np.copy(self.data.intensities),
            errors      = np.copy(self.data.errors),
            temp        = np.copy(self.data.temp),
            norm        = False,
            qIdx        = np.copy(self.data.qIdx))


    def binData(self, binSize):
        """ Binning in energy dimension does nothing as it
            makes no sense for FWS data type. """

        return




    def getD2OSignal(self):
        """ Returns the D2O signal for each q-value. """

        D2OSignal = self.data.intensities

        return D2OSignal * self.volFraction




class DataTypeDecorator(fD2OType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data,
                         dataType.rawData, dataType.resData,
                         dataType.D2OData, dataType.ECData)
