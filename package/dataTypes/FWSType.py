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

class DataTypeDecorator(FWSType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)



