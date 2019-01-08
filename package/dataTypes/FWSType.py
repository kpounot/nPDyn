import numpy as np

from collections import namedtuple

from .baseType import BaseType
from ..fileFormatParser import guessFileFormat, readFile, fileImporters


class FWSType(BaseType):

    def __init__(self, fileName, data=None):
        super().__init__(fileName, data)



    def importData(self, fileFormat=None):
        """ Extract data from file and store them in self.data and self.rawData attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. """

        if fileFormat:
            data = readFile(fileFormat, self.fileName, True)
        else:
            data = guessFileFormat(self.fileName, True)

        self.data       = data
        self.rawData    = data


