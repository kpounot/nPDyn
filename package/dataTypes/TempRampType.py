import numpy as np

from collections import namedtuple

from .baseType import BaseType
from ..fileFormatParser import guessFileFormat, readFile, fileImporters


class TempRampType(BaseType):

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



    def substractEC(self, ECFunc, subFactor):
        """ This method can be used to substract empty cell data from QENS data.
            Empty cell signal is rescaled using the given 'subFactor' prior to substraction. """

        #_Substracting the empty cell intensities from experimental data
        self.data = self.data._replace(intensities = self.data.intensities - np.max(ECFunc, axis=1) )


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)



    def normalize_usingLowTemp(self, nbrBins):
        """ Normalizes data using low temperature signal. An average is performed over the given
            number of bins for each q value and data are divided by the result. """


 
