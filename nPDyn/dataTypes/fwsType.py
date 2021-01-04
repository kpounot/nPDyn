"""Specialized type for fixed-window scans (FWS)."""

import numpy as np

from nPDyn.dataTypes.baseType import BaseType
from nPDyn.fileFormatParser import guessFileFormat, readFile
from nPDyn.dataManipulation.binData import binData


class FWSType(BaseType):
    def __init__(
        self,
        fileName,
        data=None,
        rawData=None,
        resData=None,
        D2OData=None,
        ECData=None,
        model=None,
    ):
        """Specialized class for fixed-window scans (FWS).

        This class inherits from :class:`baseType` class.

        """
        super().__init__(
            fileName, data, rawData, resData, D2OData, ECData, model
        )

    def importData(self, fileFormat=None):
        """Extract data from file.

        Data are stored in *data* and *rawData* attributes.

        If no fileFormat is given, tries to guess it, try hdf5 format
        if format cannot be guessed.

        Parameters
        ----------
        fileFormat : str, optional
            file format to be used, can be 'inx' or 'mantid'

        """
        if fileFormat:
            data = readFile(fileFormat, self.fileName, True)
        else:
            data = guessFileFormat(self.fileName, True)

        self.data = data
        self._rawData = self.data._replace(
            qVals=np.copy(self.data.qVals),
            times=np.copy(self.data.times),
            intensities=np.copy(self.data.intensities),
            errors=np.copy(self.data.errors),
            temps=np.copy(self.data.temps),
            norm=False,
            qIdx=np.copy(self.data.qIdx),
            energies=np.copy(self.data.energies),
            observable=np.copy(self.data.observable),
            observable_name=np.copy(self.data.observable_name),
        )

    def binData(self, binSize, axis):
        """Bin data for selected dimension except 'energies'."""
        if axis == "energies":
            return
        else:
            self.data = binData(self.data, binSize, axis)
