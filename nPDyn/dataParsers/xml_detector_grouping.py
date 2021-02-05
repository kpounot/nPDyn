"""

Classes
^^^^^^^

"""

import numpy as np

from defusedxml import cElementTree as ET


class IN16B_XML:
    """This class handles xml files containing detector grouping."""

    def __init__(self, sourceFile, numTubes):
        self.sourceFile = ET.parse(sourceFile)
        self.root = self.sourceFile.getroot()

        self.numTubes = numTubes

        self.nameList = None
        self.valList = None

        self._parseXML()

    def _parseXML(self):
        """ Find the groups of channels to be kept for each PSD. """
        nameList = []
        valList = []

        for child in self.root:
            nameList.append(child.attrib["name"])
            valList.append(child[0].attrib["val"])

        self.nameList = np.array(nameList)
        self.valList = np.array(valList)

    def getPSDValues(self):
        """Extract the values for each detector tube and
        return then in a numpy array using the same value
        range for each tube.

        """
        # Computes the size of the detector tube.
        size = int(self.valList[self.nameList == "tube1"][0]) - 1
        size /= self.numTubes

        nbrSD = self.valList.size - self.numTubes

        PSDValues = []
        for idx, val in enumerate(self.valList[nbrSD:]):
            group = np.array(val.split("-")).astype(int)
            group -= int(idx * size)
            PSDValues.append(group)

        return np.array(PSDValues).astype(int)
