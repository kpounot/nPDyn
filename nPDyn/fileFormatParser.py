"""This module handles the different file format that can be recognized
by nPDyn and it calls the roght algorithm to parse them.

"""
import re

from nPDyn.dataParsers import processNexus, WorkspaceHandler, convert

fileImporters = {
    "inx": convert,
    "nexus": processNexus,
    "mantid": WorkspaceHandler,
}


def readFile(fileFormat, dataFile, FWS=False):
    """Extract data from file using the given file format.

    Parameters
    ----------
    fileFormat : {'nexus', 'inx', 'mantid'}
        Format of the data to be read.
        Options are:
            - 'nexus', a Mantid generated .nxs file
            - 'inx', a SLAW generated .inx file
            - 'mantid', a Mantid `WorkspaceGroup` or `MatrixWorkspace`
    dataFile : str or object
        Data file to be read.
    FWS : bool
        Whether the data are obtained from fixed-window scans or not.

    """
    data = fileImporters[fileFormat](dataFile, FWS)
    return data


def guessFileFormat(dataFile, FWS=False):
    """Tries to guess file format based on file name.

    In case it cannot be guessed, the default try is hdf5 format.

    """
    if isinstance(dataFile, str):
        if re.search(".inx", dataFile):
            return readFile("inx", dataFile, FWS)
        elif re.search(".nxs", dataFile):
            return readFile("nexus", dataFile, FWS)
        else:
            raise ValueError(
                "The file format cannot be guessed.\n"
                "Please provide the format to be used."
            )
    else:  # Try a Mantid Workspace
        return readFile("mantid", dataFile, FWS)
