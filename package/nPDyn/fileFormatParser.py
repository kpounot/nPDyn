import sys

import re

from nPDyn.dataParsers import *



fileImporters = {   'inx'   : inxConvert.convert,
                    'hdf5'  : h5process.processData }



def readFile(fileFormat, dataFile, FWS=False):
    """ Extract data from file using the given file format. 
        
        fileFormat can be 'hdf5' or 'inx' (other to come later) 

    """

    try:
        data = fileImporters[fileFormat](dataFile, FWS)
    except Exception as e:
        print(e)
        return

    return data



def guessFileFormat(dataFile, FWS=False):
    """ Tries to guess file format based on file name.

        In case it cannot be guessed, the default try is hdf5 format. 

    """


    if re.search('.inx', dataFile):
        return readFile('inx', dataFile, FWS) 

    elif re.search('.nxs', dataFile):
        return readFile('hdf5', dataFile, FWS)

    else: #_Try a generic hdf5 file (.h5 extension or else)
        return readFile('hdf5', dataFile, FWS)


