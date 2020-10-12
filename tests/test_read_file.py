import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import numpy as np

from nPDyn.fileFormatParser import guessFileFormat, readFile, fileImporters
from nPDyn.dataParsers.inxConvert import convert


class TestReadFile(unittest.TestCase):

    def test_inx_file_noFWS(self):

        testFile = testsPath + "sample_data/D_syn_fibers_QENS_10K.inx"

        result = readFile('inx', testFile, False)

        self.assertTrue(np.array_equal( result.qVals, 
                                        np.array([0.2182, 0.2685, 0.348 , 0.4593, 0.6026, 
                                                  0.783 , 0.9562, 1.1206, 1.2746, 1.417 , 
                                                  1.5463, 1.6613, 1.7611, 1.8446]) ), 
                        ".inx file type couldn't be guessed correctly")



