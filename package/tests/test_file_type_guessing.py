import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import numpy as np

from nPDyn.fileFormatParser import guessFileFormat, readFile, fileImporters
from nPDyn.dataParsers.inxConvert import convert


class TestFileTypeGuess(unittest.TestCase):

    def test_inx_file_noFWS(self):

        testFile = testsPath + "sample_data/D_syn_fibers_QENS_10K.inx"

        result = guessFileFormat(testFile)


        self.assertTrue(np.array_equal( result.qVals, 
                                        np.array([0.2182, 0.2685, 0.348 , 0.4593, 0.6026, 
                                                  0.783 , 0.9562, 1.1206, 1.2746, 1.417 , 
                                                  1.5463, 1.6613, 1.7611, 1.8446]) ), 
                        ".inx file type couldn't be guessed correctly")


    def test_mantid_file_noFWS(self):

        testFile = testsPath + "sample_data/vana_QENS_280K.nxs"

        result = guessFileFormat(testFile)


        self.assertTrue(np.array_equal( result.qVals.round(8), 
                                        np.array([0.19102381, 0.29274028, 0.43594975, 0.56783502, 0.69714328,
                                                  0.8232508 , 0.94556232, 1.06350556, 1.17653097, 1.28411303,
                                                  1.38575216, 1.48097673, 1.56934508, 1.65044748, 1.72390798,
                                                  1.7893861 , 1.84657839, 1.89521982]) ), 
                        ".nxs file type couldn't be guessed correctly")



    def test_mantid_file_FWS(self):

        testFile = testsPath + "sample_data/lys_part_01_FWS.nxs"

        result = guessFileFormat(testFile)

        self.assertTrue(np.array_equal( result.qVals.round(8), 
                                        np.array([0.19102381, 0.29274028, 0.43594975, 0.56783502, 0.69714328,
                                                  0.8232508 , 0.94556232, 1.06350556, 1.17653097, 1.28411303,
                                                  1.38575216, 1.48097673, 1.56934508, 1.65044748, 1.72390798,
                                                  1.7893861 , 1.84657839, 1.89521982]) ), 
                        ".nxs file type couldn't be guessed correctly")




