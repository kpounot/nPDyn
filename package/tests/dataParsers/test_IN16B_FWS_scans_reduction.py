import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import glob

import numpy as np

from nPDyn.dataParsers.IN16B_FWS_scans_reduction import IN16B_FWS


class TestRawData_IN16B_FWS(unittest.TestCase):

    def test_dirImport(self):

        dataPath = testsPath + "sample_data/lys_part_01"

        result = IN16B_FWS(dataPath)
        result.process()

        self.assertTrue( len(result.qList) == 32, "Raw data from FWS could not be extracted correctly.")


    def test_listImport(self):

        dataPath = testsPath + "sample_data/lys_part_01"
        dataFiles = glob.glob(dataPath)

        result = IN16B_FWS(dataPath)
        result.process()

        self.assertTrue( len(result.qList) == 32, "Raw data from FWS could not be extracted correctly.")


