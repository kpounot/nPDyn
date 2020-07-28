import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import glob

import numpy as np

from nPDyn.dataParsers.IN16B_QENS_scans_reduction import IN16B_QENS


class TestRawData_IN16B_QENS(unittest.TestCase):

    def test_dirImport(self):

        dataPath = testsPath + "sample_data/vanadium/"

        result = IN16B_QENS(dataPath)
        result.process()

        self.assertTrue( len(result.qList) == 4, "Raw data from vanadium could not be extracted correctly.")


    def test_listImport(self):

        dataPath = testsPath + "sample_data/vanadium/"
        dataFiles = glob.glob(dataPath)

        result = IN16B_QENS(dataPath)
        result.process()

        self.assertTrue( len(result.qList) == 4, "Raw data from vanadium could not be extracted correctly.")


