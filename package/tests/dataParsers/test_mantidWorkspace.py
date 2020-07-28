import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import glob

import numpy as np

from nPDyn.dataParsers.mantidWorkspace import processData


class TestInxData(unittest.TestCase):

    def test_import_QENS(self):

        dataPath = testsPath + "sample_data/vana_QENS_280K.nxs"

        result = processData(dataPath)

        self.assertTrue(result.intensities.shape == (18, 1024))


    def test_import_FWS(self):


        dataPath = testsPath + "sample_data/lys_part_01_FWS.nxs"

        result = processData(dataPath, True)


        self.assertTrue(result.intensities.shape == (21, 18, 4))


