import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import numpy as np

import nPDyn


class TestRawImport(unittest.TestCase):

    def test_import_QENS(self):

        dataPath = testsPath + 'sample_data/vanadium/'

        data = nPDyn.Dataset()
        data.importRawData(dataPath, 'IN16B', 'QENS')

        self.assertIsNotNone(data.datasetList[0].data.qVals)


    def test_import_FWS(self):

        dataPath = testsPath + 'sample_data/lys_part_01/'

        data = nPDyn.Dataset()
        data.importRawData(dataPath, 'IN16B', 'FWS')

        self.assertIsNotNone(data.datasetList[0].data.qVals)

 
    def test_import_res(self):

        dataPath = testsPath + 'sample_data/vanadium/'

        data = nPDyn.Dataset()
        data.importRawData(dataPath, 'IN16B', 'res')

        self.assertIsNotNone(data.resData[0].data.qVals)

 
    def test_import_ec(self):

        dataPath = testsPath + 'sample_data/vanadium'

        data = nPDyn.Dataset()
        data.importRawData(dataPath, 'IN16B', 'ec')

        self.assertIsNotNone(data.ECData.data.qVals)

 
    def test_import_fec(self):

        dataPath = testsPath + 'sample_data/lys_part_01'

        data = nPDyn.Dataset()
        data.importRawData(dataPath, 'IN16B', 'fec')

        self.assertIsNotNone(data.fECData.data.qVals)

 
    def test_import_D2O(self):

        dataPath = testsPath + 'sample_data/vanadium'

        data = nPDyn.Dataset()
        data.importRawData(dataPath, 'IN16B', 'D2O')

        self.assertIsNotNone(data.D2OData.data.qVals)

 
    def test_import_fD2O(self):

        dataPath = testsPath + 'sample_data/lys_part_01'

        data = nPDyn.Dataset()
        data.importRawData(dataPath, 'IN16B', 'fD2O')

        self.assertIsNotNone(data.fD2OData.data.qVals)






