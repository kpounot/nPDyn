import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import numpy as np

import nPDyn
from nPDyn.dataTypes.baseType import BaseType


class TestBaseType(unittest.TestCase):

    def test_import_with_fileFormat(self):
        data = BaseType(fileName=testsPath + 'sample_data/vana_QENS_280K.nxs')
        data.importData(fileFormat='mantid')
        self.assertIsNotNone(data.data.qIdx)

    def test_assignResData(self):
        dataPath = testsPath + 'sample_data/lys_part_01_QENS_before_280K.nxs'
        data = nPDyn.Dataset(QENSFiles=[dataPath])
        data.importFiles(resFiles=[testsPath + 'sample_data/vana_QENS_280K.nxs'])
        data.datasetList[0].assignResData(data.resData[0])
        self.assertIsNotNone(data.datasetList[0].resData)

    def test_assignECData(self):
        dataPath = testsPath + 'sample_data/lys_part_01_QENS_before_280K.nxs'
        data = nPDyn.Dataset(QENSFiles=[dataPath])
        data.importFiles(ECFile=testsPath + 'sample_data/empty_cell_QENS_280K.nxs')
        data.datasetList[0].assignECData(data.ECData)
        self.assertIsNotNone(data.datasetList[0].ECData)

    def test_assignD2OData(self):
        dataPath = testsPath + 'sample_data/lys_part_01_QENS_before_280K.nxs'
        data = nPDyn.Dataset(QENSFiles=[dataPath])
        data.importFiles(D2OFile=testsPath + 'sample_data/D2O_QENS_280K.nxs')
        data.datasetList[0].assignD2OData(data.D2OData)
        self.assertIsNotNone(data.datasetList[0].D2OData)

    def test_getD2OSignal(self):
        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(
            QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
            resFiles=[dataPath + 'vana_QENS_280K.nxs'],
            D2OFile=dataPath + 'D2O_QENS_280K.nxs')

        D2OSignal = data.datasetList[0].getD2OSignal(5)

        self.assertEqual(D2OSignal.shape, (1024,))
