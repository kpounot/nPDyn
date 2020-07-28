import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import numpy as np

import nPDyn


class TestFileImport(unittest.TestCase):

    def test_import_ec(self):

        dataPath = testsPath + 'sample_data/empty_cell_QENS_280K.nxs'

        data = nPDyn.Dataset(ECFile=dataPath)

        self.assertIsNotNone(data.ECData.data.qVals)


    def test_import_fec(self):

        dataPath = testsPath + 'sample_data/lys_part_01_FWS.nxs'

        data = nPDyn.Dataset(fECFile=dataPath)

        self.assertIsNotNone(data.fECData.data.qVals)


    def test_import_tec(self):

        dataPath = testsPath + 'sample_data/D_syn_fibers_elastic_10to300K.inx'

        data = nPDyn.Dataset(tECFile=dataPath)

        self.assertIsNotNone(data.tECData.data.qVals)


    def test_import_resFiles(self):

        dataPath = testsPath + 'sample_data/D_syn_fibers_QENS_10K.inx'

        dataList = [dataPath, dataPath]

        data = nPDyn.Dataset(resFiles=dataList)

        self.assertIsNotNone(data.resData[1].data.qVals)


    def test_import_D2O(self):

        dataPath = testsPath + 'sample_data/D_syn_fibers_QENS_300K.inx'

        data = nPDyn.Dataset(D2OFile=dataPath)

        self.assertIsNotNone(data.D2OData.data.qVals)


    def test_import_fD2O(self):

        dataPath = testsPath + 'sample_data/lys_part_01_FWS.nxs'

        data = nPDyn.Dataset(fD2OFile=dataPath)

        self.assertIsNotNone(data.fD2OData.data.qVals)


    def test_import_QENS(self):

        dataPath = testsPath + 'sample_data/D_syn_fibers_QENS_300K.inx'

        dataList = [dataPath, dataPath]

        data = nPDyn.Dataset(QENSFiles=dataList)

        self.assertIsNotNone(data.datasetList[1].data.qVals)


    def test_import_FWS(self):

        dataPath = testsPath + 'sample_data/lys_part_01_FWS.nxs'

        dataList = [dataPath, dataPath]

        data = nPDyn.Dataset(FWSFiles=dataList)

        self.assertIsNotNone(data.datasetList[1].data.qVals)


    def test_import_TempRamp(self):

        dataPath = testsPath + 'sample_data/D_syn_fibers_elastic_10to300K.inx'

        dataList = [dataPath, dataPath]

        data = nPDyn.Dataset(TempRampFiles=dataList)

        self.assertIsNotNone(data.datasetList[1].data.qVals)


if __name__ == '__main__':
    unittest.main()
