import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import numpy as np

import nPDyn



class TestDataProcess(unittest.TestCase):

    def test_remove_dataset(self):

        dataPath = testsPath + 'sample_data/'

        multipleQENS = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs',
                                                     dataPath + 'lys_part_01_QENS_before_280K.nxs'])

        multipleQENS.removeDataset(0)
        self.assertEqual(len(multipleQENS.datasetList), 1)


    def test_bin_data_QENS(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'])


        data.binDataset(5, 0)
        self.assertEqual(data.datasetList[0].data.intensities.shape[1], 204)


        
    def test_bin_resData(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'])


        data.binResData(5)

        self.assertEqual(data.resData[0].data.intensities.shape[1], 204)
        


    def test_binAll_data_QENSonly(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
                                      resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                                      ECFile=dataPath + 'empty_cell_QENS_280K.nxs',
                                      D2OFile=dataPath + 'D2O_QENS_280K.nxs')


        data.binAll(5)
        self.assertEqual(data.datasetList[0].data.intensities.shape[1], 204)
        self.assertEqual(data.resData[0].data.intensities.shape[1], 204)
        self.assertEqual(data.ECData.data.intensities.shape[1], 204)
        self.assertEqual(data.D2OData.data.intensities.shape[1], 204)

 
    def test_binAll_data_mixQENSandFWS(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                                      resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                                      ECFile=dataPath + 'empty_cell_QENS_280K.nxs',
                                      D2OFile=dataPath + 'D2O_QENS_280K.nxs')


        data.binAll(5)
        self.assertEqual(data.resData[0].data.intensities.shape[1], 204)
        self.assertEqual(data.ECData.data.intensities.shape[1], 204)
        self.assertEqual(data.D2OData.data.intensities.shape[1], 204)


    def test_resetAll(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                                      resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                                      ECFile=dataPath + 'empty_cell_QENS_280K.nxs',
                                      D2OFile=dataPath + 'D2O_QENS_280K.nxs')

        data.binAll(5)
        data.normalize_usingResFunc()

        data.resetAll()

        self.assertEqual(data.resData[0].data.intensities.shape[1], 1024)
        self.assertEqual(data.ECData.data.intensities.shape[1], 1024)
        self.assertEqual(data.D2OData.data.intensities.shape[1], 1024)

        self.assertTrue(data.datasetList[0].data.intensities[0].sum() < 0.3)


    def test_resetDataset(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'])

        data.binAll(5)

        data.resetAll()

        self.assertEqual(data.datasetList[0].data.intensities.shape[1], 1024)


    def test_normalize_QENS_fromQENSres(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'])

        data.normalize_usingResFunc()

        self.assertTrue(data.datasetList[0].data.intensities.sum() > 135)

    
    def test_normalize_FWS_fromQENSres(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'])

        data.normalize_usingResFunc()

        self.assertTrue(data.datasetList[0].data.intensities.sum() > 13)



    def test_normalize_ENS_fromLowTemp(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(TempRampFiles=[dataPath + 'D_syn_fibers_elastic_10to300K.inx'])

        data.normalize_ENS_usingLowTemp()

        self.assertTrue(data.datasetList[0].data.intensities.sum() < 35561)



    def test_subtractEC_fromQENS_withModel(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
                             ECFile=dataPath + 'empty_cell_QENS_280K.nxs')

        data.subtract_EC()

        self.assertTrue(data.datasetList[0].data.intensities.sum() < 7.1)



    def test_subtractEC_fromQENS_withoutModel(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
                             ECFile=dataPath + 'empty_cell_QENS_280K.nxs')

        data.subtract_EC(useModel=False)

        self.assertTrue(data.datasetList[0].data.intensities.sum() < 7)



    def test_subtractEC_fromFWS_withQENS(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                             ECFile=dataPath + 'empty_cell_QENS_280K.nxs')

        data.subtract_EC()

        self.assertTrue(data.datasetList[0].data.intensities.sum() < 1.9)



    def test_subtractEC_fromFWS_withFWS(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                             fECFile=dataPath + 'empty_cell_FWS_280K.nxs')

        data.subtract_EC()

        self.assertTrue(data.datasetList[0].data.intensities.sum() < 1.9)



    def test_absorptionCorr_dataQENS_ecQENS(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                             ECFile=dataPath + 'empty_cell_QENS_280K.nxs')

        data.absorptionCorrection(D2O=False, res=False)

        self.assertTrue(data.datasetList[0].data.intensities.sum() > 11)



    def test_absorptionCorr_dataFWS_ecQENS(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                             ECFile=dataPath + 'empty_cell_QENS_280K.nxs')

        data.absorptionCorrection(D2O=False, res=False)

        self.assertTrue(data.datasetList[0].data.intensities.sum() < 1.41)



    def test_absorptionCorr_dataFWS_fecQENS(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                             fECFile=dataPath + 'empty_cell_FWS_280K.nxs')

        data.absorptionCorrection(D2O=False, res=False)

        self.assertTrue(data.datasetList[0].data.intensities.sum() < 1.97)



    def test_absorptionCorr_dataQENS_ecQENS_D2O_and_res(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                             ECFile=dataPath + 'empty_cell_QENS_280K.nxs',
                             D2OFile=dataPath + 'D2O_QENS_280K.nxs')

        data.absorptionCorrection()

        self.assertTrue(data.resData[0].data.intensities.sum() > 34)
        self.assertTrue(data.D2OData.data.intensities.sum() > 98)



    def test_absorptionCorr_dataFWS_fecQENS_fD2O_and_res(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'],
                             resFiles=[dataPath + 'vana_QENS_280K.nxs'],
                             fECFile=dataPath + 'empty_cell_FWS_280K.nxs',
                             fD2OFile=dataPath + 'D2O_FWS_280K.nxs')

        data.absorptionCorrection()

        self.assertTrue(data.resData[0].data.intensities.sum() > 36)
        self.assertTrue(data.fD2OData.data.intensities.sum() > 0.09712)




    def test_discardDetectors(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'])

        data.discardDetectors([0,1,2,3])

        self.assertTrue(data.datasetList[0].data.qIdx.size == 14)



    def test_setQRange(self):

        dataPath = testsPath + 'sample_data/'

        data = nPDyn.Dataset(FWSFiles=[dataPath + 'lys_part_01_FWS.nxs'])

        data.setQRange(0.4,1.7)

        self.assertTrue(data.datasetList[0].data.qIdx.size == 12)







     
    





