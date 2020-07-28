import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import numpy as np

import nPDyn



class TestBaseType(unittest.TestCase):

    def test_remove_dataset(self):

        dataPath = testsPath + 'sample_data/'

        multipleQENS = nPDyn.Dataset(QENSFiles=[dataPath + 'lys_part_01_QENS_before_280K.nxs',
                                                     dataPath + 'lys_part_01_QENS_before_280K.nxs'])

        multipleQENS.removeDataset(0)
        self.assertEqual(len(multipleQENS.datasetList), 1)







     
    





