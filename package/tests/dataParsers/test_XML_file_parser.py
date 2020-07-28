import os

filePath = os.path.abspath(__file__)
testsPath = filePath[:filePath.find('tests/')+6]

import unittest

import glob

import numpy as np

from nPDyn.dataParsers.xml_detector_grouping import IN16B_XML


class TestRawData_IN16B_XML(unittest.TestCase):

    def test_getPSDValues(self):

        dataFile = testsPath + "sample_data/IN16B_grouping_cycle201.xml"

        result = IN16B_XML(dataFile, 16)

        expectArray = np.array([[56,  74],
                                [56,  74],
                                [52,  78],
                                [39,  91],
                                [45,  85],
                                [42,  88],
                                [38,  92],
                                [35,  95],
                                [31,  99],
                                [28, 102],
                                [24, 106],
                                [21, 109],
                                [17, 113],
                                [14, 116],
                                [10, 120],
                                [7, 123]])

        self.assertTrue( np.array_equal(result.getPSDValues(), expectArray), 
                         "Data from XML file could not be extracted correctly.")


