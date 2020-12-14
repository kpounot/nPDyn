import os

path = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])

import glob

import numpy as np

from nPDyn.dataParsers.xml_detector_grouping import IN16B_XML


def test_getPSDValues():
    dataFile = path + "/sample_data/IN16B_grouping_cycle201.xml"
    result = IN16B_XML(dataFile, 16)
    expectArray = np.array(
        [
            [56, 74],
            [56, 74],
            [52, 78],
            [39, 91],
            [45, 85],
            [42, 88],
            [38, 92],
            [35, 95],
            [31, 99],
            [28, 102],
            [24, 106],
            [21, 109],
            [17, 113],
            [14, 116],
            [10, 120],
            [7, 123],
        ]
    )
    assert np.array_equal(result.getPSDValues(), expectArray)
