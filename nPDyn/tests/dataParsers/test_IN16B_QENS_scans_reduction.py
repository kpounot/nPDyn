import os

path = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])

import glob

import numpy as np

from nPDyn.dataParsers.in16b_qens_scans_reduction import IN16B_QENS


def test_dirImport():
    dataPath = path + "/sample_data/vanadium/"
    result = IN16B_QENS(dataPath)
    result.process()
    assert len(result.qList) == 4


def test_listImport():
    dataPath = path + "/sample_data/vanadium/"
    dataFiles = glob.glob(dataPath)
    result = IN16B_QENS(dataPath)
    result.process()
    assert len(result.qList) == 4
