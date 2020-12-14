import os

path = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])

import glob

import numpy as np

from nPDyn.dataParsers.in16b_fws_scans_reduction import IN16B_FWS


def test_dirImport():
    dataPath = path + "/sample_data/lys_part_01"
    result = IN16B_FWS(dataPath)
    result.process()
    assert len(result.qList) == 32


def test_listImport():
    dataPath = path + "/sample_data/lys_part_01"
    dataFiles = glob.glob(dataPath)
    result = IN16B_FWS(dataPath)
    result.process()
    assert len(result.qList) == 32
