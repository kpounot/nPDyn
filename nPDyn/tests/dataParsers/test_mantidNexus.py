import os

path = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])

import glob

import numpy as np

from nPDyn.dataParsers.mantidNexus import processNexus


def test_import_QENS():
    dataPath = path + "/sample_data/vana_QENS_280K.nxs"
    result = processNexus(dataPath)
    assert result.intensities.shape == (1, 18, 1024)


def test_import_FWS():
    dataPath = path + "/sample_data/lys_part_01_FWS.nxs"
    result = processNexus(dataPath, True)
    assert result.intensities.shape == (21, 18, 4)
