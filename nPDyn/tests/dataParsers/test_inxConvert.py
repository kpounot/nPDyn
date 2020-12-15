import os

path = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])

import glob

import numpy as np

from nPDyn.dataParsers.inxConvert import convert


def test_import_QENS():
    dataPath = path + "/sample_data/D_syn_fibers_QENS_10K.inx"
    result = convert(dataPath)
    expect = np.array(
        [
            0.2182,
            0.2685,
            0.348,
            0.4593,
            0.6026,
            0.783,
            0.9562,
            1.1206,
            1.2746,
            1.417,
            1.5463,
            1.6613,
            1.7611,
            1.8446,
        ]
    )
    assert np.array_equal(result.qVals, expect)


def test_import_FWS():
    dataPath = path + "/sample_data/D_syn_fibers_elastic_10to300K.inx"
    result = convert(dataPath)
    expect = np.array(
        [
            0.2182,
            0.2685,
            0.348,
            0.4593,
            0.6026,
            0.783,
            0.9562,
            1.1206,
            1.2746,
            1.417,
            1.5463,
            1.6613,
            1.7611,
            1.8446,
        ]
    )
    assert np.array_equal(result.qVals, expect)
