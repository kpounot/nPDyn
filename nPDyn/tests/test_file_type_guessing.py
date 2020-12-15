import os

path = os.path.dirname(os.path.abspath(__file__))

import pytest
from pytest import approx

import numpy as np

from nPDyn.fileFormatParser import guessFileFormat, readFile, fileImporters
from nPDyn.dataParsers.inxConvert import convert


def test_inx_file_noFWS():
    testFile = path + "/sample_data/D_syn_fibers_QENS_10K.inx"
    result = guessFileFormat(testFile)
    assert result.qVals == approx(
        np.array(
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
    )


def test_mantid_file_noFWS():
    testFile = path + "/sample_data/vana_QENS_280K.nxs"
    result = guessFileFormat(testFile)
    assert result.qVals.round(8) == approx(
        np.array(
            [
                0.19102381,
                0.29274028,
                0.43594975,
                0.56783502,
                0.69714328,
                0.8232508,
                0.94556232,
                1.06350556,
                1.17653097,
                1.28411303,
                1.38575216,
                1.48097673,
                1.56934508,
                1.65044748,
                1.72390798,
                1.7893861,
                1.84657839,
                1.89521982,
            ]
        )
    )


def test_mantid_file_FWS():
    testFile = path + "/sample_data/lys_part_01_FWS.nxs"
    result = guessFileFormat(testFile)
    assert result.qVals.round(8) == approx(
        np.array(
            [
                0.19102381,
                0.29274028,
                0.43594975,
                0.56783502,
                0.69714328,
                0.8232508,
                0.94556232,
                1.06350556,
                1.17653097,
                1.28411303,
                1.38575216,
                1.48097673,
                1.56934508,
                1.65044748,
                1.72390798,
                1.7893861,
                1.84657839,
                1.89521982,
            ]
        )
    )
