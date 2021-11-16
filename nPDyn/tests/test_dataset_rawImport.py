import os

path = os.path.dirname(os.path.abspath(__file__))

import pytest

import numpy as np

from nPDyn.dataParsers import IN16B_FWS, IN16B_QENS, IN16B_BATS


def test_import_IN16B_QENS():
    dataPath = path + "/sample_data/vanadium/"
    data = IN16B_QENS(
        dataPath, detGroup=path + "/sample_data/IN16B_grouping_cycle201.xml"
    ).process()
    assert data.shape == (1, 18, 1004)


def test_import_IN16B_FWS():
    dataPath = path + "/sample_data/lys_part_01/"
    data = IN16B_FWS(
        dataPath, detGroup=path + "/sample_data/IN16B_grouping_cycle201.xml"
    ).process()
    assert data.shape == (8, 18, 4)


def test_import_IN16B_BATS():
    dataPath = path + "/sample_data/bats_data/316112:316122.nxs"
    data = IN16B_BATS(
        dataPath, detGroup=path + "/sample_data/IN16B_grouping_cycle201.xml"
    ).process()
    assert data.shape == (1, 19, 817)
