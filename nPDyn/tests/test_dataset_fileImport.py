import os

path = os.path.dirname(os.path.abspath(__file__))

import unittest
import pytest

from nPDyn.dataParsers import processNexus, inxConvert


def test_import_qens_nexus():
    dataPath = path + "/sample_data/empty_cell_QENS_280K.nxs"
    data = processNexus(dataPath)
    assert data.shape == (1, 18, 1024)


def test_import_fws_nexus():
    dataPath = path + "/sample_data/lys_part_01_FWS.nxs"
    data = processNexus(dataPath, True)
    assert data.shape == (21, 18, 4)


def test_import_qens_inx():
    dataPath = path + "/sample_data/D_syn_fibers_QENS_300K.inx"
    data = inxConvert.convert(dataPath)
    assert data.shape == (1, 14, 793)


def test_import_fws_inx():
    dataPath = path + "/sample_data/D_syn_fibers_elastic_10to300K.inx"
    data = inxConvert.convert(dataPath, True)
    assert data.shape == (2880, 14, 1)
