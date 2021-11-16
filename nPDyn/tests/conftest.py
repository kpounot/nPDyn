import os

path = os.path.dirname(os.path.abspath(__file__))

import pytest

import numpy as np

from nPDyn.dataParsers import processNexus, inxConvert, IN16B_BATS
from nPDyn import Sample


@pytest.fixture
def emptyDataset():
    return Sample(np.array([]))


@pytest.fixture
def fullQENSDataset():
    qens = processNexus(path + "/sample_data/lys_part_01_QENS_before_280K.nxs")
    res = processNexus(path + "/sample_data/vana_QENS_280K.nxs")
    ec = processNexus(path + "/sample_data/empty_cell_QENS_280K.nxs")
    bkgd = processNexus(path + "/sample_data/D2O_QENS_280K.nxs")

    return qens, res, ec, bkgd


@pytest.fixture
def mixFWSDataset():
    fws = processNexus(path + "/sample_data/lys_part_01_FWS.nxs")
    res = processNexus(path + "/sample_data/vana_QENS_280K.nxs")
    ec = processNexus(path + "/sample_data/empty_cell_QENS_280K.nxs")
    bkgd = processNexus(path + "/sample_data/D2O_QENS_280K.nxs")

    return fws, res, ec, bkgd


@pytest.fixture
def fullFWSDataset():
    fws = processNexus(path + "/sample_data/lys_part_01_FWS.nxs")
    res = processNexus(path + "/sample_data/vana_QENS_280K.nxs")
    ec = processNexus(path + "/sample_data/empty_cell_FWS_280K.nxs")
    bkgd = processNexus(path + "/sample_data/D2O_FWS_280K.nxs")

    return fws, res, ec, bkgd


@pytest.fixture
def msdDataset():
    efws = inxConvert.convert(
        path + "/sample_data/D_syn_fibers_elastic_10to300K.inx", True
    )

    return efws


@pytest.fixture
def bats_data():
    data = IN16B_BATS(
        path + "/sample_data/bats_data/",
        detGroup=path + "/sample_data/IN16B_grouping_cycle201.xml",
    ).process()

    return data
