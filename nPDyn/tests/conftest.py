import os

path = os.path.dirname(os.path.abspath(__file__))

import pytest

from nPDyn import Dataset


@pytest.fixture
def emptyDataset():
    return Dataset()


@pytest.fixture
def fullQENSDataset():
    return Dataset(
        QENSFiles=[path + "/sample_data/lys_part_01_QENS_before_280K.nxs"],
        resFiles=[path + "/sample_data/vana_QENS_280K.nxs"],
        ECFile=path + "/sample_data/empty_cell_QENS_280K.nxs",
        D2OFile=path + "/sample_data/D2O_QENS_280K.nxs",
    )


@pytest.fixture
def mixFWSDataset():
    return Dataset(
        FWSFiles=[path + "/sample_data/lys_part_01_FWS.nxs"],
        resFiles=[path + "/sample_data/vana_QENS_280K.nxs"],
        ECFile=path + "/sample_data/empty_cell_QENS_280K.nxs",
        D2OFile=path + "/sample_data/D2O_QENS_280K.nxs",
    )


@pytest.fixture
def fullFWSDataset():
    return Dataset(
        FWSFiles=[path + "/sample_data/lys_part_01_FWS.nxs"],
        resFiles=[path + "/sample_data/vana_QENS_280K.nxs"],
        fECFile=path + "/sample_data/empty_cell_FWS_280K.nxs",
        fD2OFile=path + "/sample_data/D2O_FWS_280K.nxs",
    )


@pytest.fixture
def msdDataset():
    return Dataset(
        FWSFiles=[path + "/sample_data/D_syn_fibers_elastic_10to300K.inx"]
    )
