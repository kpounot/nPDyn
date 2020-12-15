import os

path = os.path.dirname(os.path.abspath(__file__))

import unittest

import pytest

import nPDyn

try:
    from nPDyn.lib.pyabsco import py_absco_slab, py_absco_tube

    _HAS_ABSCO = True
except ImportError:
    _HAS_ABSCO = False


def test_remove_dataset(fullQENSDataset):
    fullQENSDataset.removeDataset(0)
    assert len(fullQENSDataset.dataList) == 0


def test_bin_data_QENS_energies(fullQENSDataset):
    fullQENSDataset.binDataset(5, 0)
    assert fullQENSDataset.dataList[0].data.intensities.shape[2] == 204


def test_bin_resData_energies(fullQENSDataset):
    fullQENSDataset.binResData(5)
    assert fullQENSDataset.resData[0].data.intensities.shape[2] == 204


def test_binAll_data_QENSonly_energies(fullQENSDataset):
    fullQENSDataset.binAll(5)
    assert fullQENSDataset.dataList[0].data.intensities.shape[2] == 204
    assert fullQENSDataset.resData[0].data.intensities.shape[2] == 204
    assert fullQENSDataset.ECData.data.intensities.shape[2] == 204
    assert fullQENSDataset.D2OData.data.intensities.shape[2] == 204


def test_binAll_data_mixQENSandFWS_energies(mixFWSDataset):
    mixFWSDataset.binAll(5)
    assert mixFWSDataset.resData[0].data.intensities.shape[2] == 204
    assert mixFWSDataset.ECData.data.intensities.shape[2] == 204
    assert mixFWSDataset.D2OData.data.intensities.shape[2] == 204


def test_binAll_data_mixQENSandFWS_observable(mixFWSDataset):
    mixFWSDataset.binAll(2, axis="observable")
    assert mixFWSDataset.dataList[0].data.intensities.shape[0] == 10


def test_resetAll(mixFWSDataset):
    mixFWSDataset.binAll(5)
    mixFWSDataset.normalize_usingResFunc()
    mixFWSDataset.resetAll()
    assert mixFWSDataset.resData[0].data.intensities.shape[2] == 1024
    assert mixFWSDataset.ECData.data.intensities.shape[2] == 1024
    assert mixFWSDataset.D2OData.data.intensities.shape[2] == 1024
    assert mixFWSDataset.dataList[0].data.intensities[0, 0].sum() < 0.3


def test_resetAll(mixFWSDataset):
    mixFWSDataset.binAll(5)
    mixFWSDataset.normalize_usingResFunc()
    mixFWSDataset.resetDataset()
    assert mixFWSDataset.dataList[0].data.intensities[0, 0].sum() < 0.3


def test_resetAll(mixFWSDataset):
    mixFWSDataset.scaleData(2)
    assert mixFWSDataset.dataList[0].data.intensities[0, 0].sum() > 0.06


def test_resetDataset(fullQENSDataset):
    fullQENSDataset.binAll(5)
    fullQENSDataset.resetAll()
    assert fullQENSDataset.dataList[0].data.intensities.shape[2] == 1024


def test_normalize_QENS_fromQENSres(fullQENSDataset):
    fullQENSDataset.normalize_usingResFunc()
    assert fullQENSDataset.dataList[0].data.intensities.sum() > 120


def test_normalize_QENS_fromSelf(fullQENSDataset):
    fullQENSDataset.dataList[0].normalize_usingSelf()
    assert fullQENSDataset.dataList[0].data.intensities.sum() > 310


def test_normalize_FWS_fromQENSres(mixFWSDataset):
    mixFWSDataset.normalize_usingResFunc()
    assert mixFWSDataset.dataList[0].data.intensities.sum() > 11


def test_normalize_ENS_fromLowTemp(msdDataset):
    msdDataset.normalize_usingLowTemp()
    assert msdDataset.dataList[0].data.intensities.sum() < 35561


def test_subtractEC_fromQENS_withModel(fullQENSDataset):
    fullQENSDataset.subtract_EC()
    assert fullQENSDataset.dataList[0].data.intensities.sum() < 6.7


def test_subtractEC_fromQENS_withoutModel(fullQENSDataset):
    fullQENSDataset.subtract_EC(useModel=False)
    assert fullQENSDataset.dataList[0].data.intensities.sum() < 6.7


def test_subtractEC_fromFWS_withQENS(mixFWSDataset):
    mixFWSDataset.subtract_EC()
    assert mixFWSDataset.dataList[0].data.intensities.sum() < 2


def test_subtractEC_fromFWS_withFWS(fullFWSDataset):
    fullFWSDataset.subtract_EC()
    assert fullFWSDataset.dataList[0].data.intensities.sum() < 2


@pytest.mark.skipif(
    _HAS_ABSCO is False, reason="Requires compilation of absco with GSL"
)
def test_absorptionCorr_dataQENS_ecQENS(fullQENSDataset):
    fullQENSDataset.absorptionCorrection(D2O=False, res=False)
    assert fullQENSDataset.dataList[0].data.intensities.sum() < 8.33


@pytest.mark.skipif(
    _HAS_ABSCO is False, reason="Requires compilation of absco with GSL"
)
def test_absorptionCorr_dataFWS_ecQENS(mixFWSDataset):
    mixFWSDataset.absorptionCorrection(D2O=False, res=False)
    assert mixFWSDataset.dataList[0].data.intensities.sum() < 2.41


@pytest.mark.skipif(
    _HAS_ABSCO is False, reason="Requires compilation of absco with GSL"
)
def test_absorptionCorr_dataFWS_fecQENS():
    data = nPDyn.Dataset(
        FWSFiles=[path + "/sample_data/lys_part_01_FWS.nxs"],
        resFiles=[path + "/sample_data/vana_QENS_280K.nxs"],
        fECFile=path + "/sample_data/empty_cell_FWS_280K.nxs",
    )
    data.absorptionCorrection(D2O=False, res=False)
    assert data.dataList[0].data.intensities.sum() < 1.34


@pytest.mark.skipif(
    _HAS_ABSCO is False, reason="Requires compilation of absco with GSL"
)
def test_absorptionCorr_dataQENS_ecQENS_D2O_and_res(fullQENSDataset):
    fullQENSDataset.absorptionCorrection()
    assert fullQENSDataset.resData[0].data.intensities.sum() < 34
    assert fullQENSDataset.D2OData.data.intensities.sum() < 4.95


@pytest.mark.skipif(
    _HAS_ABSCO is False, reason="Requires compilation of absco with GSL"
)
def test_absorptionCorr_dataFWS_fecQENS_fD2O_and_res(fullFWSDataset):
    fullFWSDataset.absorptionCorrection()
    assert fullFWSDataset.resData[0].data.intensities.sum() > 23
    assert fullFWSDataset.fD2OData.data.intensities.sum() > 0.09710


def test_discardDetectors(fullFWSDataset):
    fullFWSDataset.discardDetectors(0, 1, 2, 3)
    assert fullFWSDataset.dataList[0].data.qIdx.size == 14


def test_setQRange(fullFWSDataset):
    fullFWSDataset.setQRange(0.4, 1.7)
    assert fullFWSDataset.dataList[0].data.qIdx.size == 12
