import os

import unittest

import pytest

import nPDyn

try:
    from nPDyn.lib.pyabsco import py_absco_slab, py_absco_tube

    _HAS_ABSCO = True
except ImportError:
    _HAS_ABSCO = False

path = os.path.dirname(os.path.abspath(__file__))


def test_bin_data_QENS_energies(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens = qens.bin(5)
    assert qens.shape[2] == 204


def test_binAll_data_mixQENSandFWS_observable(mixFWSDataset):
    fws, res, ec, bkgd = mixFWSDataset
    fws = fws.bin(2, 0)
    assert fws.shape[0] == 2


def test_normalize_QENS_fromQENSres(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens = qens.normalize(res)
    assert qens.sum() > 120


def test_normalize_QENS_fromSelf(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens = qens.normalize()
    assert qens.sum() > 310


def test_normalize_FWS_fromQENSres(mixFWSDataset):
    fws, res, ec, bkgd = mixFWSDataset
    fws = fws.normalize(res)
    assert fws.sum() > 11


def test_normalize_ENS_fromLowTemp(msdDataset):
    msdDataset /= msdDataset[:5].mean()
    assert msdDataset.sum() < 40321


def test_subtractEC_fromQENS(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens = qens - 0.95 * ec
    assert qens.sum() < 6.7


@pytest.mark.skipif(
    _HAS_ABSCO is False, reason="Requires compilation of absco with GSL"
)
def test_absorptionCorr_dataQENS_ecQENS(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens = qens.absorptionCorrection(ec)
    assert qens.sum() < 8.47


@pytest.mark.skipif(
    _HAS_ABSCO is False, reason="Requires compilation of absco with GSL"
)
def test_absorptionCorr_dataFWS_ecQENS(mixFWSDataset):
    fws, res, ec, bkgd = mixFWSDataset
    fws = fws.absorptionCorrection(ec)
    assert fws.sum() < 2.41


def test_setQRange(fullFWSDataset):
    fws, res, ec, bkgd = fullFWSDataset
    fws = fws.get_q_range(0.4, 1.7)
    assert fws.q.size == 12


def test_array_manipulation(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens = qens.transpose().take([5], 1).squeeze().sliding_average(5, 0)
    assert qens.shape == (1019,)
