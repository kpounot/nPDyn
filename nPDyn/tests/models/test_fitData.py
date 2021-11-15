import pytest

import numpy as np

from nPDyn.models.builtins import modelPVoigt, modelD2OBackground
from nPDyn.models.presets import calibratedD2O
from nPDyn.lmfit.lmfit_builtins import ModelGaussBkgd
from nPDyn.lmfit.lmfit_presets import (
    pseudo_voigt,
    calibratedD2O,
    protein_liquid,
)


def test_fitRes_PVoigt(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    q = res.q[:, np.newaxis]
    res.fit(model=modelPVoigt(q))
    assert np.sum((res - res.fit_best()) ** 2) < 0.004


def test_fitD2O_Builtin(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    q = bkgd.q[:, np.newaxis]
    bkgd.fit(modelD2OBackground(q))
    assert np.sum((bkgd - bkgd.fit_best()) ** 2) < 0.005


def test_fitQENS_lmfit(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens, res, ec, bkgd = (
        val.get_q_range(0.4, 1.8) for val in (qens, res, ec, bkgd)
    )
    q = ec.q[:, np.newaxis]
    bkgd.fit(modelD2OBackground(q), cleanData="omit")
    qens, res, bkgd = (
        val.absorptionCorrection(ec) for val in (qens, res, bkgd)
    )
    res.fit(pseudo_voigt(q, prefix="vana_"), cleanData="omit")
    bkgd.fit(calibratedD2O(q, 280, prefix="bkgd_"), cleanData="omit")
    qens.fit(
        protein_liquid(q, qWise=False, prefix="jumpDiff_"),
        cleanData="omit",
        res=res,
        bkgd=bkgd,
        volume_fraction_bkgd=0.95,
    )
    assert np.sum((qens - qens.fit_best(x=qens.energies)) ** 2) < 0.001
    assert qens.params[0]["sigma_g"].value != 5
