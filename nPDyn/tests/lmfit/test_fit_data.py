import pytest

import numpy as np

from nPDyn.lmfit.lmfit_builtins import ModelGaussBkgd
from nPDyn.lmfit.lmfit_presets import (
    pseudo_voigt,
    calibratedD2O,
    protein_liquid,
)


def test_fitQENS_lmfit(fullQENSDataset):
    qens, res, ec, bkgd = fullQENSDataset
    qens, res, ec, bkgd = (
        val.get_q_range(0.4, 1.8) for val in (qens, res, ec, bkgd)
    )
    q = ec.q[:, np.newaxis]
    res.fit(pseudo_voigt(q, prefix="vana_"), cleanData="omit")
    bkgd.fit(calibratedD2O(q, 280, prefix="bkgd_"), cleanData="omit")
    qens.fit(
        protein_liquid(q, qWise=False),
        cleanData="omit",
        res=res,
        bkgd=bkgd,
        volume_fraction_bkgd=0.95,
    )
    assert np.sum((qens - qens.fit_best(x=qens.energies)) ** 2) < 0.001
    assert qens.params[0]["sigma_g"].value[0] != 5
