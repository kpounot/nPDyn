from itertools import product

import numpy as np
from numpy.testing import assert_array_almost_equal

import pytest

from nPDyn.models import Model, Parameters, Component
from nPDyn.models.presets import lorentzian, gaussian, linear, delta
from nPDyn.models.builtins import (
    modelPVoigt,
    modelLorentzianSum,
    modelPVoigtBkgd,
    modelGaussBkgd,
)


def get_models():
    """Set up the models with the associated convolutions."""
    pLor = Parameters(scale=1, width=2, center=0.0)
    lor = Model(pLor)
    lor.addComponent(Component("lorentzian", lorentzian))

    pGau = Parameters(scale=1, width=4, center=0.5)
    gau = Model(pGau)
    gau.addComponent(Component("gaussian", gaussian))

    pLin = Parameters(a=1, b=2)
    lin = Model(pLin)
    lin.addComponent(Component("linear", linear))

    pDel = Parameters(scale=2, center=2.3)
    delt = Model(pDel)
    delt.addComponent(Component("delta", delta))

    return lor, gau, lin, delt


@pytest.mark.parametrize("modelPair", product(get_models(), get_models()))
def test_comp_analytic_numeric(modelPair):
    X = np.linspace(-10, 10, 1000)

    numeric = np.convolve(
        modelPair[0].eval(x=X), modelPair[1].eval(x=X), mode="same"
    )

    analytic = modelPair[0].eval(x=X, convolve=modelPair[1])

    # normalize both results
    assert_array_almost_equal(
        analytic / analytic.sum(), numeric / numeric.sum(), decimal=2
    )
