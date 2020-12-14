"""Tests for convolutions"""

from itertools import product

import numpy as np
from numpy.testing import assert_array_almost_equal

import pytest

from nPDyn.lmfit.lmfit_presets import (
    lorentzian,
    gaussian,
    pseudo_voigt,
    hline,
    delta,
    jump_diff,
    rotations,
    two_diff_state,
)
from nPDyn.lmfit import ConvolvedModel
from lmfit import CompositeModel


def get_models(prefix=None):
    """Set up the models with the associated convolutions."""
    lorentz = lorentzian(1, prefix=prefix)
    gauss = gaussian(1, prefix=prefix)
    pvoigt = pseudo_voigt(1, prefix=prefix)
    delt = delta(1, prefix=prefix)
    jdiff = jump_diff(1, prefix=prefix)
    rot = rotations(1, prefix=prefix)
    twodiff = two_diff_state(1, prefix=prefix)

    return (gauss, lorentz, delt, pvoigt, jdiff, rot, twodiff)


@pytest.mark.parametrize(
    "modelPair", product(get_models("left_"), get_models("right_"))
)
def test_comp_analytic_numeric(modelPair):
    X = np.linspace(-10, 10, 1000)

    analytic = ConvolvedModel(modelPair[0], modelPair[1]).eval(x=X, q=[1])[0]
    numeric = np.convolve(
        modelPair[0].eval(x=X, q=[1])[0],
        modelPair[1].eval(x=X, q=[1])[0],
        mode="same",
    )

    # normalize both results
    assert_array_almost_equal(
        analytic / analytic.sum(), numeric / numeric.sum(), decimal=2
    )
