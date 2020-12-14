from itertools import product

import numpy as np
from numpy.testing import assert_array_almost_equal

import pytest

from nPDyn.models import Model, Parameters, Component
from nPDyn.models.presets import lorentzian, linear


def test_model_component_operators():
    X = np.linspace(-10, 10, 1000)

    params = Parameters(scale=1.0, width=5, center=0, a=2, b=0.5)

    model1 = Model(params) + Component("lor1", lorentzian)
    model2 = Model(params) + Component("lor2", lorentzian)
    model = model1 + model2
    model = model - Component("lor3", lorentzian, scale=0.5)
    model = model / Component("lin1", linear, a=0)
    model = model * Component("lin2", linear)

    lor1 = lorentzian(X, width=5)
    lor2 = lorentzian(X, width=5)
    lor3 = lorentzian(X, width=5, scale=0.5)
    lin1 = linear(X, a=0, b=0.5)
    lin2 = linear(X, a=2, b=0.5)

    expect = (lor1 + lor2 - lor3) / lin1 * lin2

    # normalize both results
    assert_array_almost_equal(model.eval(X), expect, decimal=4)
