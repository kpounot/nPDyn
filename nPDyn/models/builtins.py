"""This module provides several built-in models for incoherent
neutron scattering data fitting.

These functions generate a :class:`Model` class instance.

"""

import operator

import numpy as np

from nPDyn.models.presets import (
    linear,
    delta,
    gaussian,
    lorentzian,
    calibratedD2O,
)
from nPDyn.models.params import Parameters
from nPDyn.models.model import Model, Component


# -------------------------------------------------------
# Built-in models
# -------------------------------------------------------
def modelPVoigt(q, name="PVoigt", **kwargs):
    """A model containing a pseudo-Voigt profile.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    """
    p = Parameters(
        scale={"value": np.zeros_like(q) + 1.0, "bounds": (0.0, np.inf)},
        frac={"value": np.zeros_like(q) + 0.5, "bounds": (0.0, 1.0)},
        width={"value": np.zeros_like(q) + 1.0, "bounds": (0.0, np.inf)},
        center={"value": np.zeros_like(q) + 0.0, "bounds": (-np.inf, np.inf)},
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component("lorentzian", lorentzian, scale="scale * (1 - frac)")
    )
    m.addComponent(
        Component(
            "gaussian",
            gaussian,
            scale="scale * frac",
            width="width / np.sqrt(2 * np.log(2))",
        )
    )

    return m


def modelPVoigtBkgd(q, name="PVoigtBkgd", **kwargs):
    """A model containing a pseudo-Voigt profile with a background term.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    """
    p = Parameters(
        scale={"value": np.zeros_like(q) + 1.0, "bounds": (0.0, np.inf)},
        frac={"value": np.zeros_like(q) + 0.5, "bounds": (0.0, 1.0)},
        width={"value": np.zeros_like(q) + 1.0, "bounds": (0.0, np.inf)},
        center={"value": np.zeros_like(q) + 0.0, "bounds": (-np.inf, np.inf)},
        bkgd={"value": np.zeros_like(q) + 0.001, "bounds": (0.0, np.inf)},
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component("lorentzian", lorentzian, scale="scale * (1 - frac)")
    )
    m.addComponent(
        Component(
            "gaussian",
            gaussian,
            scale="scale * frac",
            width="width / np.sqrt(2 * np.log(2))",
        )
    )
    m.addComponent(Component("background", linear, True, a=0.0, b="bkgd"))

    return m


def modelGaussBkgd(q, name="GaussBkgd", **kwargs):
    """A model containing a Gaussian with a background term.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    """
    p = Parameters(
        scale={"value": np.zeros_like(q) + 1.0, "bounds": (0.0, np.inf)},
        width={"value": np.zeros_like(q) + 1.0, "bounds": (0.0, np.inf)},
        center={"value": np.zeros_like(q) + 0.0, "bounds": (-np.inf, np.inf)},
        bkgd={"value": np.zeros_like(q) + 0.001, "bounds": (0.0, np.inf)},
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(Component("gaussian", gaussian))
    m.addComponent(Component("background", linear, True, a=0.0, b="bkgd"))

    return m


def modelLorentzianSum(q, name="LorentzianSum", nLor=2, **kwargs):
    """A model containing a delta and a sum of Lorentzians
    with a background term.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    nLor : 2
        Number of Lorentzian to be used.
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    """
    p = Parameters(
        a0={"value": 0.5, "bounds": (0.0, 1.0)},
        center={"value": 0.0, "fixed": True},
        msd={"value": 1.0, "bounds": (0.0, np.inf)},
        bkgd={"value": np.zeros_like(q) + 0.001, "bounds": (0.0, np.inf)},
    )

    for idx in range(nLor):
        p.set("a%i" % (idx + 1), value=0.5, bounds=(0.0, 1.0))
        p.set("w%i" % (idx + 1), value=10, bounds=(0.0, np.inf))

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(Component("EISF", delta, scale="np.exp(-q**2 * msd) * a0"))
    for idx in range(nLor):
        m.addComponent(
            Component(
                r"$\mathcal{L}_{%i}$" % (idx + 1),
                lorentzian,
                scale="np.exp(-q**2 - msd) * a%i" % (idx + 1),
                width="w%i * q**2" % (idx + 1),
            )
        )
    m.addComponent(Component("background", linear, True, a=0.0, b="bkgd"))

    return m


def modelD2OBackground(
    q, volFraction=0.95, temperature=300, name="$D_2O$", **kwargs
):
    """A model for D2O background with calibrated linewidth.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    volFraction : float
        Volume fraction of the D2O in the sample.
    temperature : float
        Temperature of the sample
    name : str
        Name for the model
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    """
    p = Parameters(
        amplitude={"value": np.zeros_like(q) + 1, "bounds": (0.0, np.inf)}
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component(
            "$D_2O$ background",
            calibratedD2O,
            q=q.flatten(),
            volFraction=volFraction,
            temp=temperature,
            skip_convolve=True,
        )
    )

    return m
