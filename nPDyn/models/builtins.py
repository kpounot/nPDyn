"""This module provides several built-in models for incoherent
neutron scattering data fitting.

These functions generate a :class:`Model` class instance.

"""
import numpy as np

from nPDyn.models.presets import (
    linear,
    delta,
    gaussian,
    lorentzian,
    generalizedLorentzian,
    rotations,
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


def modelLorentzianSum(q, name="LorentzianSum", nLor=2, qWise=True, **kwargs):
    """A model containing a delta and a sum of Lorentzians.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    nLor : 2
        Number of Lorentzian to be used.
    qWise : bool
        If True, no q dependence is imposed on the parameters and
        the each spectrum is fitted independently.
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    """
    p = Parameters(
        scale={"value": np.zeros_like(q) + 1, "bounds": (0.0, np.inf)},
        a0={"value": np.zeros_like(q) + 0.5, "bounds": (0.0, 1)},
        center={"value": 0.0, "fixed": True},
    )
    aDefault = np.zeros_like(q) + 0.5
    if qWise:
        widthDefault = np.zeros_like(q) + 10
        widthStr = "w%i"
    else:
        widthDefault = 10
        widthStr = "w%i * q ** 2"

    for idx in range(nLor):
        p.set("a%i" % (idx + 1), value=aDefault, bounds=(0.0, 1.0))
        p.set("w%i" % (idx + 1), value=widthDefault, bounds=(0.0, np.inf))

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(Component("EISF", delta, scale="a0 * scale"))
    for idx in range(nLor):
        m.addComponent(
            Component(
                r"$\mathcal{L}_{%i}$" % (idx + 1),
                lorentzian,
                scale="a%i * scale" % (idx + 1),
                width=widthStr % (idx + 1),
            )
        )

    return m


def modelGeneralizedLorentzian(
    q, name="GeneralizedLorentzian", qWise=True, **kwargs
):
    """A model containing a delta and a generalized lorentzian.

    This model has been described elsewhere [#]_.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.


    References
    ----------
    .. [#] https://doi.org/10.1063/1.5121703

    """
    if qWise:
        alpha = {"value": np.zeros_like(q) + 0.5, "bounds": (0.0, 1)}
        tau = {"value": np.zeros_like(q) + 0.1, "bounds": (0.0, np.inf)}
        tauExpr = "tau"
    else:
        alpha = {"value": 0.5, "bounds": (0.0, 1)}
        tau = {"value": 0.1, "bounds": (0.0, np.inf)}
        tauExpr = "(tau * q ** 2)**(-1 / alpha)"

    p = Parameters(
        scale={"value": np.zeros_like(q) + 1, "bounds": (0.0, np.inf)},
        a={"value": np.zeros_like(q) + 0.1, "bounds": (0.0, 1)},
        center={"value": 0.0, "fixed": True},
        alpha=alpha,
        tau=tau,
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(Component("EISF", delta, scale="scale * a"))
    m.addComponent(
        Component(
            "$\\mathcal{L}_{\\alpha, \\tau}$",
            generalizedLorentzian,
            scale="scale * (1 - a)",
            tau=tauExpr,
        )
    )

    return m


def modelWater(q, name="waterDynamics", **kwargs):
    """A model containing a delta, a Lorentzian for translational
    motions, a Lorentzian for rotational motions, and a
    background term.

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
        a0={"value": 0.33, "bounds": (0.0, np.inf)},
        at={"value": 0.33, "bounds": (0.0, np.inf)},
        ar={"value": 0.33, "bounds": (0.0, np.inf)},
        wt={"value": 5, "bounds": (0.0, np.inf)},
        wr={"value": 15, "bounds": (0.0, np.inf)},
        center={"value": 0.0, "fixed": True},
        msd={"value": 1.0, "bounds": (0.0, np.inf)},
        bkgd={"value": np.zeros_like(q) + 0.001, "bounds": (0.0, np.inf)},
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component(
            "EISF",
            delta,
            scale="np.exp(-q**2 * msd) * "
            "(a0 + ar * spherical_jn(0, 0.96 * q)**2)",
        )
    )
    m.addComponent(
        Component(
            r"$\mathcal{L}_r$",
            rotations,
            scale="np.exp(-q**2 * msd) * ar",
            width="wr",
        )
    )
    m.addComponent(
        Component(
            r"$\mathcal{L}_t$",
            lorentzian,
            scale="np.exp(-q**2 * msd) * at",
            width="wt * q**2",
        )
    )
    m.addComponent(Component("background", linear, True, a=0.0, b="bkgd"))

    return m


def modelProteinJumpDiff(q, name="proteinJumpDiff", qWise=False, **kwargs):
    """A model for protein in liquid state.

    The model contains a Lorentzian of Fickian-type
    diffusion accounting for
    center-of-mass motions, a Lorentzian of width that
    obeys the jump diffusion model [#]_ accounting for
    internal dynamics.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    qWise : bool
        If True, no q dependence is imposed on the parameters and
        the each spectrum is fitted independently.
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    References
    ----------
    .. [#] https://doi.org/10.1103/PhysRev.119.863

    """
    if qWise:
        p = Parameters(
            beta={"value": np.zeros_like(q) + 1, "bounds": (0.0, np.inf)},
            a0={"value": np.zeros_like(q) + 0.5, "bounds": (0.0, 1)},
            wg={"value": np.zeros_like(q) + 5, "bounds": (0.0, np.inf)},
            wi={"value": np.zeros_like(q) + 30, "bounds": (0.0, np.inf)},
            center={"value": 0.0, "fixed": True},
        )
        widthG = "wg"
        widthI = "wg + wi"
    else:
        p = Parameters(
            beta={"value": np.zeros_like(q) + 1, "bounds": (0.0, np.inf)},
            a0={"value": np.zeros_like(q) + 0.5, "bounds": (0.0, 1)},
            wg={"value": 2, "bounds": (0.0, np.inf)},
            wi={"value": 30, "bounds": (0.0, np.inf)},
            tau={"value": 1e-2, "bounds": (0.0, np.inf)},
            center={"value": 0.0, "fixed": True},
        )
        widthG = "wg * q**2"
        widthI = "wg * q**2 + wi * q**2 / (1 + wi * q**2 * tau)"

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component(
            r"$\mathcal{L}_g$", lorentzian, scale="beta * a0", width=widthG
        )
    )
    m.addComponent(
        Component(
            r"$\mathcal{L}_i$",
            lorentzian,
            scale="beta * (1 - a0)",
            width=widthI,
        )
    )

    return m


def modelTwoStatesSwitchDiff(q, name="TwoStatesSwitch", **kwargs):
    """A model for protein in liquid state.

    This model implements the two states switching diffusion model for
    nPDyn [#]_.

    Parameters
    ----------
    q : np.ndarray
        Array of values for momentum transfer q.
    name : str
        Name for the model
    kwargs : dict
        Additional arguments to pass to Parameters.
        Can override default parameter attributes.

    References
    ----------
    .. [#] https://doi.org/10.1103/PhysRev.119.863

    """
    p = Parameters(
        beta={"value": np.zeros_like(q) + 1, "bounds": (0.0, np.inf)},
        a0={"value": np.zeros_like(q) + 0.5, "bounds": (0.0, 1)},
        tau1={"value": 1, "bounds": (0.0, np.inf)},
        tau2={"value": 1, "bounds": (0.0, np.inf)},
        g0={"value": 5, "bounds": (0.0, np.inf)},
        g1={"value": 10, "bounds": (0.0, np.inf)},
        g2={"value": 100, "bounds": (0.0, np.inf)},
        center={"value": 0, "fixed": True},
    )
    w0 = "g0 * q**2"
    w1 = (
        "(g1 * q**2 + g2 * q**2 + tau1 + tau2 + "
        "np.sqrt(((g1 - g2) * q**2 + tau1 - tau2)**2 + 4 * "
        "tau1 * tau2)) / 2"
    )
    w2 = (
        "(g1 * q**2 + g2 * q**2 + tau1 + tau2 - "
        "np.sqrt(((g1 - g2) * q**2 + tau1 - tau2)**2 + 4 * "
        "tau1 * tau2)) / 2"
    )
    s1 = (
        "beta * (1 - a0) * (tau1 / (tau1 + tau2) * g2 * q**2 + "
        "tau2 / (tau1 + tau2) * g1 * q**2 + "
        "tau1 + tau2 - %s) / (%s - %s)" % (w1, w2, w1)
    )
    s2 = (
        "beta * (1 - a0) * (tau1 / (tau1 + tau2) * g2 * q**2 + "
        "tau2 / (tau1 + tau2) * g1 * q**2 + "
        "tau1 + tau2 - %s) / (%s - %s)" % (w2, w1, w2)
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component(r"$\mathcal{L}_g$", lorentzian, scale="beta * a0", width=w0)
    )
    m.addComponent(
        Component(
            r"$\mathcal{L}_1$", lorentzian, scale=s1, width=w0 + " + " + w1
        )
    )
    m.addComponent(
        Component(
            r"$\mathcal{L}_2$", lorentzian, scale=s2, width=w0 + " + " + w2
        )
    )

    return m


def modelD2OBackground(q, name="$D_2O$", **kwargs):
    """A model for D2O background containing a single Lorentzian.

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
        scale={"value": np.ones_like(q), "bounds": (0.0, np.inf)},
        width={"value": np.ones_like(q), "bounds": (0.0, np.inf)},
        center={"value": 0.0},
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component("$D_2O$ background", lorentzian, skip_convolve=True)
    )

    return m


def modelCalibratedD2O(q, name="$D_2O$", volFraction=1, temp=300, **kwargs):
    """A model for D2O background containing a single Lorentzian.

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
        amplitude={"value": np.ones_like(q), "bounds": (0.0, np.inf)},
    )

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(
        Component(
            "$D_2O$ background",
            calibratedD2O,
            volFraction=volFraction,
            temp=temp,
            skip_convolve=True,
        )
    )

    return m
