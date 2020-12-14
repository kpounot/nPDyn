"""This module provides several preset functions that can
be used to create model components and fit your data.

"""

import numpy as np

from scipy.fft import fft, fftshift
from scipy.special import wofz, spherical_jn

from nPDyn.models.d2O_calibration.interpD2O import getD2Odata


# -------------------------------------------------------
# Define some functions for the models
# -------------------------------------------------------
def gaussian(x, scale=1, width=1, center=0):
    """A normalized Gaussian function

    Parameters
    ----------
    x : np.ndarray
        x-axis values, can be an array of any shape
    scale : int, float,  np.ndarray
        scale factor for the normalized function
    width : int, np.ndarray
        width of the lineshape
    center : int, float,  np.ndarray
        center from the zero-centered lineshape

    """
    res = scale * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    res /= np.sqrt(2 * np.pi * width ** 2)
    return res


def lorentzian(x, scale=1, width=1, center=0):
    """A normalized Lorentzian function.

    Parameters
    ----------
    x : np.ndarray
        x-axis values, can be an array of any shape
    scale : int, float, np.ndarray
        scale factor for the normalized function
    width : int, np.ndarray
        width of the lineshape
    center : int, float, np.ndarray
        center from the zero-centered lineshape

    """
    res = scale * width / (np.pi * ((x - center) ** 2 + width ** 2))
    return res


def voigt(x, scale=1, sigma=1, gamma=1, center=0):
    """A normalized Voigt profile.

    Parameters
    ----------
    x : np.ndarray
        Values for the x-axis, can be an array of any shape
    scale : int, float, np.ndarray
        Scale factor for the normalized function
    sigma : int, float, np.ndarray
        Line width of the Gaussian component.
    gamma : int, float, np.ndarray
        Line width of the Lorentzian component.
    center : int, float, np.ndarray
        Center from the zero-centered lineshape

    """
    res = scale * wofz(((x - center) + 1j * gamma) / (sigma * np.sqrt(2))).real
    res /= sigma * np.sqrt(2 * np.pi)

    return res


def delta(x, scale=1, center=0):
    """A Dirac delta centered on *center*

    Parameters
    ----------
    x : np.ndarray
        x-axis values, can be an array of any shape
    scale : int, float,  np.ndarray
        scale factor for the normalized function
    center : int, float,  np.ndarray
        position of the Dirac Delta in energy

    """
    center = (x - center) ** 2
    center = np.argwhere(center == center.min())
    out = x * 0
    out[center] = scale / (x[1] - x[0])
    return out


def linear(x, a=0.0, b=1.0):
    """A linear model of the form :math:`a x + b`"""
    return a * x + b


def calibratedD2O(x, q, volFraction, temp, amplitude=1.0):
    """Lineshape for D2O where the Lorentzian width was obtained
    from a measurement on IN6 at the ILL.

    Parameters
    ----------
    q : np.array or list
        Array of momentum transfer q values
    volFraction : float in [0, 1]
        Volume fraction of the D2O in the sample.
    temp : float
        Sample temperature used for the experiment.
    amplitude : float
        Amplitude of the D2O signal. The parameter to be fitted.

    """
    out = (
        amplitude
        * getD2Odata(volFraction)(temp, q)
        / (np.pi * (x ** 2 + getD2Odata(volFraction)(temp, q) ** 2))
    )

    return out


# -------------------------------------------------------
# Define some analytic convolutions
# -------------------------------------------------------
def conv_lorentzian_lorentzian(x, comp1, comp2, params1, params2, **kwargs):
    """Convolution between two Lorentzians

    Parameters
    ----------
    x : np.ndarray
        x-axis values
    comp1 : :class:`Component`
        First component to be used for the convolution.
    comp2 : :class:`Component`
        Second component to be used for the convolution.
    params1 : :class:`Parameters`
        Parameters for `comp1`.
    params2 : :class:`Parameters`
        Parameters for `comp2`.
    kwargs : dict
        Additional keyword arguments to pass to the method
        :meth:`processFuncArgs` for `comp1` and `comp2`.

    """
    p1 = comp1.processFuncArgs(params1, **kwargs)
    p2 = comp2.processFuncArgs(params2, **kwargs)

    scale = p1["scale"] * p2["scale"]
    width = p1["width"] + p2["width"]
    center = p1["center"] + p2["center"]

    return lorentzian(x, scale, width, center)


def conv_gaussian_gaussian(x, comp1, comp2, params1, params2, **kwargs):
    """Convolution between two Gaussians

    Parameters
    ----------
    x : np.ndarray
        x-axis values
    comp1 : :class:`Component`
        First component to be used for the convolution.
    comp2 : :class:`Component`
        Second component to be used for the convolution.
    params1 : :class:`Parameters`
        Parameters for `comp1`.
    params2 : :class:`Parameters`
        Parameters for `comp2`.
    kwargs : dict
        Additional keyword arguments to pass to the method
        :meth:`processFuncArgs` for `comp1` and `comp2`.

    """
    p1 = comp1.processFuncArgs(params1, **kwargs)
    p2 = comp2.processFuncArgs(params2, **kwargs)

    scale = p1["scale"] * p2["scale"]
    width = np.sqrt(p1["width"] ** 2 + p2["width"] ** 2)
    center = p1["center"] + p2["center"]

    return gaussian(x, scale, width, center)


def conv_lorentzian_gaussian(x, comp1, comp2, params1, params2, **kwargs):
    """Convolution between a Lorentzian and a Gaussian

    Parameters
    ----------
    x : np.ndarray
        x-axis values
    comp1 : :class:`Component`
        First component to be used for the convolution.
    comp2 : :class:`Component`
        Second component to be used for the convolution.
    params1 : :class:`Parameters`
        Parameters for `comp1`.
    params2 : :class:`Parameters`
        Parameters for `comp2`.
    kwargs : dict
        Additional keyword arguments to pass to the method
        :meth:`processFuncArgs` for `comp1` and `comp2`.

    """
    if comp1.func.__name__ != "lorentzian":
        comp1, comp2, params1, params2 = _switchComps(
            comp1, comp2, params1, params2
        )

    p1 = comp1.processFuncArgs(params1, **kwargs)
    p2 = comp2.processFuncArgs(params2, **kwargs)

    scale = p1["scale"] * p2["scale"]
    gamma = p1["width"]
    sigma = p2["width"]
    center = p1["center"] + p2["center"]

    return voigt(x, scale, sigma, gamma, center)


def conv_delta(x, comp1, comp2, params1, params2, **kwargs):
    """Convolution between a Lorentzian and a Gaussian

    Parameters
    ----------
    x : np.ndarray
        x-axis values
    comp1 : :class:`Component`
        First component to be used for the convolution.
    comp2 : :class:`Component`
        Second component to be used for the convolution.
    params1 : :class:`Parameters`
        Parameters for `comp1`.
    params2 : :class:`Parameters`
        Parameters for `comp2`.
    kwargs : dict
        Additional keyword arguments to pass to the method
        :meth:`processFuncArgs` for `comp1` and `comp2`.

    """
    if comp1.func.__name__ != "delta":
        comp1, comp2, params1, params2 = _switchComps(
            comp1, comp2, params1, params2
        )

    p1 = comp1.processFuncArgs(params1, **kwargs)
    p2 = comp2.processFuncArgs(params2, **kwargs)

    p2["center"] += p1["center"]

    return comp2.func(x, **p2)


def conv_linear(x, comp1, comp2, params1, params2, **kwargs):
    """Convolution with a linear model.

    The linear model is assumed to be used for a background and
    is thus not convolved. The function returns simply the linear
    model.
    If `comp2` is also a linear model, the two models are simply added.

    Parameters
    ----------
    x : np.ndarray
        x-axis values
    comp1 : :class:`Component`
        First component to be used for the convolution.
    comp2 : :class:`Component`
        Second component to be used for the convolution.
    params1 : :class:`Parameters`
        Parameters for `comp1`.
    params2 : :class:`Parameters`
        Parameters for `comp2`.
    kwargs : dict
        Additional keyword arguments to pass to the method
        :meth:`processFuncArgs` for `comp1` and `comp2`.

    """
    if comp1.func.__name__ != "linear":
        comp1, comp2, params1, params2 = _switchComps(
            comp1, comp2, params1, params2
        )

    p1 = comp1.processFuncArgs(params1, **kwargs)
    p2 = comp2.processFuncArgs(params2, **kwargs)

    if comp2.func.__name__ == "linear":
        return comp1.func(x, **p1) + comp2.func(x, **p2)
    else:
        return comp1.func(x, **p1)


def _switchComps(comp1, comp2, p1, p2):
    """Switch components and parameters."""
    tmpComp = comp1
    tmpParam = p1
    comp1 = comp2
    comp2 = tmpComp
    p1 = p2
    p2 = tmpParam
    return comp1, comp2, p1, p2
