""" This module provides several preset models that can
    be used to fit your data.

    All convolutions are already defined with the models
    that start with 'res'.

"""

import numpy as np

from scipy.special import wofz

from nPDyn.models.Model import Model

# -------------------------------------------------------
# Defines the functions for the models
# -------------------------------------------------------
def gaussian(x, scale=1, width=1, shift=0):
    """ A normalized Gaussian function

        Parameters
        ----------
        x : np.ndarray
            x-axis values, can be an array of any shape
        scale : int, np.ndarray
            scale factor for the normalized function
        width : int, np.ndarray
            width of the lineshape
        shift : int, np.ndarray
            shift from the zero-centered lineshape

    """

    res = scale * np.exp(-(x - shift)**2 / (2 * width**2)) 
    res /= np.sqrt(2 * np.pi * width**2)

    return res


def lorentzian(x, scale=1, width=1, shift=0):
    """ A normalized Lorentzian function.

        Parameters
        ----------
        x : np.ndarray
            x-axis values, can be an array of any shape
        scale : int, np.ndarray
            scale factor for the normalized function
        width : int, np.ndarray
            width of the lineshape
        shift : int, np.ndarray
            shift from the zero-centered lineshape

    """

    res = scale * width / (np.pi * ((x - shift)**2 + width**2))

    return res


def delta(x, scale=1, pos=0, energy_axis=0):
    """ A Dirac delta centered on *pos*

        Parameters
        ----------
        x : np.ndarray
            x-axis values, can be an array of any shape
        scale : int, np.ndarray
            scale factor for the normalized function
        pos : int, np.ndarray
            position of the Dirac Delta in energy
        energy_axis : int
            in case :attr:`x` is a multidimensional array,
            use this value to define which axis corresponds
            to the energies.

    """

    pos = (x - pos)**2

    pos = np.argwhere(pos == pos.min())[:,energy_axis]

    x *= 0
    x[pos] = scale

    return x




# -------------------------------------------------------
# Defines the analytic convolutions
# -------------------------------------------------------
def conv_lorentzian_lorentzian(x, comp1, comp2, p1, p2):
    """ Convolution between two Lorentzians

        Parameters
        ----------
        x : np.ndarray
            x-axis values
        comp1 : tuple (function, paramProcessingFunc, [convolutions])
            first component for convolution (the type will be
            determined by the function instance)
        comp2 : tuple (function, paramProcessingFunc, [convolutions])
            second component for convolution (the type will be
            determined by the function instance)

    """

    p1 = comp1[1](p1)
    p2 = comp2[1](p2)

    conv_scale = p1[0] * p2[0]
    conv_width = p1[1] + p2[1]
    conv_shift = p1[2] + p2[2]
    
    return lorentzian(x, conv_scale, conv_width, conv_shift)


def conv_lorentzian_gaussian(x, comp1, comp2, p1, p2):
    """ Convolution between a Lorentzian and a Gaussian.

        Parameters
        ----------
        x : np.ndarray
            x-axis values
        comp1 : tuple (function, paramProcessingFunc, [convolutions])
            first component for convolution (the type will be
            determined by the function instance)
        comp2 : tuple (function, paramProcessingFunc, [convolutions])
            second component for convolution (the type will be
            determined by the function instance)

    """

    if comp1[0].__name__ == 'lorentzian':
        p1 = comp1[1](p1)
        p2 = comp2[1](p2)
    else:
        p1 = comp2[1](p2)
        p2 = comp1[1](p1)

    shift = p1[2] + p2[2]

    res = p1[0] * p2[0]
    res *= wofz(((x - shift) + 1j * p1[1]) / (p2[1] * np.sqrt(2))).real
    res /= p2[1] * np.sqrt(2 * np.pi) 

    return res


def conv_delta(x, comp1, comp2, p1, p2):
    """ Convolution between a Lorentzian and a Dirac delta.

        Parameters
        ----------
        x : np.ndarray
            x-axis values
        comp1 : tuple (function, paramProcessingFunc, [convolutions])
            first component for convolution (the type will be
            determined by the function instance)
        comp2 : tuple (function, paramProcessingFunc, [convolutions])
            second component for convolution (the type will be
            determined by the function instance)

    """

    if comp1[0].__name__ == 'lorentzian' or comp1[0].__name__ == 'gaussian':
        f = comp1[0]
        pr1 = comp1[1](p1)
        pr2 = comp2[1](p2)
    else:
        f = comp2[0]
        pr1 = comp2[1](p2)
        pr2 = comp1[1](p1)

    scale = pr1[0] * pr2[0]
    shift = pr2[1]

    return f(x, scale, pr1[1], shift)
