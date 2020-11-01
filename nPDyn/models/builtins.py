"""This module provides several built-in models for incoherent
neutron scattering data fitting.

These functions generate a :class:`Model` class instance. 

"""

import operator

import numpy as np

from nPDyn.models.presets import (linear, delta, gaussian, lorentzian)
from nPDyn.models.params import Parameters
from nPDyn.models.model import Model, Component


# -------------------------------------------------------
# Built-in models
# -------------------------------------------------------
def modelPVoigtBkgd(q, name='PVoigtBkgd', **kwargs):
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
        scale={'value': np.zeros_like(q) + 1., 'bounds': (0., np.inf)},
        frac={'value': np.zeros_like(q) + 0.5, 'bounds': (0., 1.)},
        width={'value': np.zeros_like(q) + 1., 'bounds': (0., np.inf)},
        center={'value': np.zeros_like(q) + 0., 'bounds': (-np.inf, np.inf)},
        bkgd={'value': np.zeros_like(q) + 0.001, 'bounds': (0., np.inf)})

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(Component(
        "lorentzian", lorentzian, scale='scale * (1 - frac)'))
    m.addComponent(Component(
        "gaussian", gaussian, scale='scale * frac', 
        width='width / np.sqrt(2 * np.log(2))'))
    m.addComponent(Component(
        "background", linear, a=0., b='bkgd'))

    return m

def modelGaussBkgd(q, name='GaussBkgd', **kwargs):
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
        scale={'value': np.zeros_like(q) + 1., 'bounds': (0., np.inf)},
        width={'value': np.zeros_like(q) + 1., 'bounds': (0., np.inf)},
        center={'value': np.zeros_like(q) + 0., 'bounds': (-np.inf, np.inf)},
        bkgd={'value': np.zeros_like(q) + 0.001, 'bounds': (0., np.inf)})

    p.update(**kwargs)

    m = Model(p, name)

    m.addComponent(Component("gaussian", gaussian))
    m.addComponent(Component("background", linear, a=0., b='bkgd'))

    return m
