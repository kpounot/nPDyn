"""This module provides several built-in models for incoherent
neutron scattering data fitting.

These functions generate a `Model` class instance from 
the **lmfit** package [#]_. 


References
----------

.. [#] https://lmfit.github.io/lmfit-py/

"""

import operator

import numpy as np

from lmfit import Model, CompositeModel
from lmfit.models import update_param_vals, COMMON_GUESS_DOC

from nPDyn.models.lmfit_presets import (
    linear, delta, gaussian, lorentzian,
    jump_diff, rotations, voigt, pseudo_voigt,
    kww, two_diff_state, hline, build_2D_model)

from nPDyn.models.lmfit_convolutions import getGlobals

# -------------------------------------------------------
# Helper functions for the models
# -------------------------------------------------------
def guess_from_qens(pars, pGlobals, data, x, q, prefix=None):
    """Estimate starting values from 2D peak data and create Parameters.
    
    Notes
    -----
    The dataset should be of shape (number of q-values, energies),
    that is, the function should be called for each value of
    'observable'.

    """
    if prefix is None:
        prefix = ''

    # guess starting values
    for qId, qVal in enumerate(q):
        center = x[np.argmax(data[qId])]
        amplitude = np.max(data[qId]) * np.pi
        b = (np.mean(data[qId, :25]) + np.mean(data[qId, -25:])) / 2

        halfMax = (np.max(data[qId]) - np.min(data[qId])) / 2
        inter = list((data[qId] - halfMax)**2)
        id1 = np.argmin(inter)
        inter.pop(id1)
        id2 = np.argmin(inter)
        sigma = np.sqrt((x[id1] - x[id2])**2)
        sigma = sigma if sigma != 0. else 1.  # to avoid division by zero

        for name, par in zip(('amplitude', 'center', 'sigma', 'b'),
                             (amplitude, center, sigma, b)):
            if par in pGlobals:
                name = prefix + name
            else:
                name = '%s%s_%i' % (prefix, name, qId)

            if name in pars.keys():
                pars[name].set(value=par)

    return pars

def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    pars.update_constraints()
    return pars

# -------------------------------------------------------
# Built-in models
# -------------------------------------------------------
class ModelPVoigtBkgd(CompositeModel):
    def __init__(self, q, **kwargs):
        """A pseudo-voigt profile with a background term.

        Parameters
        ----------
        q : np.array or list
            Array of momentum transfer q-values to be used.
        kwargs : dict
            Additional keyword arguments to pass to :func:`build_2D_model`

        """
        prefix = ''
        if 'prefix' in kwargs.keys():
            prefix = kwargs.pop('prefix')

        left = pseudo_voigt(q, prefix='%spv_' % prefix, **kwargs)
        right = hline(q, prefix='%sbkgd_' % prefix, **kwargs)

        super().__init__(left, right, operator.add)

    def guess(self, data, x=None, q=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()

        # guess parameters
        for comp in self.components:
            pGlobals = getGlobals(comp.make_funcargs(params=pars))
            pars.update(guess_from_qens(
                pars, pGlobals, data, x, q, prefix=comp.prefix))

        return update_param_vals(pars, self.prefix, **kwargs)

    guess.__doc__ = COMMON_GUESS_DOC


class ModelGaussBkgd(CompositeModel):
    def __init__(self, q, **kwargs):
        """A Gaussian with a background term.

        Can be useful for empty can signal.

        Parameters
        ----------
        q : np.array or list
            Array of momentum transfer q-values to be used.
        kwargs : dict
            Additional keyword arguments to pass to :func:`build_2D_model`

        """
        prefix = ''
        if 'prefix' in kwargs.keys():
            prefix = kwargs.pop('prefix')

        left = gaussian(q, prefix='%sgauss_' % prefix, **kwargs)
        right = hline(q, prefix='%sbkgd_' % prefix, **kwargs)

        super().__init__(left, right, operator.add)

    def guess(self, data, x=None, q=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()

        # guess parameters
        for comp in self.components:
            pGlobals = getGlobals(comp.make_funcargs(params=pars))
            pars.update(guess_from_qens(
                pars, pGlobals, data, x, q, prefix=comp.prefix))

        return update_param_vals(pars, self.prefix, **kwargs)

    guess.__doc__ = COMMON_GUESS_DOC


class ModelDeltaLorentzians(CompositeModel):
    def __init__(self, q, nLor=2, **kwargs):
        """A Dirac delta with a given number of Lorentzians.

        Parameters
        ----------
        q : np.array or list
            Array of momentum transfer q-values to be used.
        nLor : int, optional
            Number of Lorentzians to be included in the model.
        kwargs : dict
            Additional keyword arguments to pass to :func:`build_2D_model`

        """
        prefix = ''
        if 'prefix' in kwargs.keys():
            prefix = kwargs.pop('prefix')

        left = delta(q, prefix='%sdelta_' % prefix, **kwargs)
    
        right = lorentzian(q, prefix='%sl0_' % prefix, **kwargs)
        for i in range(1, nLor):
            right = CompositeModel(
                right, 
                lorentzian(q, prefix='%sl%i_' % (prefix, i), **kwargs),
                operator.add)

        super().__init__(left, right, operator.add)

    def guess(self, data, x=None, q=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()

        # guess parameters
        for comp in self.components:
            pGlobals = getGlobals(comp.make_funcargs(params=pars))
            pars.update(guess_from_qens(
                pars, pGlobals, data, x, q, prefix=comp.prefix))

        return update_param_vals(pars, self.prefix, **kwargs)

    guess.__doc__ = COMMON_GUESS_DOC
