"""Can be used to perform analytic convolutions between models."""

from copy import deepcopy

from collections import OrderedDict

import numpy as np

from scipy.signal import fftconvolve

try:
    from lmfit import Model, CompositeModel
except ImportError:
    print(
        "The lmfit package cannot be found, please install it to use "
        "the interface with nPDyn."
    )

    class Model:
        def __init__(self, tmp, **kwargs):
            pass

    class CompositeModel:
        def __init__(self, tmp, **kwargs):
            pass


from nPDyn.lmfit.convolutions import (
    conv_lorentzian_lorentzian,
    conv_lorentzian_pvoigt,
    conv_gaussian_pvoigt,
    conv_gaussian_gaussian,
    conv_gaussian_lorentzian,
    conv_gaussian_jumpdiff,
    conv_gaussian_rotations,
    conv_jumpdiff_pvoigt,
    conv_rotations_pvoigt,
    conv_delta,
    conv_linear,
)


class ConvolvedModel(Model):
    """Combine two models (`left` and `right`) with the provided analytic
    convolution function(s).

    Parameters
    ----------
    left : :class:`Model` or :class:`CompositeModel`
        Left-hand model.
    right : :class:`Model` or :class:`CompositeModel`
        Right-hand model.
    on_undefined_conv: {'numeric', 'raise'}, optional
        Determine the behavior when a pair of model has no
        analytic convolution associated with it:
            - 'numeric' results in a numerical convolution
            - 'raise' raises a KeyError
        (default 'numeric')
    convMap : mapping, optional
        Dictionary of dictionaries to map the convolution function
        to a pair of model. A default convMap is already present in
        the class but can be overridden by this argument.
    **kws : optional
        Additional keywords are passed to `Model` when creating this
        new model.

    Notes
    -----
    The two models must use the same independent variables.
    Only the parameters from left and right are used and exposed.
    The parameters of the convolution function are not exposed outside
    the class. They are only used internally and determined inside the
    convolution function by the combination of the parameters and keywords
    provided for left and right.

    The :meth:`eval_components` returns the convoluted components from
    `left` by default. This behavior can be changed by using
    `returnComponents="right"` in the keyword arguments passed to
    the method.

    Examples
    --------
    First create two models to be convolved (here two Lorentzians):

    >>> l1 = lmfit.Model.LorentzianModel()
    >>> l2 = lmfit.Model.LorentzianModel()

    Define the convolution function using:

    >>> def myConv(left, right, params, **kwargs):
    ...     lp = left.make_funcargs(params, **kwargs)
    ...     rp = right.make_funcargs(params, **kwargs)
    ...     amplitude = lp['amplitude'] * rp['amplitude']
    ...     sigma = lp['sigma'] + rp['sigma']
    ...     center = lp['center'] + rp['center']
    ...     out = sigma / (np.pi * ((lp['x'] - center)**2 + sigma**2))
    ...     return out

    Eventually perform the convolution:

    >>> convModel = ConvolvedModel(l1, l2)

    Assign the convolution function `myConv` to the pair of 'lorentzian'
    using:

    >>> convModel.convMap = {'lorentzian': {'lorentzian': myConv}}

    """

    _names_collide = (
        "\nTwo models have parameters named '{clash}'. " "Use distinct names."
    )
    _bad_arg = "ConvolvedModel: argument {arg} is not a Model"

    def __init__(
        self, left, right, on_undefined_conv="numeric", convMap=None, **kws
    ):
        if not isinstance(left, Model):
            raise ValueError(self._bad_arg.format(arg=left))
        if not isinstance(right, Model):
            raise ValueError(self._bad_arg.format(arg=right))
        if not np.isin(on_undefined_conv, ["numeric", "raise"]):
            raise ValueError(
                "Parameter 'on_undefined_conv' should be either "
                "'numeric' or 'raise'."
            )

        self.left = left
        self.right = right

        self._on_undefined_conv = on_undefined_conv

        name_collisions = set(left.param_names) & set(right.param_names)
        if len(name_collisions) > 0:
            msg = ""
            for collision in name_collisions:
                msg += self._names_collide.format(clash=collision)
            raise NameError(msg)

        # we assume that all the sub-models have the same independent vars
        if "independent_vars" not in kws:
            kws["independent_vars"] = self.left.independent_vars
        if "nan_policy" not in kws:
            kws["nan_policy"] = self.left.nan_policy

        def _tmp(self, *args, **kws):
            pass

        Model.__init__(self, _tmp, **kws)

        for side in (left, right):
            prefix = side.prefix
            for basename, hint in side.param_hints.items():
                self.param_hints["%s%s" % (prefix, basename)] = hint

        # set a default convolution map
        self.convMap = {
            "lorentzian": {
                "lorentzian": conv_lorentzian_lorentzian,
                "gaussian": conv_gaussian_lorentzian,
                "pvoigt": conv_lorentzian_pvoigt,
                "delta": conv_delta,
                "linear": conv_linear,
            },
            "gaussian": {
                "lorentzian": conv_gaussian_lorentzian,
                "gaussian": conv_gaussian_gaussian,
                "delta": conv_delta,
                "linear": conv_linear,
                "pvoigt": conv_gaussian_pvoigt,
            },
            "pvoigt": {
                "lorentzian": conv_lorentzian_pvoigt,
                "gaussian": conv_gaussian_pvoigt,
                "jump_diff": conv_jumpdiff_pvoigt,
                "linear": conv_linear,
                "delta": conv_delta,
                "rotations": conv_rotations_pvoigt,
            },
            "jump_diff": {
                "gaussian": conv_gaussian_jumpdiff,
                "delta": conv_delta,
                "linear": conv_linear,
                "pvoigt": conv_jumpdiff_pvoigt,
            },
            "rotations": {
                "gaussian": conv_gaussian_rotations,
                "delta": conv_delta,
                "linear": conv_linear,
                "pvoigt": conv_rotations_pvoigt,
            },
            "delta": {
                "gaussian": conv_delta,
                "lorentzian": conv_delta,
                "voigt": conv_delta,
                "pvoigt": conv_delta,
                "jump_diff": conv_delta,
                "rotations": conv_delta,
                "linear": conv_linear,
                "two_diff_state": conv_delta,
            },
            "linear": {
                "gaussian": conv_linear,
                "lorentzian": conv_linear,
                "voigt": conv_linear,
                "pvoigt": conv_linear,
                "jump_diff": conv_linear,
                "rotations": conv_linear,
                "two_diff_state": conv_linear,
            },
        }

        # override default convolutions with provided ones
        if convMap is not None:
            self.convMap.update(convMap)

    def _parse_params(self):
        self._func_haskeywords = (
            self.left._func_haskeywords or self.right._func_haskeywords
        )
        self._func_allargs = self.left._func_allargs + self.right._func_allargs
        self.def_vals = {
            self.right.prefix + key: val
            for key, val in self.right.def_vals.items()
        }
        self.def_vals.update(
            {
                self.left.prefix + key: val
                for key, val in self.left.def_vals.items()
            }
        )
        self.opts = deepcopy(self.right.opts)
        self.opts.update(self.left.opts)

    def _reprstring(self, long=False):
        return "(%s %s)" % (
            self.left._reprstring(long=long),
            self.right._reprstring(long=long),
        )

    def eval(self, params=None, **kwargs):
        """Evaluate model function for convolved model."""
        return self._convolve(self.left, self.right, params, **kwargs)

    def eval_components(self, **kwargs):
        """Return OrderedDict of name, results for each component."""
        returnComponents = "left"
        if "returnComponents" in kwargs.keys():
            returnComponents = kwargs["returnComponents"]

        return self._convolve(
            self.left, self.right, returnComponents=returnComponents, **kwargs
        )

    @property
    def param_names(self):
        """Return parameter names for composite model."""
        return self.left.param_names + self.right.param_names

    @property
    def components(self):
        """Return components for composite model."""
        return self.left.components + self.right.components

    @property
    def on_undefined_conv(self):
        """Return the parameter 'on_undefined_conv'"""
        return self._on_undefined_conv

    @on_undefined_conv.setter
    def on_undefined_conv(self, value):
        """Setter for 'on_undefined_conv parameter'."""
        if not np.isin(value, ["numeric", "raise", "skip"]):
            raise ValueError(
                "Parameter 'on_undefined_conv' should be either "
                "of 'numeric', 'raise' or 'skip'."
            )
        self._on_undefined_conv = value

    def _operators(self):
        """Return the operators for composite model."""
        left = self.left
        right = self.right
        lops = []
        rops = []
        while hasattr(left, "op") or hasattr(right, "op"):
            if hasattr(left, "op"):
                lops.insert(0, left.op)
                left = left.left
            if hasattr(right, "op"):
                rops.insert(0, right.op)
                right = right.right

        return lops, rops

    def _get_state(self):
        return (self.left._get_state(), self.right._get_state())

    def _set_state(self, state, funcdefs=None):
        raise NotImplementedError(
            "This method is not available yet with "
            "the 'ConvolvedModel' class."
        )

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function arguments for all functions."""
        out = self.right._make_all_args(params=params, **kwargs)
        out.update(self.left._make_all_args(params=params, **kwargs))
        return out

    def _convolve(
        self, left, right, params=None, returnComponents=None, **kwargs
    ):
        r"""Perform a convolution between `left` and `right`.

        If the convolutions between function in `left` and the
        function in `right` is defined in `convolutions` attribute,
        then use this corresponding function for analytical
        convolution.
        Else, the behavior is determined by the `on_undefined_conv`
        parameter.

        Parameters
        ----------
        left : :class:`Model` or :class:`CompositeModel`
            Model or CompositeModel to be used for convolution.
        right : :class:`Model` or :class:`CompositeModel`
            Model or CompositeModel to be used for convolution.
        params : Parameters, optional
            Parameters to be given to the model functions.
        kwargs : dict
            Additional keyword arguments to pass to the model functions.

        Returns
        -------
        An array containing the result of the convolution.

        Notes
        -----
        This composition of models works differently than for other operators,
        as the operators is applied between each pair of components in left
        and right components.
        That is, for ``left = a1 + a2 - a3`` and ``right = b1 * b2``, where
        a's and b's are instances of :class:`Model` class:

        .. math::
            left \otimes right = (a_1 \otimes b_1 * a_1 \otimes b_2)
                                 + (a_2 \otimes b_1 * a_2 \otimes b_2)
                                 - (a_3 \otimes b_1 * a_3 \otimes b_2)

        """
        if returnComponents == "right":
            tmp = left
            left = right
            right = tmp

        lcomponents = left.components
        rcomponents = right.components

        lops, rops = self._operators()

        models = []
        compPrefix = []
        for lidx, lcomp in enumerate(lcomponents):
            tmpRes = []
            for ridx, rcomp in enumerate(rcomponents):
                if lcomp.func.__name__ in self.convMap.keys():
                    funcName = rcomp.func.__name__
                    leftConvMap = self.convMap[lcomp.func.__name__]
                    if funcName in leftConvMap.keys():
                        if params is None:
                            params = self.make_params()
                        convFunc = leftConvMap[funcName]
                        tmpRes.append(convFunc(lcomp, rcomp, params, **kwargs))
                    else:
                        if self._on_undefined_conv == "numeric":
                            convFunc = lambda l, r: fftconvolve(
                                l, r, mode="same", axes=-1
                            )
                            tmpRes.append(
                                CompositeModel(lcomp, rcomp, convFunc).eval(
                                    params, **kwargs
                                )
                            )
                        elif self._on_undefined_conv == "raise":
                            raise KeyError(
                                "Convolution function between %s and %s is "
                                "not defined."
                                % (lcomp.func.__name__, funcName)
                            )
                else:
                    if self._on_undefined_conv == "numeric":
                        convFunc = lambda l, r: fftconvolve(
                            l, r, mode="same", axes=-1
                        )
                        tmpRes.append(
                            CompositeModel(lcomp, rcomp, convFunc).eval(
                                params, **kwargs
                            )
                        )

            # apply operators from the right
            if len(tmpRes) > 1:
                for idx, rop in enumerate(rops):
                    tmpRes[0] = rop(tmpRes[idx], tmpRes[idx + 1])
            models.append(tmpRes[0])
            compPrefix.append(lcomp.prefix)

        if returnComponents is None:
            # finally apply operators from the left
            if len(models) > 1:
                for idx, lop in enumerate(lops):
                    models[0] = lop(models[idx], models[idx + 1])
            return models[0]
        else:
            out = OrderedDict()
            for val in zip(compPrefix, models):
                out[val[0]] = val[1]
            return out
