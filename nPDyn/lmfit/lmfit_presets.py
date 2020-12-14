"""This module provides several function builders that can
be used to fit your data.

These functions generate a `Model` class instance from
the **lmfit** package [#]_.


References
----------

.. [#] https://lmfit.github.io/lmfit-py/

"""

from types import FunctionType, CodeType

from string import Formatter

import numpy as np

from scipy.fft import fft, fftshift
from scipy.special import wofz, spherical_jn

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


from nPDyn.models.d2O_calibration.interpD2O import getD2Odata


# -------------------------------------------------------
# Function builder for lmfit
# -------------------------------------------------------
def build_2D_model(
    q,
    funcName,
    funcBody,
    defVals=None,
    bounds=None,
    vary=None,
    expr=None,
    paramGlobals=None,
    prefix="",
    var="x",
):
    """Builds a 2D `lmfit.Model`.

    Parameters
    ----------
    q : np.array or list
        momentum transfer q values of scattering.
    funcName : str
        name of the function to be built.
    funcBody : str
        formatted string for the function to be used (in 1D).
        For a gaussian the string
        `"{a} * np.exp(-(x - {cen})**2 / {width}*{q}**2)"`
        will lead to a model with parameters of root names
        'a', 'cen' and 'width'. If these parameters are not in
        argument `paramGlobals`, the parameter names will be
        'a_1', 'a_2', 'a_3', ..., 'a_n', where n is the length
        of the array `q`.
    defVals : dict, optional
        dictionary of default values for the parameters of the
        form {'a': 1., 'cen': 0.05, 'width': 2}.
        If None, set to 1.0 for all parameters.
    bounds : dict, optional
        dictionary of bounds for the parameters of the form
        {'a': (0., np.inf}, 'cen': (-10, 10)}.
        If None, set to (-np.inf, np.inf) for all parameters.
    vary : dict, optional
        dictionary of parameter hint 'vary' for the parameters of
        the form {'a': False, 'cen': True}.
        If None, set to True for all parameters.
    expr : dict, optional
        dictionary of parameter hint 'expr' for the parameters of
        the form {'a': 'width / sqrt(2)'}.
        If None, set to None for all parameters.
    paramGlobals : list, optional
        defines which parameters should be considered as 'global', that is,
        a parameter that is fixed for all momentum transfer q values.
        If set to ['width'], then the resulting model will have parameters
        of the form ('a_1', ..., 'a_n', 'cen_1', ..., 'cen_n', 'width'), where
        n is the length of the parameter `q`.
    prefix : str, optional
        prefix to be given to the model name
    var : str, optional
        name of the primary independent variable
        (default 'x')

    """
    # make sure q is a numpy array of shape (q.size, 1)
    if isinstance(q, list):
        q = np.array(q)
    if isinstance(q, (int, float)):
        q = np.array([q])
    q = q.reshape((q.size, 1))

    # parse the parameter names to check for "q"
    paramNames = [
        pName for _, pName, _, _ in Formatter().parse(funcBody) if pName
    ]
    paramNames = list(set(paramNames))
    param_q = True if "q" in paramNames else False
    paramNames = [p for p in paramNames if p != "q"]

    # check default values, bounds and expression
    # if none, try to give default parameters
    paramDefVals = {p: 1.0 for p in paramNames}
    paramBounds = {p: (-np.inf, np.inf) for p in paramNames}
    paramVary = {p: True for p in paramNames}
    paramExpr = {p: None for p in paramNames}

    if defVals is not None:
        paramDefVals.update(defVals)
    if bounds is not None:
        paramBounds.update(bounds)
    if vary is not None:
        paramVary.update(vary)
    if expr is not None:
        paramExpr.update(expr)

    if paramGlobals is None:
        paramGlobals = []

    # set the parameter dictionary for `lmfit.Model` function
    params = {}
    for pId, pName in enumerate(paramNames):
        if pName in paramGlobals:
            params[pName] = (
                paramDefVals[pName],
                paramBounds[pName][0],
                paramBounds[pName][1],
                paramVary[pName],
                paramExpr[pName],
            )
        else:
            for qId, qVal in enumerate(q):
                params[pName + "_%i" % qId] = (
                    paramDefVals[pName],
                    paramBounds[pName][0],
                    paramBounds[pName][1],
                    paramVary[pName],
                    paramExpr[pName],
                )

    # set the function text
    funcTxt = "def {funcName}({var}, q, {funcArgs}):\n".format(
        funcName=funcName,
        var=var,
        funcArgs=", ".join(
            ["%s=%f" % (key, val[0]) for key, val in params.items()]
        ),
    )
    funcTxt += "    out = []\n"
    for qId, qVal in enumerate(q):
        pMap = {"q": "%f" % qVal} if param_q else {}
        for p in paramNames:
            if p in paramGlobals:
                pMap.update({p: p})
            else:
                pMap.update({p: p + "_%i" % qId})
        funcTxt += "    out.append(" + funcBody.format(**pMap) + ")\n"
    funcTxt += "    out = np.vstack(out)\n"
    funcTxt += "    return out"

    # compile the function
    func_code = compile(funcTxt, "<string>", "exec")
    code = [
        entry for entry in func_code.co_consts if isinstance(entry, CodeType)
    ][0]
    func = FunctionType(
        code, globals(), funcName, argdefs=func_code.co_consts[-1]
    )

    # instantiate the `lmfit.Model`
    model = Model(func, independent_vars=[var, "q"], prefix=prefix)

    # set the parameter hints
    for p in paramNames:
        if p in paramGlobals:
            model.set_param_hint(
                p,
                min=params[p][1],
                max=params[p][2],
                vary=params[p][3],
                expr=params[p][4],
            )
        else:
            for qId, qVal in enumerate(q):
                key = p + "_%i" % qId
                # add suffix to parameter in expr if expr is not None
                qExpr = None
                if params[key][4] is not None:
                    if p in params[key][4]:
                        qExpr = params[key][4].replace(p, "%s_%i" % (p, qId))
                model.set_param_hint(
                    key,
                    min=params[key][1],
                    max=params[key][2],
                    vary=params[key][3],
                    expr=qExpr,
                )

    return model


# -------------------------------------------------------
# Defines the functions for the models
# -------------------------------------------------------
def linear(q, **kwargs):
    """Linear model that can be used for background.

    The model reads: :math:`a*x + b`

    Notes
    -----
    Two parameters:
        - a
        - b

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"a": 0, "b": 0.1},
        "bounds": {"a": (-np.inf, np.inf), "b": (0.0, np.inf)},
    }

    defaults.update(kwargs)

    model = build_2D_model(
        q, "linear", "np.zeros_like(x) + {a} * x + {b}", **defaults
    )

    return model


def hline(q, **kwargs):
    """A horizontal line."""
    # set default values (will be overridden by any in 'kwargs')
    defaults = {"defVals": {"b": 0.1}, "bounds": {"b": (0.0, np.inf)}}

    defaults.update(kwargs)

    model = build_2D_model(q, "linear", "np.zeros_like(x) + {b}", **defaults)

    return model


def gaussian(q, qwise=True, **kwargs):
    """Normalized Gaussian lineshape.

    .. math::

        \\rm G(x, q; a, c, \\sigma) =
            \\frac{a}{\\sqrt(\\pi \\sigma} e^{-(x - c)^2 / \\sigma}

    where the shape of the output array depends on the shape of the
    independent variable `q` and :math:`\\sigma` can have an
    explicit dependence on q as :math:`\\sigma q**2`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    qwise : bool, optional
        whether the width (sigma) has explicit dependence on q
        (default False)
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - center
        - sigma

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"amplitude": 1, "center": 0.0, "sigma": 1},
        "bounds": {
            "amplitude": (0.0, np.inf),
            "center": (-np.inf, np.inf),
            "sigma": (0.0, np.inf),
        },
    }

    defaults.update(kwargs)

    if qwise:
        model = build_2D_model(
            q,
            "gaussian",
            "{amplitude} / np.sqrt(np.pi * 2 * {sigma}**2) "
            "* np.exp(-(x - {center})**2 / (2 * {sigma}**2))",
            **defaults
        )
    else:
        model = build_2D_model(
            q,
            "gaussian",
            "{amplitude} / np.sqrt(np.pi * 2 * ({sigma} * {q}**2)**2) "
            "* np.exp(-(x - {center})**2 / (2 * ({sigma} * {q}**2)**2))",
            paramGlobals=["sigma"],
            **defaults
        )

    return model


def lorentzian(q, qwise=False, **kwargs):
    """Normalized Lorentzian lineshape.

    .. math::

        \\rm \\mathcal{L}(x, q; a, c, \\sigma) =
            \\frac{a}{\\pi} \\frac{\\sigma}{(x - c)^2 + \\sigma^2}

    where the shape of the output array depends on the shape of the
    independent variable `q` and :math:`\\sigma` can have an
    explicit dependence on q as :math:`\\sigma q**2`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    qwise : bool, optional
        whether the width (sigma) has explicit dependence on q
        (default False)
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - center
        - sigma

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"amplitude": 1, "center": 0.0, "sigma": 2},
        "bounds": {
            "amplitude": (0.0, np.inf),
            "center": (-np.inf, np.inf),
            "sigma": (0.0, np.inf),
        },
    }

    defaults.update(kwargs)

    if qwise:
        model = build_2D_model(
            q,
            "lorentzian",
            "{amplitude} / np.pi "
            "* {sigma} / ((x - {center})**2 + {sigma}**2)",
            **defaults
        )
    else:
        model = build_2D_model(
            q,
            "lorentzian",
            "{amplitude} / np.pi "
            "* {sigma} * {q}**2 / ((x - {center})**2 + ({sigma} * {q}**2)**2)",
            paramGlobals=["sigma", "amplitude"],
            **defaults
        )

    return model


def jump_diff(q, qwise=False, **kwargs):
    """Normalized Lorentzian with jump-diffusion model.

    The shape of the output array depends on the shape of the
    independent variable `q`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    qwise : bool, optional
        whether the width (sigma) has explicit dependence on q
        (default False)
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - center
        - sigma
        - tau

    References
    ----------
    For more information see: http://doi.org/10.1103/PhysRev.119.863

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"amplitude": 1, "center": 0.0, "sigma": 1, "tau": 1},
        "bounds": {
            "amplitude": (0.0, np.inf),
            "center": (-np.inf, np.inf),
            "sigma": (0.0, np.inf),
            "tau": (0.0, np.inf),
        },
    }

    defaults.update(kwargs)

    if qwise:
        model = build_2D_model(
            q,
            "jump_diff",
            "{amplitude} / np.pi "
            "* {sigma} / ((x - {center})**2 + {sigma}**2)",
            **defaults
        )
    else:
        model = build_2D_model(
            q,
            "jump_diff",
            "{amplitude} / np.pi * "
            "({sigma} * {q}**2 / (1 + {sigma}*{q}**2*{tau})) / "
            "((x - {center})**2 + ({sigma} * {q}**2 / "
            "(1 + {sigma} * {q}**2 * {tau}))**2)",
            paramGlobals=["amplitude", "sigma", "tau"],
            **defaults
        )

    return model


def delta(q, **kwargs):
    """Normalized Dirac delta.

    where the shape of the output array depends on the shape of the
    independent variable `q`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - center

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"amplitude": 1, "center": 0.0},
        "bounds": {"amplitude": (0.0, np.inf), "center": (-np.inf, np.inf)},
    }

    defaults.update(kwargs)

    model = build_2D_model(
        q, "delta", "getDelta(x, {amplitude}, {center})", **defaults
    )

    return model


def calibratedD2O(q, volFraction, temp, **kwargs):
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
    kwargs : dict, optional
        Additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"amplitude": 1},
        "bounds": {"amplitude": (0.0, np.inf)},
    }

    defaults.update(kwargs)

    model = build_2D_model(
        q,
        "calibrationD2O",
        "{amplitude} * getD2Odata(%s)(%f, {q}) "
        "/ (np.pi * (x**2 + getD2Odata(%s)(%f, {q})**2))"
        % (volFraction, temp, volFraction, temp),
        **defaults
    )

    return model


def rotations(q, qwise=False, **kwargs):
    """Normalized Lorentzian accounting for rotational motions in liquids.

    .. math::

        S_r(q, \\omega) = A_r J_0^2(qd) \\delta(\\omega) +
            \\sum_{l=1} (2l + 1) J_l^2(qd) \\frac{1}{\\pi}
            \\frac{l(l+1) \\sigma}{(\\omega - center)^2 + (l(l+1) \\sigma)^2}

    The shape of the output array depends on the shape of the
    independent variable `q`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    qwise : bool, optional
        whether the width (sigma) has explicit dependence on q
        (default False)
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - center
        - sigma
        - bondDist

    References
    ----------
    For more information see: http://doi.org/10.1139/p66-108

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {
            "amplitude": 1,
            "center": 0.0,
            "sigma": 1,
            "bondDist": 0.96,
        },  # default for O-H distance in water
        "bounds": {
            "amplitude": (0.0, np.inf),
            "center": (-np.inf, np.inf),
            "sigma": (0.0, np.inf),
            "bondDist": (0.0, np.inf),
        },
        "vary": {"bondDist": False},
        "paramGlobals": ["bondDist"],
    }

    # parse the keywords arguments provided by the user
    if not qwise:
        paramGlobals = ["sigma", "amplitude", "bondDist"]
        if "paramGlobals" in kwargs.keys():
            paramGlobals = list(set(paramGlobals + kwargs.pop("paramGlobals")))
        kwargs["paramGlobals"] = paramGlobals

    defaults.update(kwargs)

    model = build_2D_model(
        q,
        "rotations",
        "spherical_jn(0, {q}*{bondDist})**2 * "
        "getDelta(x, {amplitude}, {center})"
        " + "
        "{amplitude} * np.sum([spherical_jn(l, {q}*{bondDist})**2 "
        "* (2 * l + 1) * "
        "l * (l + 1) * {sigma} / (np.pi * ((x - {center})**2 + "
        "(l * (l + 1) * {sigma})**2)) for l in range(1, 5)], axis=0)",
        **defaults
    )

    return model


def voigt(q, **kwargs):
    """Voigt profile.

    The shape of the output array depends on the shape of the
    independent variable `q`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - center
        - sigma
        - gamma

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"amplitude": 1, "center": 0.0, "sigma": 1, "gamma": 1},
        "bounds": {
            "amplitude": (0.0, np.inf),
            "center": (-np.inf, np.inf),
            "sigma": (0.0, np.inf),
            "gamma": (0.0, np.inf),
        },
    }

    defaults.update(kwargs)

    model = build_2D_model(
        q,
        "voigt",
        "{amplitude} * wofz((x - {center}) + 1j * {gamma}).real / "
        "({sigma} * np.sqrt(2)) / ({sigma} * np.sqrt(2 * np.pi))",
        **defaults
    )

    return model


def pseudo_voigt(q, **kwargs):
    """Pseudo-Voigt profile.

    The shape of the output array depends on the shape of the
    independent variable `q`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - fraction
        - center
        - sigma

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {
            "amplitude": 1,
            "center": 0.0,
            "sigma": 1,
            "fraction": 0.5,
        },
        "bounds": {
            "amplitude": (0.0, np.inf),
            "center": (-np.inf, np.inf),
            "sigma": (0.0, np.inf),
            "fraction": (0.0, 1.0),
        },
    }

    defaults.update(kwargs)

    model = build_2D_model(
        q,
        "pvoigt",
        "{amplitude} * ({fraction} * np.exp(-(x - {center})**2 / "
        "(2 * ({sigma} / np.sqrt(2 * np.log(2)))**2)) / "
        "np.sqrt(2 * np.pi * ({sigma} / np.sqrt(2 * np.log(2)))**2) +"
        "(1 - {fraction}) * {sigma} / (np.pi * ((x - {center})**2 +"
        "{sigma}**2)))",
        **defaults
    )

    return model


def kww(q, **kwargs):
    """Fourier transform of the Kohlrausch-William-Watts (KWW) function.

    The shape of the output array depends on the shape of the
    independent variable `q`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - tau
        - beta

    References
    ----------
    For more information, see:
    https://en.wikipedia.org/wiki/Stretched_exponential_function

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {"amplitude": 1, "tau": 1, "beta": 1.0},
        "bounds": {
            "amplitude": (0.0, np.inf),
            "tau": (0.0, np.inf),
            "beta": (0.0, np.inf),
        },
    }

    defaults.update(kwargs)

    model = build_2D_model(
        q,
        "kww",
        "{amplitude} * fftshift(fft(np.exp(-t / {tau})**{beta})).real",
        var="t",
        **defaults
    )

    return model


def two_diff_state(q, qwise=False, **kwargs):
    """Two state switching diffusion model.

    The shape of the output array depends on the shape of the
    independent variable `q`.

    Parameters
    ----------
    q : np.array or list
        array of momentum transfer q values
    kwargs : dict, optional
        additional keywords to pass to :func:`build_2D_model`.

    Notes
    -----
    The parameter root names are:
        - amplitude
        - center
        - gamma1
        - gamma2
        - tau1
        - tau2

    References
    ----------
    For more information, see: http://doi.org/10.1063/1.4950889
    or http://doi.org/10.1039/C4CP04944F

    """
    # set default values (will be overridden by any in 'kwargs')
    defaults = {
        "defVals": {
            "amplitude": 1,
            "center": 0.0,
            "gamma1": 1.0,
            "gamma2": 1.0,
            "tau1": 1.0,
            "tau2": 1.0,
        },
        "bounds": {
            "amplitude": (0.0, np.inf),
            "center": (-np.inf, np.inf),
            "gamma1": (0.0, np.inf),
            "gamma2": (0.0, np.inf),
            "tau1": (0.0, np.inf),
            "tau2": (0.0, np.inf),
        },
    }

    defaults.update(kwargs)

    # parse the keywords arguments provided by the user
    if not qwise:
        paramGlobals = ["amplitude", "gamma1", "gamma2", "tau1", "tau2"]
        if "paramGlobals" in kwargs.keys():
            paramGlobals = list(set(paramGlobals + kwargs.pop("paramGlobals")))
        kwargs["paramGlobals"] = paramGlobals

    # define the different component of the model
    Lambda = (
        "({gamma1} - {gamma2} + (1 / {tau1}) - (1 / {tau2}))**2"
        "+ 4 / ({tau1} * {tau2})"
    )

    lambda_1 = "{gamma1} + (1 / {tau1}) + {gamma2} + (1 / {tau2}) / 2"
    lambda_1 = lambda_1 + " + " + Lambda + " / 2"

    lambda_2 = "{gamma1} + (1 / {tau1}) + {gamma2} + (1 / {tau2}) / 2"
    lambda_2 = lambda_1 + " - " + Lambda + " / 2"

    alpha = (
        "1 / (" + lambda_2 + " - " + lambda_1 + ") * ("
        "{tau1} / ({tau1} + {tau2}) * "
        "({gamma2} + (1 / {tau1}) + (1 / {tau2}) - " + lambda_1 + ")+"
        "{tau2} / ({tau1} + {tau2}) * "
        "({gamma1} + (1 / {tau1}) + (1 / {tau2}) - " + lambda_1 + "))"
    )

    # build the models for the two coupled Lorentzians
    model = build_2D_model(
        q,
        "two_diff_state",
        "{amplitude} * (" + alpha + " * " + lambda_1 + " /"
        "(np.pi * (x - {center})**2 + " + lambda_1 + "**2) +"
        "(1 - " + alpha + ")" + " * " + lambda_2 + " /"
        "(np.pi * (x - {center})**2 + " + lambda_2 + "**2))",
        **defaults
    )

    return model


# -------------------------------------------------------
# Helper functions for the models
# -------------------------------------------------------
def getDelta(x, amplitude, center):
    """Helper function for the Dirac delta model."""

    out = np.zeros_like(x)

    peakPos = np.argmin((x - center) ** 2)

    out[peakPos] = 1 / (x[1] - x[0])
    out = amplitude * out

    return out
