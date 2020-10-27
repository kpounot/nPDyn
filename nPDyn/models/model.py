"""This module provides a template class to build models 
that can be used to fit the data.

"""

import numpy as np
from copy import deepcopy
from inspect import signature
import ast
import astunparse


class findParamNames(ast.NodeTransformer):
    def __init__(self, params):
        """Helper class to parse strings to evaluation for function
        arguments in :class:`Component`.

        Parameters
        ----------
        params : :class:`Parameters`
            An instance of Parameters from which the parameter
            names are to be found and substituted by the corresponding
            values.

        """
        super().__init__()
        self.params = params

    def visit_Name(self, node):
        """Name visitor."""
        if node.id in self.params.keys():
            res = ast.Attribute(
                value=ast.Subscript(
                    value=ast.Name(id='params', ctx=node.ctx),
                    slice=ast.Index(value=ast.Str(s=node.id)),
                    ctx=node.ctx),
                attr='value',
                ctx=node.ctx)
            return res
        else:
            return node


class Model:
    def __init__(self, 
                 params=None,
                 eisfComponents=None,
                 qisfComponents=None,
                 bkgdComponents=None,
                 xLabel='$\hbar \omega ~ [\mu eV]$',
                 yLabel='$S(q, \omega)$',
                 fitMethod=None,
                 convolutions=None): 
        """Model class to be used within nPDyn.

        The model is structured in components that can be added 
        together, each component consisting of a name, a callable
        function and a dictionary of parameters. The parameters
        of two different components can have the same name such
        that they can be shared by several components just like
        for the switching diffusive state model.

        Also, the components are separated in two classes, namely
        *eisfComponents* and *qisfComponents*, in order to
        provide the possibility to separatly extract the elastic
        and quasi-elastic parts for analysis and plotting.


        Parameters
        ----------
        params : :class:`Parameters` instance
            Parameters to be used with the model
        eisfComponents : dict
            List of model components that account for the
            elastic incoherent structure factor (eisf).
        qisfComponents : dict
            List of model components that account for the
            elastic incoherent structure factor (eisf).
        bkgdComponents : dict
            List of model components that account for the
            background (bkgd).
        xLabel : str
            Label for the primary variable to be used.
        yLabel : str
            Label for the y-axis (what the model values represent).
        convolutions : dict of dict
            Dictionary that defines the mapping '(function1, function2)'
            to 'convolutionFunction(function1, function2)'. Analytic
            convolutions or user defined operators can be defined
            this way.

        """
        self.params = deepcopy(params) if params is not None else {}
        self.eisfComponents = (eisfComponents if eisfComponents 
                               is not None else {})
        self.qisfComponents = (qisfComponents if qisfComponents
                               is not None else {})
        self.bkgdComponents = (bkgdComponents if bkgdComponents
                               is not None else {})
        self.xLabel = xLabel
        self.yLabel = yLabel

    def addComponent(self, compType, name, func, **funcArgs):
        """Add a component to the model.

        This function builds a :class:`Component` instance
        automatically, such that the user does not have to
        build it himself.
        
        Parameters
        ----------
        compType : {'eisf', 'qisf', 'bkgd'}
            Type of component to be added. Either elastic
            incoherent structure factor ('eisf'), quasi-elastic
            incoherent structure factor ('qisf') or background ('bkgd').
        name : str
            Name of the component.
        func : callable
            Callable function for the component.
        funcArgs : dict
            Dictionary of values, expressions for function arguments
            as described in :class:`Component`.
        
        """
        component = Component(func, **funcArgs)
        if compType == 'eisf':
            self._eisfComponents[name] = component
        if compType == 'qisf':
            self._qisfComponents[name] = component
        if compType == 'bkgd':
            self._bkgdComponents[name] = component

    # --------------------------------------------------
    # fitting
    # --------------------------------------------------
    def fit(self, x, params=None, data=None, errors=None,
            fit_method='curve_fit', fit_kws=None, **kwargs):
        pass

    # --------------------------------------------------
    # accessors
    # --------------------------------------------------
    def getModel(self, x, params=None, convolve=None, **kwargs):
        """Perform the assembly of the components and call
         the provided functions with their parameters to 
         compute the model.

        Parameters
        ----------
        xVar : np.ndarray
            Values for the x-axis variable
        params : list, np.array, optional
            Parameters to be passed to the components.
            Will override existing parameters in `self.params`.
        convolve : :class:`Model`
            Another model to be convolved with this one.
        kwargs:
            Additional keyword arguments to be passed to the components.

        Returns
        -------
        The computed model in an array, the dimensions of which depend
        on `xVar` and `params` attributes and the function called.

        """
        res = []
    
        # if no parameters override, use the ones from the class instance
        if params is None:
            params = self.params
        else:  # Override self.params values with the ones from params
            if isinstance(params, dict):
                params = {**self.params, **params}
            else:  # assumes a list or numpy array
                params = self._listToParams(params)

        components = {**self.eisfComponents, 
                      **self.qisfComponents}

        res = []
        for key, comp in components.items():
            if convolve is None:
                res.append(comp.eval(x, params, **kwargs))
            else:
                res.append(self._convolve(x, params, comp, convolve, **kwargs))

        # the background components are not convoluted by default
        for key, comp in self.bkgdComponents.items():
            res.append(comp.eval(x, params, **kwargs))

        return np.array(res).sum(0)

    def getComponents(self, x, params=None, convolve=None, 
                      compType='eisf', **kwargs):
        """Performs the computation of the components 
        with the given x-axis values, parameters and convolutions.


        Parameters
        ----------
        xVar : np.ndarray
            values for the x-axis variable
        params : list, np.array, optional
            parameters to be passed to the components.
        convolve : :class:`Model`
            another model to be convolved with this one.
        compType : {'eisf', 'qisf', 'bkgd', 'all'}
            component type to return, either 'eisf', 'qisf',
            'bkgd' or 'all'


        Returns
        -------
        Two lists, the first containing the components names and
        the second containing the computed lineshapes.

        """
        names = []
        res = []
    
        # if no parameters override, use the ones from the class instance
        if params is None:
            params = self.params
        else:  # Override self.params values with the ones from params
            if isinstance(params, dict):
                params = {**self.params, **params}
            else:  # assumes a list or numpy array
                params = self._listToParams(params)

        if compType == 'eisf':
            comps = self.eisfComponents
        if compType == 'qisf':
            comps = self.qisfComponents
        if compType == 'bkgd':
            comps = self.bkgdComponents
        if compType == 'all':
            comps = {**self.eisfComponents,
                     **self.qisfComponents,
                     **self.bkgdComponents}

        # gets the output arrays for each component and sum
        for key, comp in comps.items():
            names.append(key)
            if convolve is None:
                res.append(comp.eval(x, params, **kwargs))
            else:
                res.append(self._convolve(x, params, comp, convolve, **kwargs))

        return names, np.array(res)

    def _convolve(self, x, params, comp, model, **kwargs):
        """This method allows to identify the type of function
        given in the two components of the convolution.

        If these function are refered to in the third
        parameter of the component given with the *comp*
        attribute, an analytical convolution will be performed
        using the corresponding convolution function.
        Otherwise, a numerical convolution is performed.

        Parameters
        ----------
        xVar : np.ndarray
            array for values corresponding to the x-axis
        comp : tuple(function, parameters, [convolutions]) 
            component of the current :class:`Model` instance
            to be convolved with the components of the other
            model.
        model : :class:`Model`
            another :class:`Model` instance that is used for 
            convolution.

        """
        res = []

        convComps = {**model.eisfComponents, **model.qisfComponents}

        for key, val in convComps.items():
            if len(comp) == 2:  # no convolution defined, go numerical
                res.append(np.convolve(comp[0](xVar, *comp[1](params)),
                                       val[0](xVar, *val[1](model.params))))
            else:
                try:
                    convFunc = comp[2][val[0].__name__]
                    res.append(convFunc(xVar, comp, val, params, model.params))
                except KeyError:
                    res.append(np.convolve(comp[0](xVar, *comp[1](params)),
                                           val[0](xVar, *val[1](model.params))))

        return np.array(res).sum(0)


class Component:
    def __init__(self, func, **funcArgs):
        """Component class to be used with the :class:`Model` class.

        Parameters
        ----------
        func : callable
            The function to be used for this component.
        funcArgs : dict of str, int, float or arrays
            Values to be passed to the function arguments.
            This is a dicitonary of argument names mapped to values.
            The values can be of different types:
                - **int, float or array** - the values are directly passed to
                the function.
                - **str** - the values are evaluated first. If any word in
                the string is present in the `Model.params` dictionary keys,
                the corresponding parameter value is substituted.

        Examples
        --------
        For a `Model` class that has the following key in its `params`
        attribute: ('amplitude', 'sigma'), the component for a 
        Lorentzian, the width of which depends on a defined vector *q*,
        can be created using:

        >>> def lorentzian(x, amplitude, sigma):
        ...     return amplitude / np.pi * sigma / (x**2 + sigma**2)
        >>> myComp = Component(
        ...     lorentzian, amplitude='amplitude', sigma='sigma * q**2')

        If the Lorentzian width is constant, use:

        >>> myComp = Component(lorentzian, amplitude='amplitude', sigma=5)

        """
        if not hasattr(func, "__call__"):
            raise AttributeError("Parameter 'func' should be a callable.")
        self.func = func

        self.funcArgs = {}
        self._guessArgs()
        self.funcArgs.update(funcArgs)

    def eval(self, x, params, **kwargs):
        """Evaluate the components using the given parameters.
        
        Parameters
        ----------
        params : :class:`Parameters` instance
            Parameters to be passed to the component
        kwargs : dict
            Additional parameters to be passed to the function.
            Can override params.
        
        """
        params = deepcopy(params)

        for key, val in kwargs.items():
            if isinstance(val, (int, float, list, np.ndarray)):
                params.update(key, value=val)
            elif isinstance(val, (tuple)):
                params.update(key, *val)
            else:  # assume a dictionary of attributes
                params.update(key, **val)

        args = {}
        for key, arg in self.funcArgs.items():
            if isinstance(arg, str):
                for pKey in params.keys():
                    arg = ast.parse(arg)
                    ast.fix_missing_locations(
                        findParamNames(params).visit(arg))
                args[key] = eval(astunparse.unparse(arg))
            else:
                args[key] = arg

        return self.func(x, **args)

    def _guessArgs(self):
        """Guess arguments from function signature."""
        sig = signature(self.func)
        for key, param in sig.parameters.items():
            if key != 'x':
                self.funcArgs[key] = key
