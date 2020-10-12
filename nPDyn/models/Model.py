""" This module provides a template class to build models 
    that can be used to fit the data.

    The documentation of the :func:`setModel` function provides all
    the information on how to build and use the model, as well
    as the description of the available methods.

"""


import numpy as np


# -------------------------------------------------------
# Model class
# -------------------------------------------------------
class Model:
    """ This class can be used to define a model fully compatible
        with other fitting and plotting methods from nPDyn.

        Just like other specialized data types, the instance of
        a dataType class is decorated such that all attributes
        and methods in the parent class remains available and
        new attributes and methods are added dynamically.

        Use the following to add a model to a dataType instance:

        .. code-block:: python

            import nPDyn

            from nPDyn.dataTypes.Model import setModel

            data = nPDyn.Dataset()
            data.importFiles(QENSFiles=['myFile.nxs'])

            data.datasetList[0] = setModel(data.datasetList[0])
            data.datasetList[0].addComponent(**myKwargs)

        The model can be easily reset to its initial state using:

        .. code-block:: python

            # This won't keep trace of any previous model's attributes
            data.datasetList[0] = setModel(data.datasetList[0])

        Also a predefined model can be used as a set of *kwargs*
        to pass to the function.

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
        dataset : nPDyn dataType class
            a specialized dataType class as defined in nPDyn.
        params : dict
            dictionary of parameter names with their associated
            values. 
        eisfComponents : list
            list of model components that account for the
            elastic incoherent structure factor (eisf).
        qisfComponents : list
            list of model components that account for the
            quasi-elastic incoherent structure factor (qisf).
        bkgdComponents : list
            list of model components that account for the
            background (bkgd).
        xVar : array-like
            array of values for the variable to be used
            (energies, temperatures, or other).
        xVarName : str
            name for the xVar array
        fitMethod : str
            fitting method to be used to optimize the parameters.
            Currently, the following are supported: `curve_fit`,
            `basinhopping`

    """


    def __init__(self, 
                 dataset=None,
                 params=None,
                 fixedParams=None,
                 eisfComponents=None,
                 qisfComponents=None,
                 bkgdComponents=None,
                 xVar=None,
                 fitMethod=None): 

        self.dataset = dataset

        self.params = params if params is not None else {}
        self.fixedParams = fixedParams if fixedParams is not None else []
        self.paramsErr = {}

        self.eisfComponents = (eisfComponents if eisfComponents 
                               is not None else {})
        self.qisfComponents = (qisfComponents if qisfComponents
                               is not None else {})
        self.bkgdComponents = (bkgdComponents if bkgdComponents
                               is not None else {})

        self.xVar = xVar

        self.fitMethod = fitMethod

    # --------------------------------------------------
    # fitting
    # --------------------------------------------------

    # --------------------------------------------------
    # accessors
    # --------------------------------------------------
    def getModel(self, xVar=None, params=None, convolve=None):
        """ Performs the assembly of the components and call
            the provided functions with their parameters to 
            compute the model.


            Parameters
            ----------
            xVar : np.ndarray
                values for the x-axis variable
            params : list, np.array, optional
                parameters to be passed to the components.
            convolve : :class:`Model`
                another model to be convolved with this one.


            Returns
            -------
            The computed model in an array, the dimensions of which depend
            on the :attr:`xVar` attribute and the function called.

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


        if xVar is None:
            xVar = self.xVar

        # gets the output arrays for each component and sum
        components = {**self.eisfComponents, 
                      **self.qisfComponents}

        res = []
        for key, comp in components.items():
            if convolve is None:
                res.append(comp[0](xVar, *comp[1](params)))
            else:
                res.append(self._convolve(xVar, params, comp, convolve))

        # the background components are not convoluted by default
        for key, comp in self.bkgdComponents.items():
            res.append(comp[0](xVar, *comp[1](params)))

        return np.array(res).sum(0)


    def getComponents(self, xVar=None, params=None, 
                      convolve=None, compType='eisf'):
        """ Performs the computation of the components 
            with the given x-axis values, parameters and 
            convolutions.


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

        if xVar is None:
            xVar = self.xVar

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
                res.append(comp[0](xVar, *comp[1](params)))
            else:
                res.append(self._convolve(xVar, params, comp, convolve))


        return names, np.array(res)


    def _paramsToList(self):
        """ Converts the dictionary of parameters to a list of values.

            Do not return the parameters that are in the :attr:`fixedParams`
            attribute.

        """
        
        params = []

        for key, val in self.params.items():
            if key not in self.fixedParams:
                if isinstance(val, (list, np.ndarray)):
                    params.append(val)
                else:  # assumes a integer or float
                    params.append([val])

        return np.concatenate(params).flatten()


    def _listToParams(self, pList):
        """ Use the given list to convert a list of parameters to
            a dictionary similar to the one of the model.
            The list should contain the same number of parameters as
            in the :attr:`params` of the class and in the same order
            as provided by `self.params.items()`.

        """

        params = {**self.params}

        pList = list(pList)

        for key, val in params.items():
            if key not in self.fixedParams:
                if isinstance(val, np.ndarray):
                    valShape = val.shape
                    valSize = val.size

                    newVal = [pList.pop(0) for idx in range(valSize)]
                    newVal = np.array(newVal).reshape(valShape)

                    params[key] = newVal

                elif isinstance(val, list):
                    valSize = len(val)

                    newVal = [pList.pop(0) for idx in range(valSize)]

                    params[key] = newVal

                else:  # assumes integer or float
                    params[key] = pList.pop(0)

        return params


    def _convolve(self, xVar, params, comp, model):
        """ This method allows to identify the type of function
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
