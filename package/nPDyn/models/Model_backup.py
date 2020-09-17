""" This module provides a template class to build models 
    that can be used to fit the data.

    The documentation of the :func:`setModel` function provides all
    the information on how to build and use the model, as well
    as the description of the available methods.

"""


from inspect import signature, _empty


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
        eisfComponents : list
            list of model components that account for the
            elastic incoherent structure factor (eisf).
        qisfComponents : list
            list of model components that account for the
            elastic incoherent structure factor (eisf).
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
                 dataset,
                 params=None,
                 fixedParams=None,
                 eisfComponents=None,
                 qisfComponents=None,
                 bkgdComponents=None,
                 xVar=None,
                 fitMethod=None): 

        self.dataset = dataset

        self.formatParamNames = (formatParamNames if formatParamNames 
                                 is not None else {})

        self.params = params if params is not None else {}
        self.fixedParams = fixedParams if fixedParams is not None else []
        self.paramsErr = {}

        self.eisfComponents = (eisfComponents if eisfComponents 
                               is not None else [])
        self.qisfComponents = (qisfComponents if qisfComponents
                               is not None else [])
        self.bkgdComponents = (bkgdComponents if bkgdComponents
                               is not None else [])

        self.xVar = None

        self.fitMethod = fitMethod

    # --------------------------------------------------
    # fitting
    # --------------------------------------------------

    # --------------------------------------------------
    # accessors
    # --------------------------------------------------
    def getModel(self, xVar=None, params=None, convolve=None, dataset=None):
        """ Performs the assembly of the components and call
            the provided functions with their parameters to 
            compute the model.


            Parameters
            ----------
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
            params = self.getAllParams()

        if xVar is None:
            xVar = self.xVar

        if dataset is None:
            dataset = self.dataset


        # gets the output arrays for each component and sum


    def getComponents(self, xVar=None, params=None, 
                      convolve=None, dataset=None):
        """ Performs the calculation of the different components in the 
            model and returns the resulting arrays in a dictionary
            of with the components names as keys.

            Parameters
            ----------
            convolve : :class:`Model`
                another model to be convolved with this one.

        """

        res = []

        allParams = self.getAllParams()



    def getComponentList(self):
        """ Returns the list of components by listing first
            all components in :attr:`eisfComponents`, then 
            in :attr:`qisfComponents`, and finally in
            :attr:`bkgdComponents`.

        """

        return self.eisfComponents + self.qisfComponents + self.bkgdComponents




    def getAllParams(self):
        """ Parses the parameters in the different components.

            Returns
            -------
            a dictionary with the parameter names as keys and their
            associated values (either initial or optimized if updated).

            Notes
            -----
            if a parameter name appears in several components, it will
            be considered as one parameter used by all the associated
            components and will be optimized once.

        """

        self._params = {}
        
        for comp in self.getComponentList():
            for pKey, val in comp.params.items():
                if pKey not in comp.fixedParams:
                    self._params[pKey] = val

        return self._params



# -------------------------------------------------------
# Component class
# -------------------------------------------------------
class Component:
    """ This class defines a component to be used within the 
        :class:`Model` class.

        The basic structure of the component is to have a set 
        of parameters to be optimized and a callable that returns
        a specific lineshape. Some parameters may also be fixed.

        To allow for analytic convolution, the :class:`Component`
        also contains a dictionary that link the other convoluted
        component to a function that returns the convoluted
        lineshape (e.g. sum of the half-width at half maximum for
        two Lorentzians, or Voigt profile for a convolution 
        between a Gaussian and a Lorentzian).

        Parameters
        ----------
        name : str
            type name of the component (used to define the rules for 
            analytical convolutions)
        func : callable
            function to be called to obtain the component's lineshape.
            For each argument in the function's signature, the same 
            name should correspond to a key in the :attr:`params` of
            this class or some of the values in the  :attr:`fixedParams` 
            attributes of the :class:`Model` class.
            The function can use a processed dataset using a *dataset*
            optional argument such that any properties like q-values
            can be used within the function.
            The function signature should be of the form,

            .. code-block:: python

                myFunc(x, param1, param2, param3=5, dataset=dataset)

        params : dict, optional
            dictionary of parameters names as keys and starting values
            as values. If not empty, will override the default values
            obtained from the :attr:`func` attribute signature.
        convolutions : dict, optional
            dictionary in which the keys refers to the name of another
            component to be convoluted with and the associated value is
            a function with the following signature,
            `convolutionFunc(x, params, thisComp, Comp2)` where *x* is the
            x-axis array of values, *params* are the parameters to be passed 
            to the Component, *thisComp* is this :class:`Component` class
            instance and *comp2* is the other component to be used for the
            convolution.

    """

    def __init__(self, 
                 name, 
                 func, 
                 params=None, 
                 fixedParams=None, 
                 convolutions=None):

        self.name = name

        self.func = func

        self.params = params if params is not None else {}
        self.fixedParams = fixedParams if fixedParams is not None else []

        self.paramsErr = {}

        sig = list(signature(self.func).parameters.items())
        for val in sig:
            if val[0] not in list(self.params.keys()):
                if val[0] not in ['x', 'dataset']:
                    default = val[1].default
                    if default == _empty:
                        self.params[val[0]] = 1
                    else:
                        self.params[val[0]] = default

        self.convolutions = convolutions if convolutions is not None else {}


    def getComponent(self, x, params=None, dataset=None):
        """ Accessor to the component lineshape.

            The methods calls the :attr:`func` of the class
            with the given values in x-axis and the given parameters.

            Any set of parameters can be given, the method makes use
            of the :attr:`func` signature to pick the parameters
            needed only.

            Parameters
            ----------
            x : np.ndarray
                array of values for the x-axis (or higher dimensional).
            params : list, np.ndarray
                list of parameters values
                
        """

        if params is None:
            params = list(self.params.values())

        return self.func(x, *params, dataset=dataset)


    def getConvolved(self, x, params=None, comp=None, dataset=None):
        """ Accessor to the component lineshape.

            The methods calls the :attr:`func` of the class
            with the given values in x-axis and the given parameters.

            Any set of parameters can be given, the method makes use
            of the :attr:`func` signature to pick the parameters
            needed only.

            Parameters
            ----------
            x : np.ndarray
                array of values for the x-axis (or higher dimensional).
            params : dict
                dictionary of parameters with their names as key and 
                their associated values. Can contain parameters
                not used by the function too.
            comp : :class:`Component` or :class:`Model`
                another :class:`Component` instance to be convolved with
                this one. The name should match one name in the 
                :attr:`convolutions` dictionary attribute for the method
                to call the right convolution method.

        """
    
        if params is None:
            params = list(self.params.values())

        try:
            if isinstance(comp, Component):
                res = self.convolutions[comp.name](
                    x, params, self, comp, dataset)
            elif isinstance(comp, Model):
                res = []
                for sglComp in comp.getComponentList():
                    res.append(self.convolutions[comp.name](
                        x, params, self, comp, dataset))
                res = np.array(res).sum(0)
        except KeyError:
            print("The convolution with this Component instance "
                  " is not defined!\n"
                  "Performing a numerical convolution...")

            if isinstance(comp, Component):
                return fftconvolve(self.getComponent(x, params, dataset),
                                   comp.getComponent(x, params, dataset),
                                   mode='same')
            elif isinstance(comp, Model):
                return fftconvolve(self.getComponent(x, params, dataset),
                                   comp.getModel(x, params, dataset),
                                   mode='same')

        return res
