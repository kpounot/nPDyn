""" This module provides a template class to build models 
    that can be used to fit the data.

    The documentation of the :func:`setModel` function provides all
    the information on how to build and use the model, as well
    as the description of the available methods.

"""



class Model:
    """ This function can be used to define a model fully compatible
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
            dictionary of parameters with parameter names as 
            keys and parameters values as corresponding values.
        fixedParams : dict
            fixed parameters for the model, any parameter in
            this dictionary should be present in params.
        paramsErr : dict
            if available, this is used to store the errors 
            estimated by the fitting algorithm.
        eisfComponents : dict
            dictionary of model components as defined in the
            :func:`addComponent` function that account for the
            elastic incoherent structure factor (eisf).
        qisfComponents : dict
            dictionary of model components as defined in the
            :func:`addComponent` function that account for the
            elastic incoherent structure factor (eisf).
        bkgdComponents : dict
            dictionary of model components as defined in the
            :func:`addComponent` function that account for the
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


    def __init__(self, dataset, 
                 params={},
                 fixedParams={},
                 paramsErr={},
                 eisfComponents={},
                 qisfComponents={},
                 bkgdComponents={},
                 xVar=None,
                 fitMethod=None): 

        self.dataset = dataset

        self.params = params
        self.fixedParams = fixedParams
        self.paramsErr = paramsErr

        self.eisfComponents = eisfComponents
        self.qisfComponents = qisfComponents
        self.bkgdComponents = bkgdComponents

        self.xVar = None

        self.fitMethod = fitMethod


    def addComponent(self, name, f, params, compType='eisf'):
        """ This method defines a component to be used in the :class:`Model`.

            A component is defined by its unique name, a callable function
            that returns a specific lineshape and its parameters.

            The function add to the corresponding dictionary the component
            as a tuple entry of the form:
                
                - self.eisfComponents[name] = (f, params)
                - self.qisfComponents[name] = (f, params)
                - self.bkgdComponents[name] = (f, params)

            depending on the *compType* argument.


            Notes
            -----
            The :class:`Model` class will call the function using a vector
            *x* of shape (1, n) with n being the size of the :attr:`xVar` 
            variable (energies, temperatures or other)


            Parameters
            ----------
            name : str
                the name of the component
            f : callable
                a callable function of the form f(x, **params),
                where x is the x-axis values (energies, temperature)
                and params is a dictionary containing the parameters
                names and their associated values.
            params : dict
                dictionary containing parameters names and 
                their associated values.
            compType : {'eisf', 'qisf', 'bkgd'}
                can be 'eisf' if this component accounts for the 
                elastic incoherent structure factor or 'qisf' for
                quasi-elastic or 'bkgd' for background terms.

        """




    # --------------------------------------------------
    # convolution
    # --------------------------------------------------

    # --------------------------------------------------
    # fitting
    # --------------------------------------------------

    # --------------------------------------------------
    # accessors
    # --------------------------------------------------

    def getModel(self):
        """ Performs the assembly of the components and call
            the provided functions with their parameters to 
            compute the model.

            Returns
            -------
            The computed model in an array, for which the last dimension
            represents the :attr:`xVar` attribute.

        """

    def getComponents(self):
        """ Performs the calculation of the different components in the 
            model and returns the resulting arrays in a dictionary
            of with the components names as keys.


        
        """
