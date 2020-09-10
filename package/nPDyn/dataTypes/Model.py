""" This module provides a template class to build models 
    that can be used to fit the data.

    The documentation of the :func:`setModel` function provides all
    the information on how to build and use the model, as well
    as the description of the available methods.

"""


from inspect import signature


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


    def __init__(self, 
                 dataset,
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


    def addComponent(self, comp, compType='eisf'):
        """ This method adds a :class:`Component` to be used 
            in the :class:`Model`.

            A component is defined by its unique name, 
            a :class:`Component` and a type of component.

            The function add to the corresponding dictionary the component
            as a tuple entry of the form:
                
                - self.eisfComponents[name] = :class:`Component` instance
                - self.qisfComponents[name] = :class:`Component` instance
                - self.bkgdComponents[name] = :class:`Component` instance

            depending on the *compType* argument.


            Notes
            -----
            The :class:`Model` class will call the component's function
            using a vector *x* of shape (1, n) with n being the size of 
            the :attr:`xVar` variable (energies, temperatures or other).


            Parameters
            ----------
            comp : :class:`Component`
                a :class:`Component` instance 
            compType : {'eisf', 'qisf', 'bkgd'}
                can be 'eisf' if this component accounts for the 
                elastic incoherent structure factor or 'qisf' for
                quasi-elastic or 'bkgd' for background terms.

        """



    # --------------------------------------------------
    # fitting
    # --------------------------------------------------

    # --------------------------------------------------
    # accessors
    # --------------------------------------------------
    def getModel(self, convolve=True):
        """ Performs the assembly of the components and call
            the provided functions with their parameters to 
            compute the model.


            Returns
            -------
            The computed model in an array, for which the last dimension
            represents the :attr:`xVar` attribute.

        """



    def getComponents(self, convolve=True):
        """ Performs the calculation of the different components in the 
            model and returns the resulting arrays in a dictionary
            of with the components names as keys.

        """





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
        between a Gaussian and a Lorentzian.).

        Parameters
        ----------
        name : str
            name of the component (used to define the rules for 
            analytical convolutions)
        convolutions : dict
            dictionary in which the keys refers to the name of another
            component to be convoluted with and the associated value is
            a function that takes 
    """

    def __init__(self, 
                 name,
                 convolutions={}):

        self.convolutions = convolutions

    def getComponent(self, x, dataset):
        """ Accessor to the component lineshape.

            The function takes 

