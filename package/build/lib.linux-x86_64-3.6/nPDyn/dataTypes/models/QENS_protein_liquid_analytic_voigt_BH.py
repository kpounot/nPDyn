import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.QENSType import DataTypeDecorator
from nPDyn.fit.fitQENS_models import protein_liquid_analytic_voigt as model



class Model(DataTypeDecorator):
    """ This class provides a model for protein dynamics in liquid state for QENS data.

        The model (:py:func:`~fitQENS_models.protein_liquid_analytic_voigt`) is given by:

        .. math::

            S(q, \\omega) = R(q, \\omega ) \\otimes \\left[ \\beta ( a_{0} \\mathcal{L}_{\\gamma }
                                    + (1 - a_{0}) \\mathcal{L}_{\\Gamma } ) \\right]
                                    + \\beta_{D_{2}O} \\mathcal{L}_{D_{2}O}

        where, R is the resolution function, q is the scattering angle, :math:`\\omega` the energy offset, 
        :math:`\\mathcal{L}_{\\gamma}` is a single Lorentzian accounting for global diffusion motions,
        :math:`\\mathcal{L}_{\\Gamma}` is a Lorentzian of width obeying jump diffusion model as 
        decribed by Singwi and Sjölander [#]_ ,
        :math:`\\mathcal{L}_{D_{2}O}` is the :math:`D_{2}O` lineshape, :math:`a_{0}` acts as an EISF,
        and :math:`\\beta` and :math:`\\beta_{D_{2}O}` are scalars.

        The Scipy basinhopping routine is used.

        References:

        .. [#] https://journals.aps.org/pr/abstract/10.1103/PhysRev.119.863

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ['g0', 'g1', 'tau', 'beta', 'a0'] 
        self.BH_iter    = 20
        self.disp       = True



    def fit(self, p0=None, bounds=None):
        """ Global fit """

        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if p0 is None: #_Using default initial values
            p0 = [2, 20, 0.1] 
            p0 = p0 + [0.2 for i in self.data.qIdx] + [0.2 for i in self.data.qIdx]

        if bounds is None: #_Using default bounds
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = ( [(0.0, np.inf), (0.0, np.inf), (0, np.inf)]
                        + [(0., maxI) for i in self.data.qIdx] 
                        + [(0., 1) for i in self.data.qIdx] )


        #_D2O signal 
        D2OSignal = self.getD2OSignal()


        result = optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self, D2OSignal), 'bounds':bounds } )



        #_Creating a list with the same parameters for each q-values (makes code for plotting easier)
        out = []
        for qIdx in self.data.qIdx:
            out.append(result)

        self.params = out    






    def qWiseFit(self, p0=None, bounds=None):
        """ q-wise fit """

        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)

        if p0 is None: #_Using default initial values
            p0 = [0.8, 1, 10, 0.1, 0.5] 

        if bounds is None: #_Using default bounds
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = [(0., np.inf), (0., np.inf), (0., np.inf), (0., maxI), (0., 1)] 


        #_D2O signal 
        D2OSignal = self.getD2OSignal()


        result = []
        for i, qIdx in enumerate(self.data.qIdx):

            print("\nFitting model for q index %i\n" % qIdx, flush=True)
            result.append(optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self, D2OSignal, i), 'bounds':bounds } ))



        self.params = result




    def getModel(self, qIdx):
        """ Returns the fitted model for the given q value. """

        return self.model(self.getParams(qIdx), self, self.getD2OSignal(), qIdx, False)


    
#--------------------------------------------------
#_Parameters accessors
#--------------------------------------------------
    def getParams(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == 5:
            params = self.params[qIdx].x
        else:
            params = self.params[qIdx].x[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx] ]

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == 5:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
        else:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
            params = params[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx] ]

        return params


    def getEISFfactor(self, qIdx):
        """ Returns the contribution factor - usually called s0 - of the EISF. """

        return self.getParams(qIdx)[0]



    def getWeights_and_lorWidths(self, qIdx):
        """ Accessor for weights/contribution factors and width of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian width
        if len(self.params[0].x) == 5:
            weights     = [self.params[qIdx].x[4], 1 - self.params[qIdx].x[4]]
            beta        = self.params[qIdx].x[3]
        else:
            weights     = [self.params[qIdx].x[3+self.data.qIdx.size+qIdx], 
                                                    1 - self.params[qIdx].x[3+self.data.qIdx.size+qIdx]]
            beta        = self.params[qIdx].x[3+qIdx]
        
        weights = np.array(weights) * beta
        lorWidths   = self.params[qIdx].x[0:2]
        labels      = ['Global', 'Internal']

        return weights, lorWidths, labels




    def getWeights_and_lorErrors(self, qIdx):
        """ Accessor for weights/contribution factors errors and width errors of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params.lowest_optimization_result.hess_inv.todense())) 
                                                                                 for params in self.params ] )
        if len(self.params[0].x) == 5:
            weightsErr     = [errList[qIdx][4], errList[qIdx][4]]
        else:
            weightsErr     = [errList[qIdx][3+self.data.qIdx.size+qIdx], 
                                                    errList[qIdx][3+self.data.qIdx.size+qIdx]]
 
        lorErr = errList[qIdx,0:2]

        return weightsErr, lorErr




    def getBackground(self, qIdx):
        """ Accessor for background term, None for this model. """

        return None


    def get_betaSlice(self):
        """ For global fit, returns the slice corresponding to beta parameter(s) """

        return slice(3, 3+self.data.qIdx.size)



    def get_a0Slice(self):
        """ For global fit, returns the slice corresponding to a0 parameter(s) """

        return slice(3+self.data.qIdx.size, None)


    def getsubCurves(self, qIdx):
        """ Accessor for individual components of the fits for given q-value.

            :returns:
                
                - resolution function curve
                - global diffusion Lorentzian
                - internal dynamics Lorentzian
                - labels for the previous two Lorentzians

        """

        #_D2O signal 
        D2OSignal = self.getD2OSignal()

        resF, gLor, iLor = self.model(self.getParams(qIdx), self, D2OSignal, qIdx, False, 
                                      returnSubCurves=True)
        labels      = [r'$L_{\Gamma_{global}}(q, \omega)$', r'$L_{\Gamma_{internal}}(q, \omega)$']

        return resF[qIdx], gLor[qIdx], iLor[qIdx], labels



