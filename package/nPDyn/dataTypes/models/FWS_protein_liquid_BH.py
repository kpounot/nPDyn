import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.FWSType import DataTypeDecorator
from nPDyn.fit.fitQENS_models import protein_liquid_analytic_voigt as model



class Model(DataTypeDecorator):
    """ This class provides a model for protein dynamics in liquid state for fixed-window scans data.

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

        self.model       = model
        self.params      = None
        self.paramsNames = ['$D_s$', '$D_i$', '$\\tau$', '$\\beta$', '$a_0$'] 
        self.BH_iter     = 20
        self.disp        = True



    def fit(self, p0=None, bounds=None):
        """ Global fit making use of Scipy basinhopping routine. """

        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)


        if not bounds: 
            bounds = [(0, np.inf) for i in range(3)]
            bounds += [(0, np.inf) for i in range(len(self.data.qIdx))]
            bounds += [(0, 1) for i in range(len(self.data.qIdx))]


        #_D2O signal 
        D2OSignal = self.getD2OSignal()
        
        out = []
        for tIdx in range(self.data.intensities.shape[0]):
            print("\nFitting model for scan index %i" % tIdx, flush=True)

            if p0 is None: #_Using default initial values
                p0  = [15, 30, 1]  
                p0  = p0 + [np.max(self.data.intensities[tIdx,i]) for i in self.data.qIdx] 
                p0 += [0.5 for i in self.data.qIdx]

            if tIdx > 0:
                p0 = out[-1].x

            result = optimize.basinhopping( self.model, 
                                            p0,
                                            niter = self.BH_iter,
                                            niter_success = 0.5*self.BH_iter,
                                            disp=self.disp,
                                            minimizer_kwargs={ 'args':(self, D2OSignal, None, True, tIdx), 
                                                               'bounds':bounds } )



            r  = "\nFinal result for scan %i:\n" % (tIdx)
            r += "    g0    = %.2f\n" % result.x[0]
            r += "    g1    = %.2f\n" % result.x[1]
            r += "    tau   = %.2f\n" % result.x[2]
            print(r)


            #_Creating a list with the same parameters for each q-values (makes code for plotting easier)
            for qIdx in self.data.qIdx:
                out.append(result)

        self.params = np.array( out ).reshape( (self.data.intensities.shape[0], self.data.qIdx.size) )    




    def qWiseFit(self, p0=None, bounds=None):
        """ q-wise fit making use of Scipy basinhopping routine. """

        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)


        if not p0: #_Using default initial values
            p0 = [15, 30, 1, 0.5, 0.5] 

        if not bounds: 
            bounds = [(0, np.inf) for i in range(4)] + [(0., 1)]


        #_D2O signal 
        D2OSignal = self.getD2OSignal()


        result = []
        for tIdx in range(self.data.intensities.shape[0]):
            print("\nFitting model for scan index %i" % tIdx, flush=True)

            for i, qIdx in enumerate(self.data.qIdx):

                if tIdx > 0 and qIdx > 0:
                    p0 = result[self.data.qIdx.size*(tIdx-1) + qIdx - 1].x

                print("Fitting model for q index %i\n" % qIdx, flush=True)
                result.append(optimize.basinhopping( self.model, 
                                            p0,
                                            niter = self.BH_iter,
                                            niter_success = 0.5*self.BH_iter,
                                            disp=self.disp,
                                            minimizer_kwargs={ 'args':(self, D2OSignal, i, True, tIdx), 
                                                               'bounds':bounds } ))


                r  = "\nFinal result for scan %i and q-value %i:\n" % (tIdx, qIdx)
                r += "    g0    = %.2f\n" % result[-1].x[0]
                r += "    g1    = %.2f\n" % result[-1].x[1]
                r += "    tau   = %.2f\n" % result[-1].x[2]
                r += "    beta  = %.2f\n" % result[-1].x[3]
                r += "    a0    = %.2f\n" % result[-1].x[4]
                print(r)



        self.params = np.array( result ).reshape( (self.data.intensities.shape[0], self.data.qIdx.size) )


    def getModel(self, scanIdx, qIdx):
        """ Returns the fitted model for the given q value. """

        return self.model(self.getParams(qIdx)[scanIdx], self, self.getD2OSignal(), 
                                                                        qIdx, False, scanIdx)



#--------------------------------------------------
#_Parameters accessors
#--------------------------------------------------
    def getParams(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        params = []
        if len(self.params[0][0].x) == 5:
            for tIdx in range(self.data.intensities.shape[0]):
                params.append(self.params[tIdx][qIdx].x)
        else:
            for tIdx in range(self.data.intensities.shape[0]):
                params.append(self.params[tIdx][qIdx].x[ [0,1,2,3+qIdx, 3+self.data.qIdx.size+qIdx] ])

        return np.array(params)



    def getParamsErrors(self, qIdx):
        """ Accessor for parameter errors of the model for the given q value """

        paramsErr = []
        if len(self.params[0][0].x) == 5:
            for tIdx in range(self.data.intensities.shape[0]):
                params = self.params[tIdx][qIdx].lowest_optimization_result.hess_inv.todense()
                params = np.sqrt( np.diag( params ) )
                paramsErr.append(params)
        else:
            for tIdx in range(self.data.intensities.shape[0]):
                params = self.params[tIdx][qIdx].lowest_optimization_result.hess_inv.todense()
                params = np.sqrt( np.diag( params ) )
                params = params[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx] ]
                paramsErr.append(params)

        return np.array(paramsErr)




    def getWeights_and_lorWidths(self, qIdx):
        """ Accessor for weights/contribution factors and width of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian width
        outWeights   = []
        lorWidths = []
        if len(self.params[0][0].x) == 5:
            for tIdx in range(self.data.intensities.shape[0]):
                weights = [self.params[tIdx][qIdx].x[4], 1 - self.params[tIdx][qIdx].x[4]]
                beta    = self.params[tIdx][qIdx].x[3]
                weights = np.array(weights) * beta

                outWeights.append(weights)

                lorWidths.append(self.params[tIdx][qIdx].x[0:2])
        else:
            for tIdx in range(self.data.intensities.shape[0]):
                weights     = [self.params[tIdx][qIdx].x[3+self.data.qIdx.size+qIdx], 
                                            1 - self.params[tIdx][qIdx].x[3+self.data.qIdx.size+qIdx]]
                beta        = self.params[tIdx][qIdx].x[3+qIdx]
                weights = np.array(weights) * beta

                outWeights.append(weights)

                lorWidths.append(self.params[tIdx][qIdx].x[0:2])
        


        labels      = ['Global', 'Internal']

        return np.array(outWeights), np.array(lorWidths), labels




    def getWeights_and_lorErrors(self, qIdx):
        """ Accessor for weights/contribution factors errors and width errors of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params[qIdx].lowest_optimization_result.hess_inv.todense())) 
                                                                                 for params in self.params ] )
        weightsErr  = []
        lorErr      = []
        if len(self.params[0][0].x) == 5:
            for tIdx in range(self.data.intensities.shape[0]):
                weightsErr.append( [errList[tIdx][4], errList[tIdx][4]] )
                lorErr.append( errList[tIdx][0:2] )
        else:
            for tIdx in range(self.data.intensities.shape[0]):
                weightsErr.append( [errList[tIdx][3+self.data.qIdx.size+qIdx], 
                                                        errList[tIdx][3+self.data.qIdx.size+qIdx]] )
                lorErr.append( errList[tIdx][0:2] )
 

        return np.array(weightsErr), np.array(lorErr)




    def getBackground(self, qIdx):
        """ Accessor for background term, None for this model. """

        return None



    def get_betaSlice(self):
        """ For global fit, returns the slice corresponding to beta parameter(s) """

        return slice(3, 3+self.data.qIdx.size)



    def get_a0Slice(self):
        """ For global fit, returns the slice corresponding to a0 parameter(s) """

        return slice(3+self.data.qIdx.size, None)


