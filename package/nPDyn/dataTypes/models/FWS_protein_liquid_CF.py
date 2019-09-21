import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.FWSType import DataTypeDecorator
from nPDyn.fit.fitQENS_models import protein_liquid_analytic_voigt_CF as model



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

        The Scipy *curve_fit* routine is used.

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
        """ Global fit of data using Scipy *curve_fit* routine. """
        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)


        if p0 is None: #_Using default initial values
            p0 = [15, 30, 1]  
            p0 = p0 + [0.2 for i in self.data.qIdx] + [0.8 for i in self.data.qIdx]

        if bounds is None: #_Using default bounds
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = (0., [np.inf, np.inf, np.inf, maxI, 1.])


        #_D2O signal 
        D2OSignal = self.getD2OSignal()

        X = np.array( [self.data.X for i in self.data.qIdx] ).flatten()
        
        out = []
        for tIdx in range(self.data.intensities.shape[0]):
            print("\nFitting model for scan index %i" % tIdx, flush=True)

            if len(p0) == self.data.intensities.shape[0]:
                q_p0 = p0[tIdx][0]
            else:
                q_p0 = p0


            result = optimize.curve_fit( lambda x, *params: self.model(x, params, self, D2OSignal, None, tIdx),
                                         X,
                                         self.data.intensities[tIdx][self.data.qIdx].flatten(),
                                         q_p0,
                                         sigma=self.data.errors[tIdx][self.data.qIdx].flatten(),
                                         bounds=bounds,
                                         method='trf' )


            r  = "\nFinal result for scan %i:\n" % (tIdx)
            r += "    g0    = %.2f\n" % result[0][0]
            r += "    g1    = %.2f\n" % result[0][1]
            r += "    tau   = %.2f\n" % result[0][2]
            print(r)


            #_Creating a list with the same parameters for each q-values (makes plotting easier)
            for qIdx in self.data.qIdx:
                out.append(result)

        self.params = np.array( out ).reshape( (self.data.intensities.shape[0], self.data.qIdx.size) )    




    def qWiseFit(self, p0=None, bounds=None):
        """ q-wise data fit using Scipy *curve_fit* routine. """
        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)


        if p0 is None: #_Using default initial values
            p0 = [15, 30, 1, 0.1, 0.8] 

        if bounds is None: #_Using default bounds
            maxX = 2.5 * np.max( self.data.X )
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = (0., [np.inf, np.inf, np.inf, maxI, 1]) 


        #_D2O signal 
        D2OSignal = self.getD2OSignal()


        result = []
        for tIdx in range(self.data.intensities.shape[0]):
            print("\nFitting model for scan index %i" % tIdx, flush=True)

            for i, qIdx in enumerate(self.data.qIdx):

                if len(p0) == self.data.intensities.shape[0]:
                    q_p0 = p0[tIdx][i]
                else:
                    q_p0 = p0


                print("Fitting model for q index %i\n" % qIdx, flush=True)
                result.append(optimize.curve_fit( lambda x, *params: 
                                                        self.model(x, params, self, D2OSignal, qIdx, tIdx),
                                                  self.data.X,
                                                  self.data.intensities[tIdx][qIdx],
                                                  q_p0,
                                                  sigma=self.data.errors[tIdx][qIdx],
                                                  bounds=bounds,
                                                  method='trf' ))


                r  = "\nFinal result for scan %i and q-value %i:\n" % (tIdx, qIdx)
                r += "    g0    = %.2f\n" % result[-1][0][0]
                r += "    g1    = %.2f\n" % result[-1][0][1]
                r += "    tau   = %.2f\n" % result[-1][0][2]
                r += "    beta  = %.2f\n" % result[-1][0][3]
                r += "    a0    = %.2f\n" % result[-1][0][4]
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
        if len(self.params[0][0][0]) == 5:
            for tIdx in range(self.data.intensities.shape[0]):
                params.append(self.params[tIdx][qIdx][0])
        else:
            for tIdx in range(self.data.intensities.shape[0]):
                params.append(self.params[tIdx][qIdx][0][ [0,1,2,3+qIdx, 3+self.data.qIdx.size+qIdx] ])

        return np.array(params)



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        paramsErr = []
        if len(self.params[0][0][0]) == 5:
            for tIdx in range(self.data.intensities.shape[0]):
                params = self.params[tIdx][qIdx][1]
                params = np.sqrt( np.diag( params ) )
                paramsErr.append(params)
        else:
            for tIdx in range(self.data.intensities.shape[0]):
                params = self.params[tIdx][qIdx][1]
                params = np.sqrt( np.diag( params ) )
                params = params[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx] ]
                paramsErr.append(params)

        return np.array(paramsErr)




    def getWeights_and_lorWidths(self, qIdx):
        """ Accessor for weights/contribution factors and width of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian width
        outWeights   = []
        lorWidths = []
        if len(self.params[0][0][0]) == 5:
            for tIdx in range(self.data.intensities.shape[0]):
                weights = [self.params[tIdx][qIdx][0][4], 1 - self.params[tIdx][qIdx][0][4]]
                beta    = self.params[tIdx][qIdx][0][3]
                weights = np.array(weights) * beta

                outWeights.append(weights)

                lorWidths.append(self.params[tIdx][qIdx][0][0:2])
        else:
            for tIdx in range(self.data.intensities.shape[0]):
                weights     = [self.params[tIdx][qIdx][0][3+self.data.qIdx.size+qIdx], 
                                            1 - self.params[tIdx][qIdx][0][3+self.data.qIdx.size+qIdx]]
                beta        = self.params[tIdx][qIdx][0][3+qIdx]
                weights = np.array(weights) * beta

                outWeights.append(weights)

                lorWidths.append(self.params[tIdx][qIdx][0][0:2])
        


        labels      = ['Global', 'Internal']

        return np.array(outWeights), np.array(lorWidths), labels




    def getWeights_and_lorErrors(self, qIdx):
        """ Accessor for weights/contribution factors errors and width errors of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params[qIdx][1] )) for params in self.params ] )
        weightsErr  = []
        lorErr      = []
        if len(self.params[0][0][0]) == 5:
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
