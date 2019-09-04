import numpy as np

from collections import namedtuple
from scipy import optimize

from ..QENSType import DataTypeDecorator
from ...fit.fitQENS_models import protein_liquid_analytic_voigt_CF as model



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ['g0', 'g1', 'tau', 'beta', 'a0'] 
        self.BH_iter    = 20
        self.disp       = True


    def fit(self, p0=None, bounds=None):
        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if p0 is None: #_Using default initial values
            p0 = [2, 20, 0.1] 
            p0 = p0 + [0.2 for i in self.data.qIdx] + [0.2 for i in self.data.qIdx]

        if bounds is None: #_Using default bounds
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = ( [(0.6, np.inf), (0.6, np.inf), (0, np.inf)]
                        + [(0., maxI) for i in self.data.qIdx] 
                        + [(0., 1) for i in self.data.qIdx] )


        #_D2O signal 
        D2OSignal = self.getD2OSignal()

        X = np.array( [self.data.X for i in self.data.qIdx] ).flatten()

        result = optimize.curve_fit( lambda x, *params: self.model(x, params, self, D2OSignal),
                                     X,
                                     self.data.intensities.flatten(),
                                     p0,
                                     method='trf' )


        #_Creating a list with the same parameters for each q-values (makes code for plotting easier)
        out = []
        for qIdx in self.data.qIdx:
            out.append(result)

        self.params = out    






    def qWiseFit(self, p0=None, bounds=None):
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

            if len(p0) == self.data.qIdx.size:
                q_p0 = p0[qIdx]
            else:
                q_p0 = p0


            print("\nFitting model for q index %i\n" % qIdx, flush=True)
            result.append(optimize.curve_fit( lambda x, *params: 
                                                    self.model(x, params, self, D2OSignal, qIdx),
                                              self.data.X,
                                              self.data.intensities[qIdx],
                                              q_p0,
                                              method='trf' ))




        self.params = result




    def getModel(self, qIdx):
        """ Returns the fitted model for the given q value. """

        return self.model(self.data.X, self.getParams(qIdx), self, self.getD2OSignal(), qIdx, None)

    
#--------------------------------------------------
#_Parameters accessors
#--------------------------------------------------
    def getParams(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0][0]) == 5:
            params = self.params[qIdx][0]
        else:
            params = self.params[qIdx][0][ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx] ]

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0][0]) == 5:
            params = self.params[qIdx][1]
            params = np.sqrt( np.diag( params ) )
        else:
            params = self.params[qIdx][1]
            params = np.sqrt( np.diag( params ) )
            params = params[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx] ]

        return params


    def getEISFfactor(self, qIdx):
        """ Returns the contribution factor - usually called s0 - of the EISF. """

        return self.getParams(qIdx)[0]



    def getWeights_and_lorWidths(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian width
        if len(self.params[0][0]) == 5:
            weights     = [self.params[qIdx][0][4], 1 - self.params[qIdx][0][4]]
            beta        = self.params[qIdx][0][3]
        else:
            weights     = [self.params[qIdx][0][3+self.data.qIdx.size+qIdx], 
                                                    1 - self.params[qIdx][0][3+self.data.qIdx.size+qIdx]]
            beta        = self.params[qIdx][0][3+qIdx]
        
        weights = np.array(weights) * beta
        lorWidths   = self.params[qIdx][0][0:2]
        labels      = ['Global', 'Internal']

        return weights, lorWidths, labels




    def getWeights_and_lorErrors(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params[1])) for params in self.params ] )

        if len(self.params[0][0]) == 5:
            weightsErr     = [errList[qIdx][4], errList[qIdx][4]]
        else:
            weightsErr     = [errList[qIdx][3+self.data.qIdx.size+qIdx], 
                                                    errList[qIdx][3+self.data.qIdx.size+qIdx]]
 
        lorErr = errList[qIdx,0:2]

        return weightsErr, lorErr




    def getBackground(self, qIdx):

        return None



    def getsubCurves(self, qIdx):

        #_D2O signal 
        D2OSignal = self.getD2OSignal()

        resF, gLor, iLor = self.model(self.data.X, self.getParams(qIdx), self, D2OSignal, qIdx, False, 
                                      returnSubCurves=True)
        labels      = [r'$L_{\Gamma_{global}}(q, \omega)$', r'$L_{\Gamma_{internal}}(q, \omega)$']

        return resF[qIdx], gLor[qIdx], iLor[qIdx], labels




