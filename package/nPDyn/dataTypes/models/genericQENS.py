import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.QENSType import DataTypeDecorator



class Model(DataTypeDecorator):
    """ This class provides a model for protein dynamics in liquid state for QENS data.


    """

    def __init__(self, dataType, model):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ['$\Gamma$', '$\gamma$', '$\\tau$', '$\\beta$', '$a_0$'] 
        self.BH_iter    = 20
        self.disp       = True




    def fit(self, p0=None, bounds=None):
        """ Global fit """

        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if p0 is None: #_Using default initial values
            p0 = [5, 20, 0.1] 
            p0 = p0 + [0.2 for i in self.data.qIdx] + [0.2 for i in self.data.qIdx]

        if bounds is None: #_Using default bounds
            bounds = [(0, self.data.X.max()*3) for i in range(2)] + [(0., np.inf)]
            bounds += [(0, np.inf) for i in range(len(self.data.qIdx))]
            bounds += [(0, np.inf) for i in range(len(self.data.qIdx))]


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
            p0 = [5, 15, 10, 0.1, 0.5] 

        if bounds is None: #_Using default bounds
            bounds = [(0, self.data.X.max()*3) for i in range(2)] + [(0, np.inf) for i in range(3)]


        #_D2O signal 
        D2OSignal = self.getD2OSignal()


        result = []
        for i, qIdx in enumerate(self.data.qIdx):

            if i != 0:
                p0 = result[-1].x

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

        if len(self.params[0].x) == len(self.paramsNames):
            params = self.params[qIdx].x
        else:
            params = self.params[qIdx].x[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx] ]
            qVal = self.data.qVals[self.data.qIdx[qIdx]]**2
            params[0] = params[0] * qVal**2
            params[1] = params[1] * qVal**2 / (1 + params[1]*qVal**2*params[2])

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == len(self.paramsNames):
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
        if len(self.params[0].x) == len(self.paramsNames):
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
        if len(self.params[0].x) == len(self.paramsNames):
            weightsErr     = [errList[qIdx][4], errList[qIdx][4]]
        else:
            weightsErr     = [errList[qIdx][3+self.data.qIdx.size+qIdx], 
                                                    errList[qIdx][3+self.data.qIdx.size+qIdx]]
 
        lorErr = errList[qIdx,0:2]

        return weightsErr, lorErr




    def getBackground(self, qIdx):
        """ Accessor for background term, None for this model. """

        return None



    def getSubCurves(self, qIdx):
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


