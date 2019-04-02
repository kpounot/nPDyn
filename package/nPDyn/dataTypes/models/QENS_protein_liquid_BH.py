import numpy as np

from collections import namedtuple
from scipy import optimize

from ..QENSType import DataTypeDecorator
from ...fit.fitQENS_models import protein_liquid as model



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ['beta', 'g0', 'g1', 'tau'] 
        self.BH_iter    = 50
        self.disp       = True



    def fit(self, p0=None, bounds=None):
        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.8, 1, 10, 0.1] + [0.5 for i in self.data.qIdx]

        if not bounds: #_Using default bounds
            maxX = 2.5 * np.max( self.data.X )
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = [(0., maxI), (0.2, maxX), (0.2, maxX), (0, 100)] + [(0., 1) for i in self.data.qIdx] 



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
        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.8, 1, 10, 0.1, 0.5] 

        if not bounds: #_Using default bounds
            maxX = 2.5 * np.max( self.data.X )
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = [(0., maxI), (0.2, maxX), (0.2, maxX), (0., 100), (0., 1)] 


        #_D2O signal 
        D2OSignal = self.getD2OSignal()


        result = []
        for i, qIdx in enumerate(self.data.qIdx):

            if i != 0: #_Use the result from the previous q-value as starting parameters
                p0 = result[-1].x

            print("\nFitting model for q index %i\n" % qIdx, flush=True)
            result.append(optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self, D2OSignal, i), 'bounds':bounds } ))



        self.params = result



    def getWeights_and_lorWidths(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian width
        if len(self.params[0].x) == 5:
            weights     = [self.params[qIdx].x[4], 1 - self.params[qIdx].x[4]]
        else:
            weights     = [self.params[qIdx].x[4+qIdx], 1 - self.params[qIdx].x[4+qIdx]]
        
        weights = np.array(weights) * self.params[qIdx].x[0]
        lorWidths   = self.params[qIdx].x[1:3]
        labels      = ['Global', 'Internal']

        return weights, lorWidths, labels




    def getWeights_and_lorErrors(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params.lowest_optimization_result.hess_inv.todense())) 
                                                                                 for params in self.params ] )

        if len(self.params[0].x) == 5:
            weightsErr = [ errList[qIdx,4], errList[qIdx,4] ]
        else:
            weightsErr = [ errList[qIdx,4+qIdx], errList[qIdx,4+qIdx] ]
        
        lorErr = errList[qIdx,1:3]

        return weightsErr, lorErr




    def getBackground(self, qIdx):

        return None

