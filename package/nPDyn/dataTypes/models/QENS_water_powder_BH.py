import numpy as np

from collections import namedtuple
from scipy import optimize

from ..QENSType import DataTypeDecorator
from ...fit.fitQENS_models import water_powder as model



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model          = model
        self.params         = None
        self.paramsNames    = ['s0', 'sr', 'st', 'rotational', 'translational', 'msd', 'bkgd'] 
        self.BH_iter        = 50
        self.disp           = True


    def fit(self, p0=None, bounds=None):
        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.6, 0.2, 0.2, 2, 15, 1] + [0.001 for i in range(len(self.data.qIdx))]

        if not bounds: #_Using default bounds
            minData = 1.5 * np.min( self.data.intensities ) #_To restrict background below experimental data

            bounds = [(0., 1), (0., 1), (0., 1), (0., 60), (0., 60), (0., 10)]
            bounds += [(0., minData) for i in range(len(self.data.qIdx))]



        result = optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        interval=25,
                                        disp=self.disp,
                                        minimizer_kwargs={  'args':(self,), 
                                                            'bounds':bounds,
                                                            'options': {'maxcor': 100,
                                                                        'maxfun': 200000}})



        #_Creating a list with the same parameters for each q-values (makes code for plotting easier)
        out = []
        for qIdx in self.data.qIdx:
            out.append(result)

        self.params = out    




    def qWiseFit(self, p0=None, bounds=None):
        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.6, 0.2, 0.2, 5, 5, 1, 0.001]

        if not bounds: #_Using default bounds
            minData = np.min( self.data.intensities ) #_To restrict background below experimental data

            bounds = [(0., 1), (0., 1), (0., 1), (0., 60), (0., 60), (0., 10), (0., minData)] 


        result = []
        for i, qIdx in enumerate(self.data.qIdx):
            print("\nFitting model for q index %i\n" % qIdx, flush=True)
            result.append(optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self, i), 
                                                           'bounds':bounds,
                                                            'options': {'maxcor': 100,
                                                                        'maxfun': 200000}}) )
                                                           


        self.params = result



    def getWeights_and_lorWidths(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian width

        rotational = np.sum( [l*(l+1) * self.params[qIdx].x[3] for l in range(1,3)] )

        weights     = self.params[qIdx].x[1:3]
        lorWidths   = [rotational, self.params[qIdx].x[4] * self.data.qVals[qIdx]**2]
        labels      = ['Rotational', 'Translational']

        return weights, lorWidths, labels



    def getWeights_and_lorErrors(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params.lowest_optimization_result.hess_inv.todense())) 
                                                                                 for params in self.params ] )

        weightsErr = errList[qIdx,1:3]
        lorErr = errList[qIdx,3:5]

        return weightsErr, lorErr




    def getBackground(self, qIdx):

        if len(self.params[0].x) == 7:
            return self.params[qIdx].x[6]
        else:
            return self.params[qIdx].x[6+qIdx]

