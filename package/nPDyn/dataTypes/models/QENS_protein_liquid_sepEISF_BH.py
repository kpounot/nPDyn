import numpy as np

from collections import namedtuple
from scipy import optimize

from ..QENSType import DataTypeDecorator



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.params     = None
        self.paramsNames = ['g0', 'g1', 'tau', 'betaD2O', 'beta', 'a0', 'eisf'] 
        self.BH_iter    = 50
        self.disp       = True



    def fit(self, p0=None, bounds=None):
        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if not p0: #_Using default initial values
            p0 = [2, 20, 0.1] + [0.8 for i in self.data.qIdx] 
            p0 = p0 + [0.2 for i in self.data.qIdx] + [0.2 for i in self.data.qIdx] + [0.05]

        if not bounds: #_Using default bounds
            maxX = 2.5 * np.max( self.data.X )
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = ( [(0.5, maxX), (0.5, maxX), (0, 10)]
                        + [(0.,1) for i in self.data.qIdx]
                        + [(0., maxI) for i in self.data.qIdx] 
                        + [(0., 1) for i in self.data.qIdx] + [(0.,1)])


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
            p0 = [0.8, 1, 10, 0.8, 0.1, 0.5, 0.05] 

        if not bounds: #_Using default bounds
            maxX = 2.5 * np.max( self.data.X )
            maxI = 1.5 * np.max( self.data.intensities )
            bounds = [(0.5, maxX), (0.5, maxX), (0., 100), (0., 1), (0., maxI), (0., 1), (0., 1)] 


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



    
#--------------------------------------------------
#_Parameters accessors
#--------------------------------------------------
    def getParams(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == 7:
            params = self.params[qIdx].x
        else:
            params = self.params[qIdx].x[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx,
                                                            3+2*self.data.qIdx.size+qIdx] ]

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == 7:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
        else:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
            params = params[ [0,1,2,3+qIdx,3+self.data.qIdx.size+qIdx,3+2*self.data.qIdx.size+qIdx] ]

        return params




    def getWeights_and_lorWidths(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian width
        if len(self.params[0].x) == 7:
            weights     = [self.params[qIdx].x[5], 1 - self.params[qIdx].x[5]]
            beta        = self.params[qIdx].x[4]
        else:
            weights     = [self.params[qIdx].x[3+2*self.data.qIdx.size+qIdx], 
                                                    1 - self.params[qIdx].x[3+2*self.data.qIdx.size+qIdx]]
            beta        = self.params[qIdx].x[3+self.data.qIdx.size+qIdx]
        
        weights = np.array(weights) * beta
        lorWidths   = self.params[qIdx].x[0:2]
        labels      = ['Global', 'Internal']

        return weights, lorWidths, labels




    def getWeights_and_lorErrors(self, qIdx):
        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params.lowest_optimization_result.hess_inv.todense())) 
                                                                                 for params in self.params ] )
        if len(self.params[0].x) == 7:
            weightsErr     = [errList[qIdx][5], errList[qIdx][5]]
        else:
            weightsErr     = [errList[qIdx][3+2*self.data.qIdx.size+qIdx], 
                                                    errList[qIdx][3+2*self.data.qIdx.size+qIdx]]
 
        lorErr = errList[qIdx,0:2]

        return weightsErr, lorErr




    def getBackground(self, qIdx):

        return None



#--------------------------------------------------
#_Parameters accessors
#--------------------------------------------------
    def model(self, params, dataset, D2OSignal=None, qIdx=None, returnCost=True, scanIdx=slice(0,None)):
        """ Fitting function for protein in liquid D2O environment.

            Input:  params      -> parameters for the model (described below), usually given by scipy's routines
                    self        -> self namedtuple containing x-axis values, intensities, errors,...
                    D2OSignal   -> D2OSignal, fitted or not, to be used in the model (optional, is obtained 
                                    from self if None, but this make the whole fitting procedure much slower)
                    qIdx        -> index of the current q-value
                                   if None, a global fit is performed over all q-values
                    returnCost  -> if True, return the standard deviation of the model to experimental data
                                   if False, return only the model 
                    scanIdx     -> only for FWS (or QENS) data series, index of the scan being fitted """


        g0      = params[0]     #_global diffusion linewidth
        g1      = params[1]     #_internal diffusion linewidth
        tau     = params[2]     #_Residence time for jump diffusion model
        betaD2O = params[3]     #_Contribution factor for D2O
        beta    = params[4]     #_contribution factor of protein, same shape as # of q-values used
        a0      = params[5]     #_contribution factor of EISF
        eisf    = params[-1]

        if params.size != 7:
            betaD2O = params[3:3+self.data.qIdx.size][:,np.newaxis]
            beta    = params[3+self.data.qIdx.size:3+2*self.data.qIdx.size][:,np.newaxis]
            a0      = params[3+2*self.data.qIdx.size:][:,np.newaxis]


        X = self.data.X
        qVals = self.data.qVals[self.data.qIdx, np.newaxis] #_Reshape to a column vector
        normF = np.array( [ self.resData.params[qIdx][0][0] for qIdx in self.data.qIdx ] )
        resFunc = self.getResFunc()



        if D2OSignal is None:
            D2OSignal =  betaD2O * self.getD2OSignal()



        #_Model
        model = np.zeros((qVals.size, 2, X.size)) #_Initialize the final array

        
        if qIdx == None:
            g0 = qVals**2 * g0
            g1 = g1 * qVals**2 / (1 + g1 * qVals**2 * tau)

        model[:,0] += a0 * g0 / (X**2 + g0**2) * (1 / np.pi)
        model[:,1] += (1 - a0) * (g0+g1) / (X**2 + (g0+g1)**2) * (1 / np.pi) 

        model = beta * np.sum( model, axis=1 ) #_Summing the two lorentzians contributions


        #_Performs the convolution for each q-value, instrumental background should be contained in D2O signal
        for idx in range(model.shape[0]):
            model[idx] = eisf*resFunc[idx] + np.convolve(model[idx], resFunc[idx], mode='same') + D2OSignal[idx]
        

        cost = np.sum( (self.data.intensities[scanIdx][self.data.qIdx] - model)**2 
                                            / self.data.errors[scanIdx][self.data.qIdx]**2, axis=1) 
                                       

        if qIdx is not None:
            cost    = cost[qIdx]
            model   = model[qIdx]
        else:
            cost = np.sum(cost)



        if returnCost:
            return cost
        else:
            return model


