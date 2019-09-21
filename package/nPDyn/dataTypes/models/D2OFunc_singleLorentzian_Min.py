import numpy as np

from collections import namedtuple
from scipy import optimize
from scipy.signal import fftconvolve, convolve

from nPDyn.dataTypes.baseType import BaseType, DataTypeDecorator
from nPDyn.fit.D2O_params_from_IN6 import getD2Odata



class Model(DataTypeDecorator):
    """ This class provides a model for :math:`D_{2}O` signal fitting. 

        The model equation is given by:

        .. math::

           S(q, \\omega ) = R(q, \\omega ) \\otimes \\left[  
                        a_{0} \\frac{ \\Gamma_{D_{2}O}(q) }{ \\pi ( \\omega^{2} + \\Gamma_{D_{2}O}^{2}(q) ) } 
                        \\right] 

        where *q* is the scattering angle, :math:`\\omega` the energy offset, :math:`a_{0}` is a scalar,
        and R the resolution function.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ["a0"] #_For plotting purpose

        self.volFraction= 0.95
        self.getD2OData = getD2Odata
        self.sD2O       = getD2Odata()
        self.disp       = True


    def fit(self, p0=None, bounds=None):
        if self.disp:
            print("\nUsing Scipy's minimize to fit data from file: %s" % self.fileName, flush=True)

        if not p0: #_Using default initial values
            p0 = np.array( [0.1] ) 

        if not bounds: #_Using default bounds
            bounds = [(0., 10)]


        result = []
        for qIdx, qVal in enumerate(self.data.qVals):
            result.append( optimize.minimize( self.model, 
                                            p0,
                                            args=(self,), 
                                            bounds=bounds ) )

        self.params = result



    def getD2OContributionFactor(self):
        """ Returns the contribution factor of D2O lineshape in the model """

        aD2O = np.array([self.params[i].x[0] for i in self.data.qIdx])

        return aD2O



    def getD2OSignal(self, qIdx=None):
        """ Computes D2O line shape for each q values.
            
            If a qIdx is given, returns D2O signal only for the corresponding q value. """

        D2OSignal = np.array( [ self.model(self.params[idx].x, self, idx, False) 
                                                                    for idx in self.data.qIdx ] )

        D2OSignal *= self.volFraction


        return D2OSignal



#--------------------------------------------------
# model
#--------------------------------------------------
def model(params, dataset, qIdx=None, returnCost=True):
    """ This class can be used to fit data from D2O - q-wise or globally - using a single lorentzian and
        D2O data from IN6 at the ILL. 

        :arg params:     parameters for the model (described below), usually given by scipy's routines
        :arg dataSet:    dataSet namedtuple containing x-axis values, intensities, errors,...
        :arg qIdx:       index of the current q-value
                            if None, a global fit is performed over all q-values
        :arg returnCost: if True, return the standard deviation of the model to experimental data
                            if False, return only the model 

    """


    a0 = params[0]  #_contribution factor of lorentzian


    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector
    resFunc = dataset.getResFunc()
    resBkgd = dataset.getResBkgd()



    #_Computes D2O linewidth for each q-values
    gD2O = np.array([dataset.sD2O(dataset.data.temp, qVal) for qVal in dataset.data.qVals])

    model = np.zeros((qVals.size, X.size)) #_Initialize the final array

    model = model + gD2O

    model = a0 * model / (np.pi * (X**2 + model**2)) #_Computes the lorentzian


    #_Performs the convolution for each q-value
    for idx in range(model.shape[0]):
        model[idx] = np.convolve(model[idx], resFunc[idx], mode='same') + resBkgd[idx]


    cost = np.sum( ( 1 + np.log((dataset.data.intensities[dataset.data.qIdx] - model)**2) ) 
                                        / dataset.data.errors[dataset.data.qIdx]**2, axis=1) 


    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)


    if returnCost:
        return cost
    else:
        return model








