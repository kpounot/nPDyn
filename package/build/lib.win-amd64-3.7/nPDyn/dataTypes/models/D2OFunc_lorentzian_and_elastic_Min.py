import numpy as np

from collections import namedtuple
from scipy import optimize

from ..baseType import BaseType, DataTypeDecorator
from ...fit.D2OFit import D2OFit_withElastic as model
from ...fit.D2O_params_from_IN6 import getD2Odata



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ["a0", "a1"] #_For plotting purpose

        self.volFraction= 0.95
        self.getD2OData = getD2Odata
        self.sD2O       = getD2Odata()
        self.disp       = True




    def qWiseFit(self, p0=None, bounds=None):
        if self.disp:
            print("\nUsing Scipy's minimize to fit data from file: %s" % self.fileName, flush=True)

        if not p0: #_Using default initial values
            p0 = np.array( [0.5, 0.5] ) 

        if not bounds: #_Using default bounds
            bounds = [(0., 1), (0., 1)]


        result = []
        for qIdx, qVal in enumerate(self.data.qVals):
            result.append( optimize.minimize( self.model, 
                                            p0,
                                            args=(self,), 
                                            bounds=bounds ) )

        self.params = result


    def getD2OContributionFactor(self):
        """ Returns the contribution factor of D2O lineshape in the model """

        aD2O = np.array([self.params[i].x[1] for i in self.data.qIdx])


        return aD2O



    def getD2OSignal(self):
        """ Computes D2O line shape for each q values.
            
            If a qIdx is given, returns D2O signal only for the corresponding q value. """

        params = np.array([self.params[idx].x for idx in self.data.qIdx])
        params[:,0] = 0

        D2OSignal = np.array( [ self.model(params[idx], self, idx, False) 
                                                                    for idx in self.data.qIdx ] )

        D2OSignal *= self.volFraction


        return D2OSignal



