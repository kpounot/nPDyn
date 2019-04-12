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
        self.paramsNames = ["a0", "scale"] #_For plotting purpose

        self.volFraction= 0.95
        self.getD2OData = getD2Odata
        self.sD2O       = getD2Odata()
        self.disp       = True




    def qWiseFit(self, p0=None, bounds=None):
        if self.disp:
            print("\nUsing Scipy's minimize to fit data from file: %s" % self.fileName, flush=True)

        if not p0: #_Using default initial values
            p0 = np.array( [0.2, 0.02] ) 

        maxI = 5 * np.max( self.data.intensities )
        if not bounds: #_Using default bounds
            bounds = [(0., 1), (0., maxI)]


        result = []
        for qIdx, qVal in enumerate(self.data.qVals):
            result.append( optimize.minimize( self.model, 
                                            p0,
                                            args=(self,), 
                                            bounds=bounds ) )

        self.params = result




