import numpy as np

from collections import namedtuple

from scipy.optimize import curve_fit

from ..TempRampType import DataTypeDecorator
from ...fit import fitENS_models as models 



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = models.gaussian
        self.params     = None
        self.paramsNames = ["scale", "MSD"] #_For plotting purpose


        self.defaultBounds = (0., [10., 4.])




    def fit(self, p0=None, bounds=None):

        if not bounds:
            bounds = self.defaultBounds

        if not p0:
            p0 = [1.0, 0.0]

        qIdxList = self.data.qIdx
        
        params = []
        for idx, temp in enumerate(self.data.X):

            if idx != 0:
                p0 = params[idx-1][0]

            params.append( curve_fit(  self.model, 
                                       self.data.qVals[qIdxList],
                                       self.data.intensities[qIdxList,idx],
                                       p0=p0,
                                       bounds=bounds,
                                       sigma=self.data.errors[qIdxList,idx] ))



        self.params = params
