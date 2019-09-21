import numpy as np

from collections import namedtuple

from scipy.optimize import curve_fit

from nPDyn.dataTypes.TempRampType import DataTypeDecorator
from nPDyn.fit import fitENS_models as models 



class Model(DataTypeDecorator):
    """ This class provides a model to fit q-dependent elastic signal measured during as a series, 
        during a temperature ramp for instance. 

        A simple gaussian is used here.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = models.gaussian
        self.params     = None
        self.paramsNames = ["scale", "MSD"] #_For plotting purpose


        self.defaultBounds = (0., [10000., 15.])




    def fit(self, p0=None, bounds=None):
        """ Fitting procedure using Scipy *curve_fit*. """

        if not bounds:
            bounds = self.defaultBounds

        if not p0:
            p0 = [1.0, 0.0]

        qIdxList = self.data.qIdx
        
        params = []
        for idx, temp in enumerate(self.data.X):

            if idx > 0:
                p0 = params[idx-1][0]

            params.append( curve_fit(  self.model, 
                                       self.data.qVals[qIdxList],
                                       self.data.intensities[qIdxList,idx],
                                       p0=p0,
                                       bounds=bounds,
                                       sigma=self.data.errors[qIdxList,idx],
                                       method='trf'))



        self.params = params
