import numpy as np

from scipy.optimize import curve_fit

from nPDyn.dataTypes.TempRampType import DataTypeDecorator
from nPDyn.fit import fitENS_models as models



class Model(DataTypeDecorator):
    """ This class provides a model to fit q-dependent elastic
        signal measured during as a series,
        during a temperature ramp for instance.

        A linear model is used here.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = models.linearMSD
        self.params     = None
        self.paramsNames = ["MSD", "y0"]  # For plotting purpose




    def fit(self, p0=None, bounds=None, kwargs={}):
        """ Fitting procedure using Scipy *curve_fit*. """

        qIdxList = self.data.qIdx


        if not bounds:
            bounds = (-np.inf, np.inf)


        params = []
        for idx, temp in enumerate(self.data.X):

            Y = np.log(self.data.intensities[qIdxList, idx])

            if bounds is None:
                bounds = self.defaultBounds


            params.append(curve_fit(self.model,
                                    self.data.qVals[qIdxList]**2,
                                    Y,
                                    p0=p0,
                                    bounds=bounds,
                                    sigma=self.data.errors[qIdxList, idx], 
                                    **kwargs))



        self.params = params
