import numpy as np

from scipy.optimize import curve_fit

from nPDyn.dataTypes.TempRampType import DataTypeDecorator
from nPDyn.fit import fitENS_models as models



class Model(DataTypeDecorator):
    """ This class provides a model to fit q-dependent elastic
        signal measured during as a series,
        during a temperature ramp for instance.

        A simple gaussian is used here.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = models.gaussian
        self.params     = None
        self.paramsNames = ["MSD", "shift"]  # For plotting purpose




    def fit(self, p0=None, bounds=None, kwargs={}):
        """ Fitting procedure using Scipy *curve_fit*. """

        qIdxList = self.data.qIdx


        if not bounds:
            bounds = (-np.inf, np.inf)


        params = []
        for idx, temp in enumerate(self.data.X):

            if bounds is None:
                bounds = self.defaultBounds


            params.append(curve_fit(self.model,
                                    self.data.qVals[qIdxList],
                                    self.data.intensities[qIdxList, idx],
                                    p0=p0,
                                    bounds=bounds,
                                    sigma=self.data.errors[qIdxList, idx],
                                    **kwargs))



        self.params = params
