import numpy as np

from scipy.optimize import curve_fit

from nPDyn.dataTypes.TempRampType import DataTypeDecorator
from nPDyn.fit import fitENS_models as models



class Model(DataTypeDecorator):
    """ This class provides a model to fit q-dependent elastic signal
        measured during as a series, during a temperature ramp for instance.

        The model used is given by [#]_ :

        .. math::

            S(q, \\omega = 0) = e^{ -\\frac{q^{2} \\langle u^{2}
                                \\rangle }{6} }
                                \\left[ 1 + e^{ - \\frac{q^{4}
                                \\sigma^{2}}{72} } \\right]

        where q is the scattering angle, :math:`\\omega` the energy offset,
        :math:`\\langle u^{2} \\rangle` the mean-squared displacement,
        and :math:`\\sigma` the second moment of mean-squared displacement
        distribution.

        References:

        .. [#] http://doi.org/10.1021/jp2102868

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = models.q4_corrected_gaussian
        self.params     = None
        self.paramsNames = ["MSD", "sigma", "shift"]





    def fit(self, p0=None, bounds=None, kwargs={}):
        """ Fitting procedure that makes use of Scipy *curve_fit*. """

        if not bounds:
            bounds = (-np.inf, np.inf)

        qIdxList = self.data.qIdx

        params = []
        for idx, temp in enumerate(self.data.X):

            if idx != 0:
                p0 = params[idx - 1][0]

            params.append(curve_fit(self.model,
                                    self.data.qVals[qIdxList],
                                    self.data.intensities[qIdxList, idx],
                                    p0=p0,
                                    bounds=bounds,
                                    sigma=self.data.errors[qIdxList, idx],
                                    **kwargs))



        self.params = params
