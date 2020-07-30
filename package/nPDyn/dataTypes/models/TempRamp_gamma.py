from scipy.optimize import curve_fit

from nPDyn.dataTypes.TempRampType import DataTypeDecorator
from nPDyn.fit import fitENS_models as models



class Model(DataTypeDecorator):
    """ This class provides a model to fit q-dependent elastic
        signal measured during as a series, during a temperature ramp
        for instance.

        The model used is given by [#]_ :

        .. math::

            S(q, \\omega = 0) = frac{1} {
                 (1 + \\frac{\\sigma^2 q^2}{\\beta})^{\\beta}}

        where q is the scattering angle, :math:`\\omega` the energy offset,
        :math:`\\sigma` the mean-squared displacement,
        and :math:`\\beta` a parameter accounting for motion heterogeneity
        in the sample.

        References:

        .. [#] https://doi.org/10.1063/1.3170941

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = models.gamma
        self.params     = None
        self.paramsNames = ["MSD", "\\beta", "scaleF"]


        self.defaultBounds = (0., [10, 100, 10000.])




    def fit(self, p0=None, bounds=None):
        """ Fitting procedure that makes use of Scipy *curve_fit*. """

        if not bounds:
            bounds = self.defaultBounds

        if not p0:
            p0 = [0.2, 5, 1]

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
                                    sigma=self.data.errors[qIdxList, idx]))



        self.params = params
