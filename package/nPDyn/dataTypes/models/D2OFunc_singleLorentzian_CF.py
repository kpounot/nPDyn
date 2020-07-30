import numpy as np

from scipy import optimize

from nPDyn.dataTypes.baseType import DataTypeDecorator
from nPDyn.fit.D2O_params_from_IN6 import getD2Odata



class Model(DataTypeDecorator):
    """ This class provides a model for :math:`D_{2}O` signal fitting.

        The model equation is given by:

        .. math::

           S(q, \\omega ) = R(q, \\omega ) \\otimes \\left[
                            a_{0} \\frac{ \\Gamma_{D_{2}O}(q) }{
                            \\pi ( \\omega^{2} + \\Gamma_{D_{2}O}^{2}(q))}
                            \\right]

        where *q* is the scattering angle, :math:`\\omega` the energy offset,
        :math:`a_{0}` is a scalar, and R the resolution function.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model       = model
        self.params      = None
        self.paramsNames = ["a0"]  # For plotting purpose

        self.volFraction = 0.95
        self.getD2OData  = getD2Odata
        self.sD2O        = getD2Odata()
        self.disp        = True


    def fit(self, p0=None, bounds=None):
        if self.disp:
            print("\nUsing Scipy's curve_fit to fit data from file: %s"
                  % self.fileName, flush=True)

        if not p0:  # Using default initial values
            p0 = np.array([0.1])

        if not bounds:  # Using default bounds
            bounds = ([0., 10])


        result = []
        for qIdx, qVal in enumerate(self.data.qVals):
            result.append(optimize.curve_fit(
                lambda x, *p0: self.model(p0, self, qIdx, False),
                self.data.X,
                self.data.intensities[qIdx],
                sigma=self.data.errors[qIdx],
                p0=p0,
                bounds=bounds))

        self.params = result



    def getD2OContributionFactor(self):
        """ Returns the contribution factor of D2O lineshape in the model """

        aD2O = np.array([self.params[i][0] for i in self.data.qIdx])

        return aD2O



    def getD2OSignal(self, qIdx=None):
        """ Computes D2O line shape for each q values.

            If a qIdx is given, returns D2O signal only
            for the corresponding q value.

        """

        D2OSignal = np.array([
            self.model(self.params[idx][0], self, idx, False)
            for idx in self.data.qIdx])

        D2OSignal *= self.volFraction


        return D2OSignal



    def getParams(self, qIdx=None):
        """ Method used to obtain the fitted parameters for a
            given index of momentum transfer q.

            :arg qIdx: index momentum transfer q to be used. If None,
                       parameters for all q are returned.

        """

        params = np.array([self.params[i][0] for i in self.data.qIdx])

        if qIdx is not None:
            return params[qIdx]
        else:
            return params


# -------------------------------------------------
# model
# -------------------------------------------------
def model(params, dataset, qIdx=None, returnCost=True):
    """ This class can be used to fit data from D2O - q-wise or
        globally - using a single lorentzian and D2O data from IN6 at the ILL.

        :arg params:     parameters for the model (described below),
                         usually given by scipy's routines
        :arg dataSet:    dataSet namedtuple containing x-axis values,
                         intensities, errors,...
        :arg qIdx:       index of the current q-value
                         if None, a global fit is performed over all q-values
        :arg returnCost: if True, return the standard deviation of the model
                         to experimental data
                         if False, return only the model

    """


    a0 = params[0]  # contribution factor of lorentzian


    X = dataset.data.X
    # Reshape to a column vector
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis]
    resFunc = dataset.getResFunc()
    resBkgd = dataset.getResBkgd()



    # Computes D2O linewidth for each q-values
    gD2O = np.array([dataset.sD2O(dataset.data.temp, qVal)
                     for qVal in dataset.data.qVals])

    model = np.zeros((qVals.size, X.size))  # Initialize the final array

    model = model + gD2O

    model = a0 * model / (np.pi * (X**2 + model**2))  # Computes the lorentzian


    # Performs the convolution for each q-value
    for idx in range(model.shape[0]):
        model[idx] = np.convolve(
            model[idx], resFunc[idx], mode='same') + resBkgd[idx]


    cost = np.sum((1 + np.log((
        dataset.data.intensities[dataset.data.qIdx] - model)**2))
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
