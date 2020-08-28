import numpy as np

from scipy import optimize

from nPDyn.dataTypes.QENSType import DataTypeDecorator
from nPDyn.fit.fitQENS_models import (
    protein_liquid_switchingDiff_internal as model)



class Model(DataTypeDecorator):
    """ This class provides a model for protein dynamics in liquid state
        for QENS data.

        The model (:py:func:`~fitQENS_models.protein_liquid_analytic_voigt`)
        is given by:

        .. math::

            S(q, \\omega) = R(q, \\omega ) \\otimes \\left[ \\beta ( a_{0}
                            \\mathcal{L}_{\\gamma }
                            + (1 - a_{0}) [
                                \\alpha \\mathcal{L}_{\\lambda_1 + \\gamma} 
                                + (1 - \\alpha ) 
                                \\mathcal{L}_{\\lambda_2 + \\gamma}
                            \\right]
                            + \\beta_{D_{2}O} \\mathcal{L}_{D_{2}O}

        where, R is the resolution function, q is the scattering angle,
        :math:`\\omega` the energy offset,
        :math:`\\mathcal{L}_{\\gamma}` is a single Lorentzian accounting
        for global diffusion motions, :math:`\\alpha` and 
        :math:`\\mathcal{L}_{\\lamba_{1,2}}` are respectively the dynamic 
        sturcture factor and the Lorentzian given by the switching diffusive
        states model for two states of internal dynamics [#]_ , 
        :math:`\\mathcal{L}_{D_{2}O}` is the
        :math:`D_{2}O` lineshape, :math:`a_{0}` acts as an EISF,
        and :math:`\\beta` and :math:`\\beta_{D_{2}O}` are scalars.

        The Scipy basinhopping routine is used.

        References:

        .. [#] https://doi.org/10.1039/C4CP04944F

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = model
        self.params     = None
        self.paramsNames = ['$D_s$', '$D_{i,1}$', '$D_{i,2}$',
                            '$\\tau_1$', '$\\tau_2$','$\\beta$', '$a_0$']
        self.BH_iter    = 20
        self.disp       = True




    def fit(self, p0=None, bounds=None):
        """ Global fit """

        print("\nStarting basinhopping fitting for file: %s"
              % self.fileName, flush=True)
        print(50 * "-", flush=True)

        if p0 is None:  # Using default initial values
            p0 = [5, 30, 100, 10, 0.1]
            p0 = p0 + [1 for i in self.data.qIdx]
            p0 += [1 for i in self.data.qIdx]

        if bounds is None:  # Using default bounds
            bounds = [(0, self.data.X.max() * 3) for i in range(3)]
            bounds += [(0., np.inf), (0., np.inf)]
            bounds += [(0, np.inf) for i in range(len(self.data.qIdx))]
            bounds += [(0, 1) for i in range(len(self.data.qIdx))]


        # D2O signal
        D2OSignal = self.getD2OSignal()


        result = optimize.basinhopping(
            self.model,
            p0,
            niter = self.BH_iter,
            niter_success = 0.5 * self.BH_iter,
            disp=self.disp,
            minimizer_kwargs={'args': (self, D2OSignal), 'bounds': bounds})



        # Creating a list with the same parameters for each
        # q-values (makes code for plotting easier)
        out = []
        for qIdx in self.data.qIdx:
            out.append(result)

        self.params = out





    def qWiseFit(self, p0=None, bounds=None):
        """ q-wise fit """

        print("\nStarting basinhopping fitting for file: %s\n"
              % self.fileName, flush=True)
        print(50 * "-" + "\n", flush=True)

        if p0 is None:  # Using default initial values
            p0 = [5, 30, 100, 10, 0.1, 0.5, 0.5]

        if bounds is None:  # Using default bounds
            bounds = [(0, self.data.X.max() * 3) for i in range(3)]
            bounds += [(0, np.inf) for i in range(3)] + [(0., 1)]


        # D2O signal
        D2OSignal = self.getD2OSignal()


        result = []
        for i, qIdx in enumerate(self.data.qIdx):

            if i != 0:
                p0 = result[-1].x

            print("\nFitting model for q index %i\n" % qIdx, flush=True)
            result.append(optimize.basinhopping(
                self.model,
                p0,
                niter = self.BH_iter,
                niter_success = 0.5 * self.BH_iter,
                disp=self.disp,
                minimizer_kwargs={'args': (self, D2OSignal, i),
                                  'bounds': bounds}))



        self.params = result




    def getModel(self, qIdx):
        """ Returns the fitted model for the given q value. """

        return self.model(self.getParams(qIdx), self,
                          self.getD2OSignal(), qIdx, False)



# -------------------------------------------------
# Parameters accessors
# -------------------------------------------------
    def getParams(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == len(self.paramsNames):
            params = self.params[qIdx].x
        else:
            params = self.params[qIdx].x[
                [0, 1, 2, 3, 4, 5 + qIdx, 5 + self.data.qIdx.size + qIdx]]
            qVal = self.data.qVals[self.data.qIdx[qIdx]]**2
            params[0] = params[0] * qVal**2
            params[1] = params[1] * qVal**2
            params[2] = params[2] * qVal**2

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == len(self.paramsNames):
            params = self.params[qIdx].lowest_optimization_result
            params = params.hess_inv.todense()
            params = np.sqrt(np.diag(params))
        else:
            params = self.params[qIdx].lowest_optimization_result
            params = params.hess_inv.todense()
            params = np.sqrt(np.diag(params))
            params = params[
                [0, 1, 2, 3, 4, 5 + qIdx, 5 + self.data.qIdx.size + qIdx]]

        return params


    def getEISFfactor(self, qIdx):
        """ Returns the contribution factor of the EISF. """

        params = self.getParams(qIdx)

        return params[5] * params[6]



    def getWeights_and_lorWidths(self, qIdx):
        """ Accessor for weights/contribution factors and width
            of model Lorentzians

        """

        # For plotting purpose, gives fitted weights and lorentzian width
        if len(self.params[0].x) == len(self.paramsNames):
            weights = [self.params[qIdx].x[6], 1 - self.params[qIdx].x[6]]
            beta    = self.params[qIdx].x[5]
        else:
            weights = [self.params[qIdx].x[5 + self.data.qIdx.size + qIdx],
                       1 - self.params[qIdx].x[
                       5 + self.data.qIdx.size + qIdx]]
            beta    = self.params[qIdx].x[5 + qIdx]

        weights = np.array(weights) * beta

        lorWidths  = self.params[qIdx].x[0:3]

        bigLambda = ((lorWidths[0] - lorWidths[1] 
                     + 1 / weights[0] - 1 / weights[1])**2
                     + 4 / (weights[0] * weights[1]))

        lambda1 = ((1/2) * (lorWidths[0] + 1 / weights[0]
                            + lorWidths[1] + 1 / weights[1] + bigLambda))

        lambda2 = ((1/2) * (lorWidths[0] + 1 / weights[0]
                            + lorWidths[1] + 1 / weights[1] - bigLambda))


        alpha = ((1 / (lambda2 - lambda1)) 
                 * weights[0] * (lorWidths[1] + 1 / weights[0] 
                                 + 1 / weights[1] - lambda1) 
                 / (weights[0] + weights[1]))

        alpha += ((1 / (lambda2 - lambda1)) 
                  * weights[1] * (lorWidths[0] + 1 / weights[0] 
                                 + 1 / weights[1] - lambda1) 
                  / (weights[0] + weights[1]))

        weights[0] = alpha
        weights[1] = 1 - alpha

        labels      = ['Global', 'Internal 1', 'Internal 2']

        return weights, lorWidths, labels




    def getWeights_and_lorErrors(self, qIdx):
        """ Accessor for weights/contribution factors errors and
            width errors of model Lorentzians

        """

        # For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array([np.sqrt(np.diag(
            params.lowest_optimization_result.hess_inv.todense()))
            for params in self.params])

        if len(self.params[0].x) == len(self.paramsNames):
            weightsErr = [errList[qIdx][6], errList[qIdx][6]]
        else:
            weightsErr = [errList[qIdx][5 + self.data.qIdx.size + qIdx],
                          errList[qIdx][5 + self.data.qIdx.size + qIdx]]

        lorErr = errList[qIdx, 0:3]

        return weightsErr, lorErr




    def getBackground(self, qIdx):
        """ Accessor for background term, None for this model. """

        return None


    def get_betaSlice(self):
        """ For global fit, returns the slice corresponding
            to beta parameter(s)

        """

        return slice(5, 5 + self.data.qIdx.size)



    def get_a0Slice(self):
        """ For global fit, returns the slice corresponding to
            a0 parameter(s)

        """

        return slice(5 + self.data.qIdx.size, None)


    def getSubCurves(self, qIdx):
        """ Accessor for individual components of the fits for given q-value.

            :returns:

                - resolution function curve
                - global diffusion Lorentzian
                - internal dynamics Lorentzian
                - labels for the previous two Lorentzians

        """

        # D2O signal
        D2OSignal = self.getD2OSignal()

        resF, gLor, iLor1, iLor2 = self.model(self.getParams(qIdx),
                                              self,
                                              D2OSignal,
                                              qIdx,
                                              False,
                                              returnSubCurves=True)

        labels = [r'$L_{\gamma_{global}}(q, \omega)$',
                  r'$L_{\lambda_{internal, 1}}(q, \omega)$',
                  r'$L_{\lambda_{internal, 2}}(q, \omega)$',
                  ]

        return resF[qIdx], gLor[qIdx], iLor1[qIdx], iLor2[qIdx], labels
