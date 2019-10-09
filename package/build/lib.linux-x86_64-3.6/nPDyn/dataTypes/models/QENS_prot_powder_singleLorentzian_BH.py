import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.QENSType import DataTypeDecorator
from nPDyn.fit.fitQENS_models import protein_powder_1Lorentzian as model



class Model(DataTypeDecorator):
    """ This class provides a model for protein dynamics in powder state using one Lorentzians.

        The model (:py:func:`~fitQENS_models.protein_powder_1Lorentzian`) is given by:

        .. math::

            S(q, \\omega) = e^{ -q^{2} \\langle u^{2} \\rangle / 3}
                            R(q, \\omega ) \\otimes \\left[ a_{0} \\delta(\\omega ) 
                                + a_{1} \\mathcal{L}_{\\Gamma } \\right] + bkgd

        where, q is the scattering angle, 
        :math:`\\langle u^{2} \\rangle` is the mean-squared displacement, 
        R is the resolution function,
        :math:`\\omega` the energy offset, 
        :math:`\\mathcal{L}_{\\Gamma}` is a single Lorentzian accounting for global diffusion motions,
        and :math:`a_{1}` is scalars.

        The Scipy basinhopping routine is used.


    """  


    def __init__(self, dataType):
        super().__init__(dataType)

        self.model          = model
        self.params         = None
        self.paramsNames    = ['s0', 's1', 'g1', 'msd', 'tau', 'bkgd'] #_For plotting purpose
        self.BH_iter        = 50
        self.disp           = True


    def fit(self, p0=None, bounds=None):
        """ Global fit """

        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.4, 0.6, 2, 1, 2] + [0.001 for i in range(len(self.data.qIdx))]

        if not bounds: #_Using default bounds
            minData = 1.5 * np.min( self.data.intensities ) #_To restrict background below experimental data

            bounds = [(0., 1), (0., 1), (0., 30), (0., 10), (0., 2)]
            bounds += [(0., minData) for i in range(len(self.data.qIdx))]


        result = optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        interval=25,
                                        disp=self.disp,
                                        minimizer_kwargs={  'args':(self,), 
                                                            'bounds':bounds,
                                                            'options': {'maxcor': 100,
                                                                        'maxfun': 200000}})



        #_Creating a list with the same parameters for each q-values (makes code for plotting easier)
        out = []
        for qIdx in self.data.qIdx:
            out.append(result)

        self.params = out    




    def qWiseFit(self, p0=None, bounds=None):
        """ q-wise fit """

        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.6, 0.2, 5, 1, 2, 0.001]

        if not bounds: #_Using default bounds
            minData = np.min( self.data.intensities ) #_To restrict background below experimental data

            bounds = [(0., 1), (0., 1), (0., 1000), (0., 10), (0., 2), (0., minData)] 


        result = []
        for i, qIdx in enumerate(self.data.qIdx):
            print("\nFitting model for q index %i\n" % qIdx, flush=True)
            result.append(optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self, i), 'bounds':bounds } ))



        self.params = result



    def getModel(self, qIdx):
        """ Returns the fitted model for the given q value. """

        return self.model(self.getParams(qIdx), self, qIdx, False)



#--------------------------------------------------
#_Parameters accessors
#--------------------------------------------------
    def getParams(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == len(self.paramsNames):
            params = self.params[qIdx].x
        else:
            params = self.params[qIdx].x[ [0,1,2,3,4,5+qIdx] ]

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == len(self.paramsNames):
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
        else:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
            params = params[ [0,1,2,3,4,5+qIdx] ]

        return params


    def getEISFfactor(self, qIdx):
        """ Returns the contribution factor - usually called s0 - of the EISF. """

        return self.getParams(qIdx)[0]




    def getWeights_and_lorWidths(self, qIdx):
        """ Accessor for weights/contribution factors and width of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian width
        weights     = self.params[qIdx].x[[1]] / self.params[qIdx].x[:2].sum()
        lorWidths   = self.params[qIdx].x[[2]] * self.data.qVals[qIdx]**2
        labels      = [r'$\Gamma$']

        return weights, lorWidths, labels



    def getWeights_and_lorErrors(self, qIdx):
        """ Accessor for weights/contribution factors errors and width errors of model Lorentzians """

        #_For plotting purpose, gives fitted weights and lorentzian errors
        errList = np.array( [ np.sqrt(np.diag( params.lowest_optimization_result.hess_inv.todense())) 
                                                                                 for params in self.params ] )

        weightsErr = errList[qIdx,[1]]
        lorErr = errList[qIdx,[2]]

        return weightsErr, lorErr



    def getBackground(self, qIdx):
        """ Accessor for background term, None for this model. """

        if len(self.params[0].x) == len(self.paramsNames):
            return self.params[qIdx].x[5]
        else:
            return self.params[qIdx].x[5+qIdx]



    def getSubCurves(self, qIdx):
        """ Computes the convoluted Lorentzians that are in the model and returns them 
            individually along with their labels as the last argument. 
            They can be directly plotted as a function of energies. 

        """

        resF, lor1 = self.model(self.getParams(qIdx), self, qIdx, False, True)
        labels     = [r'$L_{\Gamma}(q, \omega)$']

        return resF[qIdx], lor1[qIdx], labels


