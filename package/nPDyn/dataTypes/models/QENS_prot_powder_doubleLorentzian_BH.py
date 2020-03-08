import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.QENSType import DataTypeDecorator
from nPDyn.fit.fitQENS_models import protein_powder_2Lorentzians as model



class Model(DataTypeDecorator):
    """ This class provides a model for protein dynamics in powder state using two Lorentzians.

        The model (:py:func:`~fitQENS_models.protein_powder_2Lorentzians`) is given by:

        .. math::

            S(q, \\omega) = e^{ -q^{2} \\langle u^{2} \\rangle / 3}
                            R(q, \\omega ) \\otimes \\left[ a_{0} \\delta(\\omega ) 
                                + a_{1} \\mathcal{L}_{\\Gamma_{1} }
                                + a_{2} \\mathcal{L}_{\\Gamma_{2} } \\right] + bkgd

        where, q is the scattering angle, 
        :math:`\\langle u^{2} \\rangle` is the mean-squared displacement, 
        R is the resolution function,
        :math:`\\omega` the energy offset, 
        :math:`\\mathcal{L}_{\\Gamma}` is a single Lorentzian accounting for global diffusion motions,
        and :math:`a_{1}` and :math:`a{2}` are scalars.

        The Scipy basinhopping routine is used.


    """  

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model          = model
        self.params         = None
        self.paramsNames    = ['$a_0$', '$a_1$', '$a_2$', '$\Gamma_1$', '$\Gamma_2$', 
                               'MSD', '$\\tau$','bkgd'] #_For plotting purpose
        self.BH_iter        = 50
        self.disp           = True

        self.globalFit = False


    def fit(self, p0=None, bounds=None):
        """ Global fit """

        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.6, 0.2, 0.2, 5, 15, 1, 2] + [self.data.intensities[self.data.qIdx[i]][:10].mean()*0.5
                                                    for i in range(len(self.data.qIdx))]

        if not bounds: #_Using default bounds
            bounds = [(0, np.inf) for i in range(7)]
            bounds += [(0, np.inf) for i in range(len(self.data.qIdx))]


        result = optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        interval=5,
                                        disp=self.disp,
                                        minimizer_kwargs={  'args':(self,), 
                                                            'bounds':bounds })



        #_Creating a list with the same parameters for each q-values (makes code for plotting easier)
        out = []
        for qIdx in self.data.qIdx:
            out.append(result)

        self.params = out    

        self.globalFit = True



    def qWiseFit(self, p0=None, bounds=None):
        """ q-wise fit """
        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.6, 0.2, 0.2, 2, 15, 1, 1, 0.001]

        if not bounds: #_Using default bounds
            bounds = [(0, np.inf) for i in range(8)]


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

        self.globalFit = False


    def getModel(self, qIdx):
        """ Returns the fitted model for the given q value. """

        return self.model(self.getParams(qIdx), self, qIdx, False)



#--------------------------------------------------
#_Parameters accessors
#--------------------------------------------------
    def getParams(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if not self.globalFit:
            params = self.params[qIdx].x
        else:
            params = self.params[qIdx].x[ [0,1,2,3,4,5,6,7+qIdx] ]

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if not self.globalFit:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
        else:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
            params = params[ [0,1,2,3,4,5,6,7+qIdx] ]

        return params


    def getEISFfactor(self, qIdx):
        """ Returns the contribution factor - usually called s0 - of the EISF. """

        return self.getParams(qIdx)[0]




    def getWeights_and_lorWidths(self, qIdx):
        """ Accessor for weights/contribution factors and width of model Lorentzians """

        weights     = self.params[qIdx].x[1:3]
        lorWidths   = self.params[qIdx].x[3:5] * self.data.qVals[qIdx]**2
        labels      = [r'$\Gamma_{0}$', r'$\Gamma_{1}$']

        return weights, lorWidths, labels



    def getWeights_and_lorErrors(self, qIdx):
        """ Accessor for weights/contribution factors errors and width errors of model Lorentzians """

        errList = np.array( [ np.sqrt(np.diag( params.lowest_optimization_result.hess_inv.todense())) 
                                                                                 for params in self.params ] )

        weightsErr = errList[qIdx,1:3]
        lorErr = errList[qIdx,3:5]

        return weightsErr, lorErr



    def getBackground(self, qIdx):
        """ Accessor for background term, None for this model. """

        if not self.globalFit:
            return self.params[qIdx].x[7]
        else:
            return self.params[qIdx].x[7+qIdx]



    def getSubCurves(self, qIdx):
        """ Computes the convoluted Lorentzians that are in the model and returns them 
            individually along with their labels as the last argument. 
            They can be directly plotted as a function of energies. 

        """

        resF, lor1, lor2 = self.model(self.getParams(qIdx), self, qIdx, False, True)
        labels      = [r'$\mathcal{L}_{\Gamma_{1}}$', r'$\mathcal{L}_{\Gamma_{2}}$']

        return resF[qIdx], lor1[qIdx], lor2[qIdx], labels


