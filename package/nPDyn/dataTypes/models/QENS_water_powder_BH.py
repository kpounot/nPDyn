import numpy as np

from collections import namedtuple
from scipy import optimize

from scipy.special import spherical_jn

from nPDyn.dataTypes.QENSType import DataTypeDecorator
from nPDyn.fit.fitQENS_models import water_powder as model



class Model(DataTypeDecorator):
    """ This class provides a model for water dynamics for powder samples with QENS data.

        The model (:py:func:`~fitQENS_models.water_powder`) is given by:

        .. math::

            S(q, \\omega) = e^{ -q^{2} \\langle u^{2} \\rangle / 3}
                            R(q, \\omega ) \\otimes \\left[ a_{0} \\delta(\\omega ) 
                                + a_{r} \\mathcal{L}_{\\Gamma_{r} }
                                + a_{t} \\mathcal{L}_{\\Gamma_{t} } \\right] + bkgd

        where, q is the scattering angle, 
        :math:`\\langle u^{2} \\rangle` is the mean-squared displacement, 
        R is the resolution function,
        :math:`a_{0}` is the EISF given by :math:`a_{i} + a_{r} j_{0}^{2}(qd)` where j is the 
        spherical bessel function, d the water O-H distance set to 0.96 angstrom,
        :math:`a_{i}` account for apparent immobile water molecules fraction,
        and :math:`a_{r}` accounts for fraction of waters undergoing rotational motions,
        :math:`\\omega` is the energy offset, 
        :math:`\\mathcal{L}_{\\Gamma_{r}}` accounts for rotations and is given by:

        .. math::

           \\mathcal{L}_{\\Gamma_{r}} = \\sum_{k=1}^{4} (2k+1) j_{k}^{2}(qd) 
                                    \\frac{ k(k+1)\\Gamma_{r} }{ \\omega^{2} + (k(k+1)\\Gamma_{r})^{2} }

        :math:`\\mathcal{L}_{\\Gamma_{t}}` is a Lorentzian of width obeying jump diffusion model as 
        decribed by Singwi and Sjölander [#]_ and accounts for translational motions.

        The Scipy basinhopping routine is used.

        References:

        .. [#] https://journals.aps.org/pr/abstract/10.1103/PhysRev.119.863

    """  

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model          = model
        self.params         = None
        self.paramsNames    = ['s0', 'sr', 'st', 'rotational', 'translational', 'msd', 'tau', 'bkgd'] 
        self.BH_iter        = 50
        self.disp           = True


    def fit(self, p0=None, bounds=None):
        """ Global fit """

        print("\nStarting basinhopping fitting for file: %s" % self.fileName, flush=True)
        print(50*"-", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.1, 0.5, 0.4, 2, 15, 1, 2] + [0.001 for i in range(len(self.data.qIdx))]

        if not bounds: #_Using default bounds
            minData = 1.5 * np.min( self.data.intensities ) #_To restrict background below experimental data

            bounds = [(0.0, 1), (0.0, 1), (0.0, 1), (0., 60), (0., 60), (0., 10), (0., 2)]
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

        self.globalFit = True



    def qWiseFit(self, p0=None, bounds=None):
        """ q-wise fit """

        print("\nStarting basinhopping fitting for file: %s\n" % self.fileName, flush=True)
        print(50*"-" + "\n", flush=True)

        if not p0: #_Using default initial values
            p0 = [0.1, 0.5, 0.4, 5, 5, 1, 2, 0.001]

        if not bounds: #_Using default bounds
            minData = np.min( self.data.intensities ) #_To restrict background below experimental data

            bounds = [(0., 1), (0., 1), (0., 1), (0., 60), (0., 60), (0., 10), (0, 2), (0., minData)] 


        result = []
        for i, qIdx in enumerate(self.data.qIdx):
            print("\nFitting model for q index %i\n" % qIdx, flush=True)
            result.append(optimize.basinhopping( self.model, 
                                        p0,
                                        niter = self.BH_iter,
                                        niter_success = 0.5*self.BH_iter,
                                        disp=self.disp,
                                        minimizer_kwargs={ 'args':(self, i), 
                                                           'bounds':bounds,
                                                            'options': {'maxcor': 100,
                                                                        'maxfun': 200000}}) )
                                                           


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

        if len(self.params[0].x) == len(self.paramsNames):
            params = self.params[qIdx].x
        else:
            params = self.params[qIdx].x[ [0,1,2,3,4,5,6,7+qIdx] ]

        return params



    def getParamsErrors(self, qIdx):
        """ Accessor for parameters of the model for the given q value """

        if len(self.params[0].x) == len(self.paramsNames):
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
        else:
            params = self.params[qIdx].lowest_optimization_result.hess_inv.todense()
            params = np.sqrt( np.diag( params ) )
            params = params[ [0,1,2,3,4,5,6,7+qIdx] ]

        return params



    def getWeights_and_lorWidths(self, qIdx):
        """ Accessor for weights/contribution factors and width of model Lorentzians """

        rotational = np.sum( [l*(l+1) * self.params[qIdx].x[3] for l in range(1,5)] )

        weights     = self.params[qIdx].x[1:3] / self.params[qIdx].x[:3].sum()
        lorWidths   = [rotational, self.params[qIdx].x[4] * self.data.qVals[qIdx]**2]
        labels      = ['Rotational', 'Translational']

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

        if len(self.params[0].x) == len(self.paramsNames):
            return self.params[qIdx].x[7]
        else:
            return self.params[qIdx].x[7+qIdx]



    def getSubCurves(self, qIdx):
        """ Computes the convoluted Lorentzians that are in the model and returns them 
            individually along with their labels as the last argument. 
            They can be directly plotted as a function of energies. 

        """

        resF, rot, trans = self.model(self.getParams(qIdx), self, qIdx, False, True)
        labels      = ['Rotational', 'Translational']

        return resF[qIdx], rot[qIdx], trans[qIdx], labels


