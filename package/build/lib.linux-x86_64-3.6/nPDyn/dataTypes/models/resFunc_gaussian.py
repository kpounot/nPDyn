import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.resType import ResType, DataTypeDecorator



class Model(DataTypeDecorator):
    """ This class provides a model to fit a resolution function using a sum of two normalized
        gaussians scaled by a common factor, plus a background term.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = self.resFunc
        self.params     = None
        self.paramsNames = ["normF", "S", "g0", "g1", "shift", "bkgd"] #_For plotting purpose



    def fit(self, p0=None, bounds=None):
        """ Fitting procedure """

        self.params = self.resFit(p0, bounds)


     
    def resFunc(self, x, normF, S, g0, g1, shift, bkgd):
        """ Double gaussian profile for resolution function.

            :arg x:      energy transfer offsets (in microeV)  
            :arg normF:  normalization factor
            :arg S:      weight factor for the lorentzian
            :arg g0, g1: gaussians width parameters
            :arg shift:  shift of the resolution function center from 0
            :arg bkgd:   background term 

        """


        return  (normF * ( S * np.exp(-(x-shift)**2 / (2*g0**2)) / (g0*np.sqrt(2*np.pi))  
                + (1-S) * np.exp(-(x-shift)**2 / (2*g1**2)) / (g1*np.sqrt(2*np.pi))
                + bkgd))  
                    



    def resFit(self, p0=None, bounds=None):
        """ Uses Scipy's curve_fit routine to fit the double gaussian profile to the experimental data. 
            
            Returns a list of fitted parameters for each scattering angle / q-value. 

        """

        resList = [] #_Fitted parameter are stored here for each q value in the dataSet

        #_Calling the scipy's curve_fit routine
        for qIdx, qWiseData in enumerate(self.data.intensities):

            #_Initial guesses for parameters based on data
            maxI    = 1.2 * np.max( qWiseData )
            maxBkgd = 5 * np.mean( qWiseData[ qWiseData > 0 ] )
            maxWidth = 0.05 * np.max(self.data.X)

            init_normF  = 0.66 * maxI
            init_bkgd   = 0.5 * maxBkgd

            if not p0:
                p0 = [init_normF, 0.1, 1, 1, 0.1, init_bkgd]

            if not bounds:
                bounds = ([0., 0., 0., 0., -10, 0.],  [maxI, 1, maxWidth, maxWidth, 10, maxBkgd])


            resList.append(optimize.curve_fit(  self.resFunc, 
                                                self.data.X,
                                                self.data.intensities[qIdx],
                                                sigma=self.data.errors[qIdx],
                                                p0=p0,
                                                bounds=bounds,
                                                max_nfev=1000000,
                                                method='trf'))

        return resList

