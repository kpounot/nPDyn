import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.ECType import ECType, DataTypeDecorator



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        scaled Gaussian given by: 

        .. math::

           S(q, \\omega ) = A \\frac{1}{ \sqrt{2 \\pi } \\gamma }
                            e^{ -(\\omega - shift)^{2} / 2\\gamma^{2} } + bkgd

        where *q* is the scattering angle, :math:`\\omega` the energy offset, A a scaling factor,
        shift a scalar to account for maximum not being at 0 exactly, and bkgd a background term.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = self.model
        self.params     = None
        self.paramsNames = ["normF", "gauW", "shift", "bkgd"] #_For plotting purpose


    def fit(self, p0=None, bounds=None):
        """ Uses Scipy's curve_fit routine to fit the pseudo-Voigt profile to the experimental data
            given in the argument resData. 
        
        """

        ECList = [] #_Fitted parameter are stored here for each q value in the dataSet

        #_Calling the scipy's curve_fit routine
        for qIdx, qWiseData in enumerate(self.data.intensities):

            #_Initial guesses for parameters based on data
            init_normF  = np.mean(qWiseData) 
            init_bkgd   = np.mean([val for val in qWiseData if val > 0])
            maxI = 2 * np.max( qWiseData )

            if p0 is None:
                p0=[init_normF, 1, 0.1, init_bkgd]

            if bounds is None:
                bounds=([0., 0., -10, 0.], [maxI, np.inf, 10, maxI])

            ECList.append(optimize.curve_fit(   self.model, 
                                                self.data.X,
                                                self.data.intensities[qIdx],
                                                sigma=self.data.errors[qIdx],
                                                p0=p0,
                                                bounds=bounds,
                                                max_nfev=10000000,
                                                method='trf'))

            self.params = ECList



                
    def model(self, x, normF, gauW, shift, bkgd):
        """ Gaussian profile for empty-cell signal.

            :arg x:     energy transfer offsets (in microeV)  
            :arg normF: normalization factor
            :arg gauW:  gaussian width parameter
            :arg shift: shift of the resolution function center from 0
            :arg bkgd:  background term 

        """

        return ( normF * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi)) + bkgd )
