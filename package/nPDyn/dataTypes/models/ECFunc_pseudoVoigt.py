import numpy as np

from collections import namedtuple
from scipy import optimize

from nPDyn.dataTypes.ECType import ECType, DataTypeDecorator



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution given by: 

        .. math::

           S(q, \\omega ) = A \\left[ S \\frac{ \\Gamma }{ \\pi ( (\\omega - shift)^{2} + \\Gamma^{2} ) }
                            + \\frac{ (1-S) }{ \sqrt{2 \\pi } \\gamma }
                            e^{ -(\\omega - shift)^{2} / 2\\gamma^{2} } \\right] + bkgd

        where *q* is the scattering angle, :math:`\\omega` the energy offset, A a scaling factor,
        S a scalar between 0 and 1, shift a scalar to account for maximum not being at 0,
        and bkgd a background term.

    """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = self.model
        self.params     = None
        self.paramsNames = ["normF", "S", "lorW", "gauW", "shift", "bkgd"] #_For plotting purpose


    def fit(self, p0=None, bounds=None):
        """ Uses Scipy's curve_fit routine to fit the pseudo-Voigt profile to the experimental data
            given in the argument resData. 
        
        """

        ECList = [] #_Fitted parameter are stored here for each q value in the dataSet

        #_Calling the scipy's curve_fit routine
        for qIdx, qWiseData in enumerate(self.data.intensities):

            #_Initial guesses for parameters based on data
            init_normF  = np.max(qWiseData) 
            init_bkgd   = qWiseData[:10].mean()
            maxI = 1.5 * np.max( qWiseData )

            if p0 is None:
                p0=[init_normF, 0.1, 1, 1, 0., init_bkgd]

            if bounds is None:
                bounds=([0., 0., 0., 0., -10, 0.], [maxI, 1, np.inf, np.inf, 10, np.inf])




            ECList.append(optimize.curve_fit(self.model, 
                                             self.data.X,
                                             self.data.intensities[qIdx],
                                             sigma=self.data.errors[qIdx],
                                             p0=p0,
                                             bounds=bounds,
                                             method='trf'))

            self.params = ECList



                
    def model(self, x, normF, S, lorW, gauW, shift, bkgd):
        """ Pseudo-Voigt profile for resolution function.

            :arg x:     energy transfer offsets (in microeV)  
            :arg normF: normalization factor
            :arg S:     weight factor for the lorentzian
            :arg lorW:  lorentzian width parameter
            :arg gauW:  gaussian width parameter
            :arg shift: shift of the resolution function center from 0
            :arg bkgd:  background term 

        """

        return  ( normF * (S * lorW/(lorW**2 + (x-shift)**2) /np.pi 
                  + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi))) 
                  + bkgd )

 

