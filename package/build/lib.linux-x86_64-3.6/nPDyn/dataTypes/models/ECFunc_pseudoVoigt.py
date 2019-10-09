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


    def fit(self):
        """ Uses Scipy's curve_fit routine to fit the pseudo-Voigt profile to the experimental data
            given in the argument resData. 
        
        """

        ECList = [] #_Fitted parameter are stored here for each q value in the dataSet

        #_Calling the scipy's curve_fit routine
        for qIdx, qWiseData in enumerate(ECData.intensities):

            #_Initial guesses for parameters based on data
            init_normF  = np.mean(ECData.intensities[qIdx]) 
            init_bkgd   = np.min([val for val in ECData.intensities[qIdx] if val > 0])

            maxI = 1.5 * np.max( ECData.intensities )

            ECList.append(optimize.curve_fit(   self.model, 
                                                ECData.X,
                                                ECData.intensities[qIdx],
                                                sigma=ECData.errors[qIdx],
                                                p0=[init_normF, 0.5, 1, 1, 0.1, init_bkgd],
                                                bounds=([0., 0., 0., 0., -10, 0.],  
                                                        [maxI, 1, np.inf, np.inf, 10, maxI]),
                                                max_nfev=10000000,
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

        return  (normF * (S * lorW/(lorW**2 + (x-shift)**2) /np.pi 
                + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi)) 
                + bkgd))

 

