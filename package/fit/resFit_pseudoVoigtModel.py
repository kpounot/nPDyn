import numpy as np
from collections import namedtuple
from scipy import optimize
 
                
def resFunc(x, normF, S, lorW, gauW, shift, bkgd):
    """ Pseudo-Voigt profile for resolution function.

        Input:  x       -> energy transfer offsets (in microeV)  
                normF   -> normalization factor
                S       -> weight factor for the lorentzian
                lorW    -> lorentzian width parameter
                gauW    -> gaussian width parameter
                shift   -> shift of the resolution function center from 0
                bkgd    -> background term """

    return  (normF * (S * lorW/(lorW**2 + (x-shift)**2) #/ np.pi 
            + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) #/ (gauW*np.sqrt(2*np.pi)) 
            + bkgd))

 
def resFit(resData):
    """ Uses Scipy's curve_fit routine to fit the pseudo-Voigt profile to the experimental data
        given in the argument resData. 
        
        Returns a list of fitted parameters for each scattering angle / q-value. """

    resList = [] #_Fitted parameter are stored here for each q value in the dataSet

    #_Calling the scipy's curve_fit routine
    for qIdx, qWiseData in enumerate(resData.intensities):

        #_Initial guesses for parameters based on data
        init_normF  = np.mean(resData.intensities[qIdx]) 
        init_bkgd   = np.min([val for val in resData.intensities[qIdx] if val > 0])

        resList.append(optimize.curve_fit(  resFunc, 
                                            resData.X,
                                            resData.intensities[qIdx],
                                            sigma=resData.errors[qIdx],
                                            #p0=[init_normF, 0.2, 1, 0.5, 0.1, init_bkgd],
                                            bounds=([0., 0., 0., 0., -10, 0.],  
                                                    [np.inf, 1, 100, 100, 10, np.inf]),
                                            maxfev=100000,
                                            method='trf'))

    return resList

