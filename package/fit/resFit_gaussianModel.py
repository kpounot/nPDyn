import numpy as np
from collections import namedtuple
from scipy import optimize
 
def resFunc(x, normF, S, g0, g1, shift, bkgd):
    """ Pseudo-Voigt profile for resolution function.

        Input:  x       -> energy transfer offsets (in microeV)  
                normF   -> normalization factor
                S       -> weight factor for the lorentzian
                g0, g1  -> gaussians width parameters
                shift   -> shift of the resolution function center from 0
                bkgd    -> background term """


    return  (normF * ( S * np.exp(-(x-shift)**2 / (2*g0**2)) / (g0*np.sqrt(2*np.pi))  
            + (1-S) * np.exp(-(x-shift)**2 / (2*g1**2)) / (g1*np.sqrt(2*np.pi))
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

        maxI    = 1.5 * np.max(resData.intensities)
        maxBkgd = np.min(resData.intensities.flatten()[ np.argwhere(
                                                    resData.intensities.flatten() > 0.0)[0] ])

        p0 = [init_normF, 0.1, 1, 0.5, 0.1, init_bkgd]
        if qIdx > 0:
            p0 = resList[qIdx-1][0]

        resList.append(optimize.curve_fit(  resFunc, 
                                            resData.X,
                                            resData.intensities[qIdx],
                                            sigma=resData.errors[qIdx],
                                            p0=p0,
                                            bounds=([0., 0., 0., 0., -10, 0.],  
                                                    [maxI, 1, 100, 100, 10, maxBkgd]),
                                            max_nfev=100000,
                                            method='trf'))

    return resList

