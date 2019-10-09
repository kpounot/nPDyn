import numpy as np
from collections import namedtuple
from scipy import optimize
 
def resFunc(self, x, normF, S, lorW, gauW, shift, bkgd):
    """ Pseudo-Voigt profile for resolution function.

        Input:  x       -> energy transfer offsets (in microeV)  
                normF   -> normalization factor
                S       -> weight factor for the lorentzian
                lorW    -> lorentzian width parameter
                gauW    -> gaussian width parameter
                shift   -> shift of the resolution function center from 0
                bkgd    -> background term """

    return  (normF * (S * lorW/(lorW**2 + (x-shift)**2) /np.pi 
            + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi))
            + bkgd))  
                
def resFit(self):

    for i, resFile in enumerate(self.dataList):
        resList = []
        for j, qWiseData in enumerate(resFile.intensities):
            resList.append(optimize.curve_fit(self.resFunc, 
                            resFile.energies,
                            resFile.intensities[j],
                            sigma=[val+0.0001 for val in resFile.errors[j]],
                            #p0 = [0.5, 1, 0.8, 50, 0], 
                            bounds=([0., 0., 0., 0., -10, 0.],  
                                    [2000, 1, 5, 5, 10, 0.5]),
                            max_nfev=10000000,
                            method='trf'))
        self.resFitList.append(resList)

    return self.resFitList 

