import numpy as np
from collections import namedtuple

from scipy import optimize

def tempRampCurveFit(parent, modelFunc, fileIdx, p0=None, bounds=None):
    """ Uses Scipy's curve fit routine to fit the q-dependence of elastic signal for each temperature
        in the parent.dataSetList[fileIdx].X array.

        Input:  parent      -> main nPDyn class instance, containing the needed parameters and functions
                modelFunc   -> model function to be used for fitting
                fileIdx     -> index the the file to be used in parent.dataSetList
                qWise       -> if True, perform a q-wise fitting and store the params in fitList
                p0          -> starting values for parameters (optional)
                bounds      -> bounds for the parameters (optional)
        
        Returns a list of fitted parameters for each temperature. """

    fitList = []
    
    qIdxList = parent.dataSetList[fileIdx].qIdx

    for idx, temp in enumerate(parent.dataSetList[fileIdx].X):

        if idx != 0:
            p0 = fitList[idx-1][0]

        fitList.append( optimize.curve_fit( modelFunc, 
                                            parent.dataSetList[fileIdx].qVals[qIdxList],
                                            parent.dataSetList[fileIdx].intensities[qIdxList,idx],
                                            p0=p0,
                                            bounds=bounds,
                                            sigma=parent.dataSetList[fileIdx].errors[qIdxList,idx],
                                            max_nfev=10000000 ))



    return fitList
