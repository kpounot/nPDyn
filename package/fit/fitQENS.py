import numpy as np
from collections import namedtuple

from scipy import optimize

def basinHopping_fit(parent, modelFunc, fileIdx, qWise, p0, bounds, BH_iter, disp):
    """ Uses Scipy's basinhopping routine to fit the pseudo-Voigt profile to the experimental data
        given in the argument. 

        Input:  parent      -> main nPDyn class instance, containing the needed parameters and functions
                modelFunc   -> model function to be used for fitting
                fileIdx     -> index the the file to be used in parent.dataSetList
                qWise       -> if True, perform a q-wise fitting and store the params in fitList
                BH_iter     -> maximum number of basinhopping iteration (optional, default=300)
                p0          -> starting values for parameters (optional)
                disp        -> if True, display basinhopping's state info
        
        Returns a list of fitted parameters for each scattering angle / q-value. """

    fitList = []
    
    if len(parent.resData) == 1:
        args = (parent.dataSetList[fileIdx],
                parent.resFunc[0],
                parent.resParams[0],
                parent.D2OFunc,
                parent.D2OParams)
    else:
        args = (parent.dataSetList[fileIdx],
                parent.resFunc[fileIdx],
                parent.resParams[fileIdx],
                parent.D2OFunc,
                parent.D2OParams)

    if qWise:
        for qIdx in parent.dataSetList[fileIdx].qIdx:
            qArgs = args + (qIdx,)
            fitList.append( optimize.basinhopping(modelFunc, 
                            p0,
                            niter = BH_iter,
                            niter_success = 0.5*BH_iter,
                            disp=disp,
                            minimizer_kwargs={  'args':qArgs,
                                                'bounds':bounds }))

    else:
        result = optimize.basinhopping( modelFunc, 
                                        p0,
                                        niter = BH_iter,
                                        niter_success = 0.5*BH_iter,
                                        disp=disp,
                                        minimizer_kwargs={  'args':args,
                                                            'bounds':bounds } )

        for qIdx in parent.dataSetList[fileIdx].qIdx:
            fitList.append(result)
        


    return fitList

