import numpy as np
from collections import namedtuple
from scipy import optimize


def modelFunc(x, qVal, temp, gD2O, a1):

    return a1 * gD2O / (gD2O**2 + x**2)


def fitFunc(params, dataSet, parent, qIdx, returnCost=True):
    """ This function takes several arguments which are automatically given by basinHopping_D2OFit 
        function. 
        
        Input:  params      -> starting random parameters for the minimization
                dataSet     -> experimental D2O data
                parent      -> dataSet class instance in which the resolution function has been fitted
                qIdx        -> current index of the q-value in the dataSet.qVals list
                returnCost  -> if used with basinhopping, the function is expected to return the cost,
                               and not the computed model lineshape itself. """

    resParams     = parent.resParams[qIdx][0]
    temp          = dataSet.temp
    qVal          = dataSet.qVals[qIdx]
    gD2O          = parent.D2OFunc(temp, qVal) #_Get the D2O lineshape's width at half maximum

    cost = 0

    model   = modelFunc(dataSet.X, qVal, temp, gD2O, *params[1:])
    resFunc = parent.resFunc(dataSet.X, *resParams)

    model = params[0] * resFunc + np.convolve(model, resFunc, mode='same') + resParams[-1]

    cost += np.sum( (dataSet.intensities[qIdx] - model)**2  
                    / (dataSet.errors[qIdx])**2 )  

    if returnCost:
        return cost
    else:
        return model

            
def basinHopping_D2OFit(D2OData, parent, BH_iter, p0, bounds, disp):
    """ Uses Scipy's basinhopping routine to fit the pseudo-Voigt profile to the experimental data
        given in the argument D2OData. 

        Input:  D2OData     -> experimental QENS data for D2O
                parent      -> the dataSet class instance (used to access the resolution function)
                BH_iter     -> maximum number of basinhopping iteration (optional, default=300)
                p0          -> starting values for parameters (optional)
                disp        -> if True, display basinhopping's state info
        
        Returns a list of fitted parameters for each scattering angle / q-value. """

    fitList = []
    for qIdx, qWiseData in enumerate(D2OData.intensities):

        fitList.append(optimize.basinhopping(fitFunc, 
                        p0,
                        niter = BH_iter,
                        niter_success = 0.25*BH_iter,
                        disp=disp,
                        minimizer_kwargs={  'args': (D2OData, parent, qIdx),
                                            'bounds': bounds }))

    return fitList

