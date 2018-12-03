import numpy as np
from collections import namedtuple
from scipy.signal import fftconvolve, convolve

def protein_powder_2Lorentzians(params, dataSet, resFunc, resParams, D2OFunc=None, 
                                                        D2OParams=None, qIdx=None, returnCost=True):
    """ This function can be used to fit data from powder protein samples - q-wise or globally -
        using two lorentzians, which is the number that should be used to fit QENS data. Indeed,
        considering the work of D.S. Sivia (1992) on bayesian analysis of QENS data fitting, it appears
        clearly that using more lorentzians just results in overfitting.

        Input:  params      -> parameters to be used in the model (described below)
                dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
                resFunc     -> resolution function to be used
                resParams   -> resolution function's parameters list for every q-values
                D2OFunc     -> D2O lineshape function (useless here, just a general pattern)
                D2OParams   -> D2O fitted parameters (useless here, just a general pattern)
                qIdx        -> index of the current q-value
                               if None, a global fit is performed over all q-values
                returnCost  -> if True, return the standard deviation of the model to experimental data
                               if False, return only the model 
                               
        Reference :
        -   D.S. Sivia, S König, C.J. Carlile and W.S. Howells (1992) Bayesian analysis of 
            quasi-elastic neutron scattering data. Physica B, 182, 341-348 """


    s0      = params[0]     #_contribution factor of elastic signal (EISF)
    sList   = params[1:3]   #_contribution factors for each lorentzian
    sList   = sList[np.newaxis,:,np.newaxis] #_Reshape for multiplication with model array along axis 1
    gList   = params[3:5]   #_lorentzian width
    msd     = params[5]     #_mean-squared displacement for the Debye-Waller factor
    bkgd    = params[6:]
    bkgd    = bkgd[:, np.newaxis]

    X = dataSet.X
    qVals = dataSet.qVals[dataSet.qIdx, np.newaxis] #_Reshape to a column vector


    #_Resolution function
    normF = np.array( [resParams[i][0][0] for i in dataSet.qIdx] )
    normF = normF.reshape(len(dataSet.qIdx), 1)
    f_res = np.array( [resFunc(X, *resParams[i][0][:-1], 0) for i in dataSet.qIdx] )
    if dataSet.norm: #_Normalizes resolution function if data were normalized
        f_res /= normF

    model = np.zeros((qVals.size, gList.size, X.size)) #_Initialize the final array

    model = model + (qVals**2 * gList)[:,:,np.newaxis] #_Adding the loretzian width, and an axis for energies

    model = model / (np.pi * (X**2 + model**2)) #_Computes the lorentzians
    model = sList * model

    #_Performs the convolution for each q-value
    for qIdx, val in enumerate(dataSet.qIdx):
        for sIdx in range(model.shape[1]):
            model[qIdx][sIdx] = np.convolve(model[qIdx, sIdx], f_res[qIdx], mode='same')

    model = np.sum(model, axis=1) #_Sum the convoluted lorentzians along axis 1 (contribution factors s)

    #_Final model, with Debye-Waller factor, EISF, convolutions and background
    model = np.exp(-qVals**2*msd/3) * (s0 * f_res + model + bkgd)

    cost = np.sum((dataSet.intensities[dataSet.qIdx] - model)**2 / dataSet.errors[dataSet.qIdx]**2, axis=1)


    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)


    if returnCost:
        return cost
    else:
        return model


