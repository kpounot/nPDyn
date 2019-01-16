import numpy as np
from collections import namedtuple
from scipy.signal import fftconvolve, convolve

def protein_powder_2Lorentzians(params, dataset, qIdx=None, returnCost=True):
    """ This class can be used to fit data from powder protein samples - q-wise or globally -
        using two lorentzians, which is the number that should be used to fit QENS data. Indeed,
        considering the work of D.S. Sivia (1992) on bayesian analysis of QENS data fitting, it appears
        clearly that using more lorentzians just results in overfitting.

        Input:  dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
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

    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector

    #_Resolution function
    normF = np.array( [dataset.resData.params[i][0][0] for i in dataset.data.qIdx] )
    normF = normF.reshape(len(dataset.data.qIdx), 1)
    f_res = np.array( [dataset.resData.model(X, *dataset.resData.params[i][0][:-1], 0) 
                                                                            for i in dataset.data.qIdx] )

    if dataset.data.norm: #_Normalizes resolution function if data were normalized
        f_res /= normF

    model = np.zeros((qVals.size, gList.size, X.size)) #_Initialize the final array

    model = model + (qVals**2 * gList)[:,:,np.newaxis] #_Adding the loretzian width, and an axis for energies

    model = model / (np.pi * (X**2 + model**2)) #_Computes the lorentzians
    model = sList * model

    #_Performs the convolution for each q-value
    for idx, val in enumerate(dataset.data.qIdx):
        for sIdx in range(model.shape[1]):
            model[idx][sIdx] = np.convolve(model[idx, sIdx], f_res[idx], mode='same')

    model = np.sum(model, axis=1) #_Sum the convoluted lorentzians along axis 1 (contribution factors s)

    #_Final model, with Debye-Waller factor, EISF, convolutions and background
    model = np.exp(-qVals**2*msd/6) * (s0 * f_res + model + bkgd)

    cost = np.sum((dataset.data.intensities[dataset.data.qIdx] - model)**2 
                                            / dataset.data.errors[dataset.data.qIdx]**2, axis=1) 


    if qIdx:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)


    if returnCost:
        return cost
    else:
        return model



