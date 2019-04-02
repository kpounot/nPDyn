import numpy as np
from collections import namedtuple
from scipy.signal import fftconvolve, convolve



def D2OFit(params, dataset, qIdx=None, returnCost=True):
    """ This class can be used to fit data from D2O - q-wise or globally - using a single lorentzian and
        D2O data from IN6 at the ILL. 

        Input:  params      -> parameters for the model (described below), usually given by scipy's routines
                dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
                qIdx        -> index of the current q-value
                               if None, a global fit is performed over all q-values
                returnCost  -> if True, return the standard deviation of the model to experimental data
                               if False, return only the model """


    a0 = params[0]  #_contribution factor of elastic signal (EISF)
    a1 = params[1]  #_contribution factor of lorentzian

    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector



    #_Resolution function
    resFunc = dataset.getResFunc()
    resBkgd = [ dataset.resData.params[i][0][-1] for i in dataset.data.qIdx ]



    #_Computes D2O linewidth for each q-values
    gD2O = np.array([dataset.sD2O(dataset.data.temp, qVal) for qVal in dataset.data.qVals])

    model = np.zeros((qVals.size, X.size)) #_Initialize the final array

    model = model + gD2O

    model = a1 * model / (np.pi * (X**2 + gD2O**2)) * (1/np.pi) #_Computes the lorentzians



    #_Performs the convolution for each q-value
    for idx in range(model.shape[0]):
        model[idx] = np.convolve(model[idx], resFunc[idx], mode='same') 


    #_Final model, with EISF, convolutions and background
    model = a0 * resFunc + model  

    cost = np.sum((dataset.data.intensities[dataset.data.qIdx] - model)**2 
                                            / dataset.data.errors[dataset.data.qIdx]**2, axis=1) 


    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)


    if returnCost:
        return cost
    else:
        return model


