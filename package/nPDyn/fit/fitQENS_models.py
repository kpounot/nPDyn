import numpy as np
from collections import namedtuple
from scipy.signal import fftconvolve, convolve
from scipy.special import spherical_jn




def protein_powder_2Lorentzians(params, dataset, qIdx=None, returnCost=True):
    """ This class can be used to fit data from powder protein samples - q-wise or globally -
        using two lorentzians, which is the number that should be used to fit QENS data. Indeed,
        considering the work of D.S. Sivia (1992) on bayesian analysis of QENS data fitting, it appears
        clearly that using more lorentzians just results in overfitting.

        Input:  params      -> parameters for the model (described below), usually given by scipy's routines
                dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
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
    bkgd    = params[6:]    #_background terms (q-dependent)
    bkgd    = bkgd[:, np.newaxis]

    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector
    resFunc = dataset.getResFunc()




    #_Model
    model = np.zeros((qVals.size, gList.size, X.size)) #_Initialize the final array

    model = model + (qVals**2 * gList)[:,:,np.newaxis] #_Adding the loretzian width, and an axis for energies

    model = model / (np.pi * (X**2 + model**2)) #_Computes the lorentzians

    model = np.sum(sList * model, axis=1) #_Sum the convoluted lorentzians along axis 1 (contribution factors s)


    #_Performs the convolution for each q-value
    for idx in range(model.shape[0]):
        model[idx] = np.convolve(model[idx], resFunc[idx], mode='same')


    #_Final model, with Debye-Waller factor, EISF, convolutions and background
    model = np.exp(-qVals**2*msd/3) * (s0 * resFunc + model) + bkgd

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








def water_powder(params, dataset, qIdx=None, returnCost=True):
    """ This class can be used to fit data from powder protein samples - q-wise or globally -
        focusing on water dynamics. Signal is deconvoluted in its rotational and translational motions
        contributions.

        Input:  params      -> parameters for the model (described below), usually given by scipy's routines
                dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
                qIdx        -> index of the current q-value
                               if None, a global fit is performed over all q-values
                returnCost  -> if True, return the standard deviation of the model to experimental data
                               if False, return only the model """
                               

    s0      = params[0]     #_contribution factor of elastic signal (EISF)
    sr      = params[1]     #_contribution factor for rotational motions
    st      = params[2]     #_contribution factor for translational motions
    gr      = params[3]     #_lorentzian width for rotational motions
    gt      = params[4]     #_lorentzian width for translational motions
    msd     = params[5]     #_mean-squared displacement for the Debye-Waller factor
    bkgd    = params[6:]    #_background terms (q-dependent)
    bkgd    = bkgd[:, np.newaxis]

    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Shape to a column vector
    resFunc = dataset.getResFunc()




    #_Model
    model = np.zeros((qVals.size, 2, X.size)) #_Initialize the final array

    #_Computes the q-independent lorentzians (rotational motions)
    preFactors  = np.array([ 2*i+1 * spherical_jn(i, 0.96*qVals[:,0])**2 for i in range(1,3)])[:,:,np.newaxis]
    rLor        = np.array([ i*(i+1) * gr / (X**2 + (i*(i+1) * gr)**2)  for i in range(1,3)])[:,np.newaxis,:]
    rLor        = np.sum( preFactors * rLor, axis=0 ) #_End up with (# qVals, X size) shaped array

    model[:,0,:] = sr * rLor


    #_Computes the q-dependent lorentzians (translational motions)
    lorW = (gt*qVals**2)
    tLor = lorW / (np.pi * (X**2 + lorW**2)) #_End up with (# qVals, X size) shaped array

    model[:,1,:] = st * tLor


    model = np.sum(model, axis=1) #_Sum the convoluted lorentzians along axis 1 (contribution factors s)

    #_Performs the convolution for each q-value
    for idx in range(model.shape[0]):
        model[idx] = np.convolve(model[idx], resFunc[idx], mode='same')


    #_Final model, with Debye-Waller factor, EISF, convolutions and background
    s0 = s0 + sr * spherical_jn(0, 0.96*qVals)**2
    model = np.exp(-qVals**2*msd/3) * (s0 * resFunc + model) + bkgd

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







def protein_liquid(params, dataset, D2OSignal=None, qIdx=None, returnCost=True):
    """ Fitting function for protein in liquid D2O environment.

        Input:  params      -> parameters for the model (described below), usually given by scipy's routines
                dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
                D2OSignal   -> D2OSignal, fitted or not, to be used in the model (optional, is obtained 
                                from dataset if None, but this make the whole fitting procedure much slower)
                qIdx        -> index of the current q-value
                               if None, a global fit is performed over all q-values
                returnCost  -> if True, return the standard deviation of the model to experimental data
                               if False, return only the model """


    beta    = params[0]     #_contribution factor of protein
    g0      = params[1]     #_global diffusion linewidth
    g1      = params[2]     #_internal diffusion linewidth
    tau     = params[3]     #_Residence time for jump diffusion model
    a0      = params[4:]    #_contribution factor of elastic signal (EISF), must be of same shape as number 
                            #_of q indices used
    a0      = a0[:,np.newaxis]


    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector
    normF = np.array( [ dataset.resData.params[qIdx][0][0] for qIdx in dataset.data.qIdx ] )
    resFunc = dataset.getResFunc()
    resBkgd = [ dataset.resData.params[i][0][-1] for i in dataset.data.qIdx ]



    if D2OSignal is None:
        D2OSignal = dataset.getD2OSignal()



    #_Model
    model = np.zeros((qVals.size, 2, X.size)) #_Initialize the final array

    
    if qIdx == None:
        g0 = qVals**2 * g0
        g1 = g1 * qVals**2 / (1 + g1 * qVals**2 * tau)

    model[:,0] += a0 * g0 / (X**2 + g0**2) * (1 / np.pi)
    model[:,1] += (1 - a0) * (g0+g1) / (X**2 + (g0+g1)**2) 

    model = np.sum( beta*model, axis=1 ) #_Summing the two lorentzians contributions


    #_Performs the convolution for each q-value, instrumental background should be contained in D2O signal
    for idx in range(model.shape[0]):
        model[idx] = np.convolve(model[idx], resFunc[idx], mode='same') + D2OSignal[idx]
    

    cost = np.sum( (dataset.data.intensities[dataset.data.qIdx] - model)**2 
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


