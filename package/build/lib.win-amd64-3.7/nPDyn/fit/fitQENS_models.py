import numpy as np
from collections import namedtuple
from scipy.signal import fftconvolve, convolve
from scipy.special import spherical_jn, wofz


from ..dataTypes.models import resFunc_gaussian, resFunc_pseudoVoigt


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
    if qIdx is None:
        gt = (gt*qVals**2)

    tLor = gt / (np.pi * (X**2 + gt**2)) #_End up with (# qVals, X size) shaped array

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

    g0      = params[0]     #_global diffusion linewidth
    g1      = params[1]     #_internal diffusion linewidth
    tau     = params[2]     #_Residence time for jump diffusion model
    beta    = params[3]     #_contribution factor of protein, same shape as # of q-values used
    a0      = params[4]     #_contribution factor of EISF

    if params.size != 5:
        beta    = params[3:3+dataset.data.qIdx.size][:,np.newaxis]
        a0      = params[3+dataset.data.qIdx.size:][:,np.newaxis]


    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector
    normF = np.array( [ dataset.resData.params[qIdx][0][0] for qIdx in dataset.data.qIdx ] )
    resFunc = dataset.getResFunc()



    if D2OSignal is None:
        D2OSignal =  dataset.getD2OSignal()



    #_Model
    model = np.zeros((qVals.size, 2, X.size)) #_Initialize the final array

    
    if qIdx == None:
        g0 = qVals**2 * g0
        g1 = g1 * qVals**2 / (1 + g1 * qVals**2 * tau)

    model[:,0] += a0 * g0 / (X**2 + g0**2) * (1 / np.pi)
    model[:,1] += (1 - a0) * (g0+g1) / (X**2 + (g0+g1)**2) * (1 / np.pi) 

    model = np.sum( model, axis=1 ) #_Summing the two lorentzians contributions


    #_Performs the convolution for each q-value, instrumental background should be contained in D2O signal
    for idx, val in enumerate(model):
        model[idx] = np.convolve(model[idx], resFunc[idx], mode='same') 
        
    model = beta * model + D2OSignal
    

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




def protein_liquid_analytic_voigt(params, dataset, D2OSignal=None, qIdx=None, returnCost=True, 
                                                                            scanIdx=slice(0,None)):
    """ Fitting function for protein in liquid D2O environment.

        This makes uses of an analytic expression for convolution with resolution function, therefore
        resolution function used here should be a sum of two gaussians or pseudo-voigt.

        Input:  params      -> parameters for the model (described below), usually given by scipy's routines
                dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
                D2OSignal   -> D2OSignal, fitted or not, to be used in the model (optional, is obtained 
                                from dataset if None, but this make the whole fitting procedure much slower)
                qIdx        -> index of the current q-value
                               if None, a global fit is performed over all q-values
                returnCost  -> if True, return the standard deviation of the model to experimental data
                               if False, return only the model 
                scanIdx     -> only for FWS (or QENS) data series, index of the scan being fitted """


    g0      = params[0]     #_global diffusion linewidth
    g1      = params[1]     #_internal diffusion linewidth
    tau     = params[2]     #_Residence time for jump diffusion model
    beta    = params[3]     #_contribution factor of protein, same shape as # of q-values used
    a0      = params[4]     #_contribution factor of EISF


    if params.size != 5:
        betaSlice   = dataset.get_betaSlice()
        a0Slice     = dataset.get_a0Slice()
        beta        = params[betaSlice][:,np.newaxis]
        a0          = params[a0Slice][:,np.newaxis]


    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector
   

    if D2OSignal is None:
        D2OSignal =  dataset.getD2OSignal()

    if qIdx == None:
        g0 = qVals**2 * g0
        g1 = g1 * qVals**2 / (1 + g1 * qVals**2 * tau)



    #_Computes the different components of the convolution
    resG0 = np.array( [dataset.resData.params[qIdx][0][2] for qIdx in dataset.data.qIdx] )[:,np.newaxis]
    resG1 = np.array( [dataset.resData.params[qIdx][0][3] for qIdx in dataset.data.qIdx] )[:,np.newaxis]
    resS  = np.array( [dataset.resData.params[qIdx][0][1] for qIdx in dataset.data.qIdx] )[:,np.newaxis]

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_G_resG0 = (g0 + resG0) / (np.pi * (X**2 + (g0 + resG0)**2))
        conv_G_resG1 = wofz((X + 1j*g0) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))
        conv_I_resG0 = (g0 + g1 + resG0) / (np.pi * (X**2 + (g0 + g1 + resG0)**2))
        conv_I_resG1 = wofz((X + 1j*(g1 + g0)) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))


    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_G_resG0 = wofz((X + 1j*g0) / (resG0 * np.sqrt(2))).real / (resG0 * np.sqrt(2*np.pi))
        conv_G_resG1 = wofz((X + 1j*g0) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))
        conv_I_resG0 = wofz((X + 1j*(g1 + g0)) / (resG0 * np.sqrt(2))).real / (resG0 * np.sqrt(2*np.pi))
        conv_I_resG1 = wofz((X + 1j*(g1 + g0)) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))

    else:
        print("The resolution function model used is not supported by this model.\n"
                + "Please use either resFunc_pseudoVoigt or resFunc_gaussian.\n")
        return


        
    model = ( a0 * (resS*conv_G_resG0 + (1-resS)*conv_G_resG1)
                    + (1-a0) * ( resS*conv_I_resG0 + (1-resS)*conv_I_resG1) ) 


    model = beta * model + D2OSignal
    

    cost = np.sum( (dataset.data.intensities[scanIdx][dataset.data.qIdx] - model)**2 
                                        * (dataset.data.intensities[scanIdx][dataset.data.qIdx] 
                                            / dataset.data.errors[scanIdx][dataset.data.qIdx]), axis=1) 
                                   

    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)



    if returnCost:
        return cost
    else:
        return model





def protein_liquid_analytic_voigt_CF(X, params, dataset, D2OSignal=None, qIdx=None, scanIdx=slice(0,None)):
    """ Fitting function for protein in liquid D2O environment.

        This makes uses of an analytic expression for convolution with resolution function, therefore
        resolution function used here should be a sum of two gaussians or pseudo-voigt.

        Input:  params      -> parameters for the model (described below), usually given by scipy's routines
                dataSet     -> dataSet namedtuple containing x-axis values, intensities, errors,...
                D2OSignal   -> D2OSignal, fitted or not, to be used in the model (optional, is obtained 
                                from dataset if None, but this make the whole fitting procedure much slower)
                qIdx        -> index of the current q-value
                               if None, a global fit is performed over all q-values
                returnCost  -> if True, return the standard deviation of the model to experimental data
                               if False, return only the model 
                scanIdx     -> only for FWS (or QENS) data series, index of the scan being fitted """


    params = np.array(params)

    g0      = params[0]     #_global diffusion linewidth
    g1      = params[1]     #_internal diffusion linewidth
    tau     = params[2]     #_Residence time for jump diffusion model
    beta    = params[3]     #_contribution factor of protein, same shape as # of q-values used
    a0      = params[4]     #_contribution factor of EISF


    if params.size != 5:
        betaSlice   = dataset.get_betaSlice()
        a0Slice     = dataset.get_a0Slice()
        beta        = params[betaSlice][:,np.newaxis]
        a0          = params[a0Slice][:,np.newaxis]


    X = dataset.data.X

    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis] #_Reshape to a column vector
   

    if D2OSignal is None:
        D2OSignal =  dataset.getD2OSignal()

    if qIdx == None:
        g0 = qVals**2 * g0
        g1 = g1 * qVals**2 / (1 + g1 * qVals**2 * tau)



    #_Computes the different components of the convolution
    resG0 = np.array( [dataset.resData.params[qIdx][0][2] for qIdx in dataset.data.qIdx] )[:,np.newaxis]
    resG1 = np.array( [dataset.resData.params[qIdx][0][3] for qIdx in dataset.data.qIdx] )[:,np.newaxis]
    resS  = np.array( [dataset.resData.params[qIdx][0][1] for qIdx in dataset.data.qIdx] )[:,np.newaxis]

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_G_resG0 = (g0 + resG0) / (np.pi * (X**2 + (g0 + resG0)**2))
        conv_G_resG1 = wofz((X + 1j*g0) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))
        conv_I_resG0 = (g0 + g1 + resG0) / (np.pi * (X**2 + (g0 + g1 + resG0)**2))
        conv_I_resG1 = wofz((X + 1j*(g1 + g0)) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))


    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_G_resG0 = wofz((X + 1j*g0) / (resG0 * np.sqrt(2))).real / (resG0 * np.sqrt(2*np.pi))
        conv_G_resG1 = wofz((X + 1j*g0) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))
        conv_I_resG0 = wofz((X + 1j*(g1 + g0)) / (resG0 * np.sqrt(2))).real / (resG0 * np.sqrt(2*np.pi))
        conv_I_resG1 = wofz((X + 1j*(g1 + g0)) / (resG1 * np.sqrt(2))).real / (resG1 * np.sqrt(2*np.pi))

    else:
        print("The resolution function model used is not supported by this model.\n"
                + "Please use either resFunc_pseudoVoigt or resFunc_gaussian.\n")
        return


        
    model = ( a0 * (resS*conv_G_resG0 + (1-resS)*conv_G_resG1)
                    + (1-a0) * ( resS*conv_I_resG0 + (1-resS)*conv_I_resG1) ) 


    model = beta * model + D2OSignal
    

    if qIdx is not None:
        model   = model[qIdx]


    return model.flatten()



