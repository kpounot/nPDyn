''' This module define the different functions used in nPDyn in the framework of Theano.
    Each function takes a vector of the values for which the function needs to be evaluated and
    a list of the parameters to be used. 
    All variables are shared and are to be defined as global parameters in the fitting/plotting scripts. 
    Their value can be changed using the 'set_value' method.

    These functions can be called in a small wrapper setting the values of the variables with the x 
    vector from "minimize" or "basinhopping" algorithms from scipy for example. 
    The use of the GPU allow to compute the cost element-wise in a very effective manner with large 
    data arrays. '''


import numpy as np
import theano
import theano.tensor as T
from scipy.signal import convolve

def resFunc_pseudo_voigt():
    ''' Resolution function for the backscattering spectrometer using a pseudo-voigt profile.

        Inputs: w[0] - normF   - scale factor which can be used for normalization
                w[1] - S       - contribution factor for the lorentzian
                w[2] - lorW    - lorentzian width
                w[3] - gauW    - gaussian width    
                w[4] - shift   - shift from 0 of the voigt profile's maximum
                w[5] - bkgd    - background term

        Output: The compiled theano function result '''

    #_Defining the variables to be used with the fitting procedure
    w = T.vector('w')               #_Vector for parameters
    X = T.vector('X')               #_Vector for x-axis values
    dataI = T.vector('dataI')       #_Vector for y-axis values
    dataErr = T.vector('dataErr')   #_Vector for data errors

    model = (w[0] * (w[1] * w[2] / (w[2]**2 + (X - w[4])**2) / np.pi 
                + (1-w[1]) * T.exp(-(X - w[4])**2 / (2*w[3]**2)) 
                    / (w[3] * T.sqrt(2*np.pi)) 
                + w[5]))

    f_loss = T.sum((dataI - model)**2 / dataErr**2)
    f_grad = T.grad(f_loss, w)

    resFunc = theano.function([w, X], model, allow_input_downcast=True)
    resCost = theano.function([w, X, dataI, dataErr], f_loss, allow_input_downcast=True)
    resGrad = theano.function([w, X, dataI, dataErr], f_grad, allow_input_downcast=True)

    return resFunc, resCost, resGrad


def modelFunc_water():
    ''' Model function for fitting the hydration water dynamics over the whole q-range.

        Inputs: w[0] - s0       - elastic contribution factor
                w[1] - st       - translational motions contribution factor
                w[2] - gt       - translational lorentzian width parameter
                w[3] - sr       - rotational motions contribution factor  
                w[4] - gr       - rotational lorentzian width parameter
                w[5] - msd      - mean-square displacement for the debye-waller factor
                w[6] - lambda   - diffusion exponent determining normal, sub or super-diffusion
                w[7+] - pBkgd   - contain the q-dependent background

        Output: The compiled theano functions results '''


    #_Defining the variables to be used with the fitting procedure
    w = T.vector('w')                           #_Vector for parameters
    X = T.vector('X')                           #_Vector for x-axis values
    dataI = T.matrix('dataI')                   #_Vector for y-axis values
    dataErr = T.matrix('dataErr')               #_Vector for data errors
    dataQ = T.vector('dataQ')                   #_Vector for q values
    bessel = T.matrix('bessel')                 #_contains the bessel function values at each order
    res_w = T.matrix('res_w')                   #_parameters for the resolution function

    #_Resolution function
    f_res = lambda K: (res_w[K][0] * (res_w[K][1] * res_w[K][2] 
            / (res_w[K][2]**2 + (X - res_w[K][4])**2) / np.pi 
            + (1-res_w[K][1]) * T.exp(-(X - res_w[K][4])**2 / (2*res_w[K][3]**2)) 
            / (res_w[K][3] * T.sqrt(2*np.pi)) 
            + res_w[K][5]))
    

    #_Lorentzians
    L = T.arange(bessel.shape[1]).dimshuffle(0, 'x')
    f_lor = lambda K: T.sum((w[1] * (w[2] * dataQ[K]**w[6] / (X**2 + w[2] * dataQ[K]**(2*w[6])))
                      + w[3] * (2*L+1) * bessel[K].dimshuffle(0, 'x')**2 
                      * L*(L+1) * w[4] / (np.pi * (X**2 + (L*(L+1) * w[4])**2))), axis=0)

    
    #_Defining the model function, a convolution between the resolution and the lorentzians
    model, updates = theano.scan(lambda K: (T.exp(-dataQ[K]**2 * w[5]**2)
                                       * ((w[0] + w[4] * bessel[K][0]**2) * f_res(K)
                                       + convolve(f_res(K), f_lor(K), mode='same')
                                       + w[7 + K])),
                                       sequences=T.arange(dataQ.shape[0]))

    modelFunc = theano.function([dataQ, res_w, w, X, bessel], model, allow_input_downcast=True)

    #_Defining the cost function and its gradient
    f_loss = T.sum((dataI - model)**2 / dataErr**2)
    f_grad = T.grad(f_loss, w)

    modelCost = theano.function([dataQ, res_w, w, X, bessel, dataI, dataErr], f_loss, 
                                                                            allow_input_downcast=True)
    modelGrad = theano.function([dataQ, res_w, w, X, bessel, dataI, dataErr], f_grad, 
                                                                            allow_input_downcast=True)

    return modelFunc, modelCost, modelGrad
