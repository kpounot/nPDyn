import numpy as np

from scipy.special import spherical_jn, wofz

from nPDyn.dataTypes.models import resFunc_gaussian, resFunc_pseudoVoigt


def protein_powder_2Lorentzians(params, dataset, qIdx=None,
                                returnCost=True, returnSubCurves=False):
    """ This class can be used to fit data from powder protein samples
        - q-wise or globally - using two Lorentzians.

        :arg params:          parameters for the model (described below),
                              usually given by scipy's routines
        :arg dataset:         dataset namedtuple containing x-axis values,
                              intensities, errors,...
        :arg qIdx:            index of the current q-value
                              if None, a global fit is performed over
                              all q-values
        :arg returnCost:      if True, return the standard deviation of
                              the model to experimental data
                              if False, return only the model
        :arg returnSubCurves: if True, returns each individual component
                              of the model after convolution

    """


    s0      = params[0]    # contribution factor of elastic signal (EISF)
    sList   = params[1:3]  # contribution factors for each lorentzian
    g0      = params[3]    # lorentzian width
    g1      = params[4]    # lorentzian width
    msd     = params[5]    # MSD for the Debye-Waller factor
    tau     = params[6]    # residence time for translational motions
    bkgd    = params[7:]   # background terms (q-dependent)
    bkgd    = bkgd[:, np.newaxis]

    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis]
    resFunc = dataset.getResFunc()

    # Computes the different components of the convolution
    resG0 = np.array([dataset.resData.params[qIdx][0][2]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resG1 = np.array([dataset.resData.params[qIdx][0][3]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resS  = np.array([dataset.resData.params[qIdx][0][1]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]

    g0 = g0 * qVals**2
    g1 = g1 * qVals**2 * tau

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_1_resG0 = (g0 + resG0) / (np.pi * (X**2 + (g0 + resG0)**2))
        conv_1_resG1 = (wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
                        / (resG1 * np.sqrt(2 * np.pi)))
        conv_2_resG0 = (g1 + resG0) / (np.pi * (X**2 + (g1 + resG0)**2))
        conv_2_resG1 = (wofz((X + 1j * (g1)) / (resG1 * np.sqrt(2))).real
                        / (resG1 * np.sqrt(2 * np.pi)))


    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_1_resG0 = (wofz((X + 1j * g0) / (resG0 * np.sqrt(2))).real
                        / (resG0 * np.sqrt(2 * np.pi)))
        conv_1_resG1 = (wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
                        / (resG1 * np.sqrt(2 * np.pi)))
        conv_2_resG0 = (wofz((X + 1j * (g1)) / (resG0 * np.sqrt(2))).real
                        / (resG0 * np.sqrt(2 * np.pi)))
        conv_2_resG1 = (wofz((X + 1j * (g1)) / (resG1 * np.sqrt(2))).real
                        / (resG1 * np.sqrt(2 * np.pi)))

    else:
        print("The resolution function model used is \
               not supported by this model.\n \
               Please use either resFunc_pseudoVoigt or resFunc_gaussian.\n")
        return



    lor1 = sList[0] * (resS * conv_1_resG0 + (1 - resS) * conv_1_resG1)
    lor2 = sList[1] * (resS * conv_2_resG0 + (1 - resS) * conv_2_resG1)


    # Final model, with Debye-Waller factor, EISF,
    # convolutions and background
    model = np.exp(-qVals**2 * msd / 3) * (s0 * resFunc + lor1 + lor2) + bkgd

    cost = np.sum((dataset.data.intensities[dataset.data.qIdx] - model)**2
                  / dataset.data.errors[dataset.data.qIdx]**2, axis=1)


    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)


    if returnCost:
        return cost

    elif returnSubCurves:
        resF = np.exp(-qVals**2 * msd / 3) * s0 * resFunc
        lor1 = np.exp(-qVals**2 * msd / 3) * lor1
        lor2 = np.exp(-qVals**2 * msd / 3) * lor2

        return resF, lor1, lor2

    else:
        return model


def protein_powder_1Lorentzian(params, dataset, qIdx=None,
                               returnCost=True, returnSubCurves=False):
    """ This class can be used to fit data from powder protein
        samples - q-wise or globally - using one Lorentzian.

        :arg params:          parameters for the model (described below)
        :arg dataset:         dataset namedtuple containing x-axis values,
                              intensities, errors,...
        :arg qIdx:            index of the current q-value
                              if None, a global fit is performed
                              over all q-values
        :arg returnCost:      if True, return the standard deviation of
                              the model to experimental data
                              if False, return only the model
        :arg returnSubCurves: if True, returns each individual component of
                              the model after convolution

    """

    s0      = params[0]     # contribution factor of elastic signal (EISF)
    s1      = params[1]     # contribution factor for the Lorentzian
    g1      = params[2]     # lorentzian width
    msd     = params[3]     # mean-squared displacement in Debye-Waller factor
    bkgd    = params[4:]    # background terms (q-dependent)
    bkgd    = bkgd[:, np.newaxis]

    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis]
    resFunc = dataset.getResFunc()

    # Computes the different components of the convolution
    resG0 = np.array([dataset.resData.params[qIdx][0][2]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resG1 = np.array([dataset.resData.params[qIdx][0][3]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resS  = np.array([dataset.resData.params[qIdx][0][1]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]

    g1 = g1 * qVals**2

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_1_resG0 = (g1 + resG0) / (np.pi * (X**2 + (g1 + resG0)**2))
        conv_1_resG1 = (wofz((X + 1j * g1) / (resG1 * np.sqrt(2))).real
                        / (resG1 * np.sqrt(2 * np.pi)))

    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_1_resG0 = (wofz((X + 1j * g1) / (resG0 * np.sqrt(2))).real
                        / (resG0 * np.sqrt(2 * np.pi)))
        conv_1_resG1 = (wofz((X + 1j * g1) / (resG1 * np.sqrt(2))).real
                        / (resG1 * np.sqrt(2 * np.pi)))

    else:
        print("The resolution function model used is not \
               supported by this model.\n \
               Please use either resFunc_pseudoVoigt or resFunc_gaussian.\n")
        return



    lor1 = s1 * (resS * conv_1_resG0 + (1 - resS) * conv_1_resG1)


    # Final model, with Debye-Waller factor, EISF, convolutions and background
    model = np.exp(-qVals**2 * msd / 3) * (s0 * resFunc + lor1)  + bkgd

    cost = np.sum((dataset.data.intensities[dataset.data.qIdx] - model)**2
                  / dataset.data.errors[dataset.data.qIdx]**2, axis=1)


    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)


    if returnCost:
        return cost

    elif returnSubCurves:
        resF = np.exp(-qVals**2 * msd / 3) * s0 * resFunc
        lor1 = np.exp(-qVals**2 * msd / 3) * lor1

        return resF, lor1

    else:
        return model







def water_powder(params, dataset, qIdx=None, returnCost=True,
                 returnSubCurves=False):
    """ This class can be used to fit data from powder protein
        samples - q-wise or globally - focusing on water dynamics.
        Signal is decomposed in its rotational and translational motions
        contributions.

        :arg params:          parameters for the model (described below),
                              usually given by scipy's routines
        :arg dataset:         dataset namedtuple containing x-axis values,
                              intensities, errors,...
        :arg qIdx:            index of the current q-value
                              if None, a global fit is performed over
                              all q-values
        :arg returnCost:      if True, return the standard deviation of the
                              model to experimental data
                              if False, return only the model
        :arg returnSubCurves: if True, returns each individual component of
                              the model after convolution

    """


    s0      = params[0]     # contribution factor of elastic signal (EISF)
    sr      = params[1]     # contribution factor for rotational motions
    st      = params[2]     # contribution factor for translational motions
    gr      = params[3]     # lorentzian width for rotational motions
    gt      = params[4]     # lorentzian width for translational motions
    msd     = params[5]     # MSD for the Debye-Waller factor
    tau     = params[6]     # exponent for gt*qVals (2 for brownian diffusion)
    bkgd    = params[7:]    # background terms (q-dependent)
    bkgd    = bkgd[:, np.newaxis]

    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis]
    resFunc = dataset.getResFunc()

    # Gets resolution function values for convolution
    resG0 = np.array([dataset.resData.params[qIdx][0][2]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resG1 = np.array([dataset.resData.params[qIdx][0][3]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resS  = np.array([dataset.resData.params[qIdx][0][1]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]


    # Computes the q-independent lorentzians (rotational motions)
    preFactors  = np.array(
        [(2 * i + 1) * spherical_jn(i, 0.96 * qVals[:, 0])**2
         for i in range(1, 8)])
    rLor = np.array([i * (i + 1) * gr for i in range(1, 8)])

    # Performs the convolution for each q-value
    rot = np.zeros((qVals.size, X.size))
    for idx, val in enumerate(rLor):
        if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
            conv_resG0 = (val + resG0) / (np.pi * (X**2 + (val + resG0)**2))
            conv_resG1 = wofz((X + 1j * val) / (resG1 * np.sqrt(2))).real
            conv_resG1 /= (resG1 * np.sqrt(2 * np.pi))


        elif isinstance(dataset.resData, resFunc_gaussian.Model):
            conv_resG0 = wofz((X + 1j * val) / (resG0 * np.sqrt(2))).real
            conv_resG0 /= (resG0 * np.sqrt(2 * np.pi))
            conv_resG1 = wofz((X + 1j * val) / (resG1 * np.sqrt(2))).real
            conv_resG1 /= (resG1 * np.sqrt(2 * np.pi))

        else:
            print("The resolution function model used is not \
                   supported by this model.\n \
                   Please use either resFunc_pseudoVoigt \
                   or resFunc_gaussian.\n")
            return

        rot += sr * preFactors[idx][:, np.newaxis]
        rot = rot * (resS * conv_resG0 + (1 - resS) * conv_resG1)




    # Computes the q-dependent lorentzians (translational motions)
    gt = (gt * qVals**tau)

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_resG0 = (gt + resG0) / (np.pi * (X**2 + (gt + resG0)**2))
        conv_resG1 = wofz((X + 1j * gt) / (resG1 * np.sqrt(2))).real
        conv_resG1 /= (resG1 * np.sqrt(2 * np.pi))


    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_resG0 = wofz((X + 1j * gt) / (resG0 * np.sqrt(2))).real
        conv_resG0 /= (resG0 * np.sqrt(2 * np.pi))
        conv_resG1 = wofz((X + 1j * gt) / (resG1 * np.sqrt(2))).real
        conv_resG1 /= (resG1 * np.sqrt(2 * np.pi))

    else:
        print("The resolution function model used is not \
               supported by this model.\n \
               Please use either resFunc_pseudoVoigt or \
               resFunc_gaussian.\n")
        return


    trans = st * (resS * conv_resG0 + (1 - resS) * conv_resG1)


    # Final model, with Debye-Waller factor, EISF, convolutions and background
    eisfF = s0 + sr * spherical_jn(0, 0.96 * qVals)**2
    model = np.exp(-qVals**2 * msd) * (eisfF * resFunc + rot + trans)

    model += bkgd

    cost = np.sum((dataset.data.intensities[dataset.data.qIdx] - model)**2
                  / dataset.data.errors[dataset.data.qIdx]**2, axis=1)


    if qIdx is not None:
        cost  = cost[qIdx]
        model = model[qIdx]
    else:
        cost = np.sum(cost)


    if returnCost:
        return cost

    elif returnSubCurves:
        res   = np.exp(-qVals**2 * msd / 3) * eisfF * resFunc
        rot   = np.exp(-qVals**2 * msd / 3) * rot
        trans = np.exp(-qVals**2 * msd / 3) * trans

        return res, rot, trans

    else:
        return model







def protein_liquid_analytic_voigt(params, dataset, D2OSignal=None,
                                  qIdx=None, returnCost=True,
                                  scanIdx=slice(0, None),
                                  returnSubCurves=False):
    """ Fitting function for protein in liquid D2O environment.

        This makes uses of an analytic expression for convolution with
        resolution function, therefore resolution function used here should
        be a sum of two gaussians or pseudo-voigt.

        :arg params:          parameters for the model (described below),
                              usually given by Scipy routines
        :arg dataset:         dataset namedtuple containing x-axis values,
                              intensities, errors,...
        :arg D2OSignal:       D2O signal data as a 2D array with shape
                              (nbr q-values, S(q, :math:`\\omega`))
        :arg qIdx:            index of the current q-value
                              if None, a global fit is performed over
                              all q-values
        :arg returnCost:      if True, return the standard deviation of the
                              model to experimental data
                              if False, return only the model
        :arg scanIdx:         only for FWS (or QENS) data series, index of
                              the scan being fitted
        :arg returnSubCurves: if True, returns each individual component of
                              the model after convolution

    """


    g0      = params[0]     # global diffusion linewidth
    g1      = params[1]     # internal diffusion linewidth
    tau     = params[2]     # Residence time for jump diffusion model
    beta    = params[3]     # contribution factor of protein
    a0      = params[4]     # contribution factor of EISF


    if params.size != 5:
        betaSlice   = dataset.get_betaSlice()
        a0Slice     = dataset.get_a0Slice()
        beta        = params[betaSlice][:, np.newaxis]
        a0          = params[a0Slice][:, np.newaxis]


    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis]
    resFunc = dataset.getResFunc()


    if D2OSignal is None:
        D2OSignal = dataset.getD2OSignal()


    if qIdx is None:
        g0 = qVals**2 * g0
        g1 = g1 * qVals**2 / (1 + g1 * qVals**2 * tau)



    # Computes the different components of the convolution
    resG0 = np.array([dataset.resData.params[qIdx][0][2]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resG1 = np.array([dataset.resData.params[qIdx][0][3]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resS  = np.array([dataset.resData.params[qIdx][0][1]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_G_resG0 = (g0 + resG0) / (np.pi * (X**2 + (g0 + resG0)**2))
        conv_G_resG1 = wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
        conv_G_resG1 /= (resG1 * np.sqrt(2 * np.pi))
        conv_I_resG0 = (g0 + g1 + resG0)
        conv_I_resG0 /= (np.pi * (X**2 + (g0 + g1 + resG0)**2))
        conv_I_resG1 = wofz((X + 1j * (g1 + g0)) / (resG1 * np.sqrt(2))).real
        conv_I_resG1 /= (resG1 * np.sqrt(2 * np.pi))


    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_G_resG0 = wofz((X + 1j * g0) / (resG0 * np.sqrt(2))).real
        conv_G_resG0 /= (resG0 * np.sqrt(2 * np.pi))
        conv_G_resG1 = wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
        conv_G_resG1 /= (resG1 * np.sqrt(2 * np.pi))
        conv_I_resG0 = wofz((X + 1j * (g1 + g0)) / (resG0 * np.sqrt(2))).real
        conv_I_resG0 /= (resG0 * np.sqrt(2 * np.pi))
        conv_I_resG1 = wofz((X + 1j * (g1 + g0)) / (resG1 * np.sqrt(2))).real
        conv_I_resG1 /= (resG1 * np.sqrt(2 * np.pi))

    else:
        print("The resolution function model used is not \
               supported by this model.\n \
               Please use either resFunc_pseudoVoigt or \
               resFunc_gaussian.\n")
        return



    model = (a0 * (resS * conv_G_resG0 + (1 - resS) * conv_G_resG1)
             + (1 - a0) * (resS * conv_I_resG0 + (1 - resS) * conv_I_resG1))


    model = beta * model + D2OSignal


    cost = np.sum(
        (dataset.data.intensities[scanIdx][dataset.data.qIdx] - model)**2
        / dataset.data.errors[scanIdx][dataset.data.qIdx]**2, axis=1)


    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)



    if returnCost:
        return cost

    elif returnSubCurves:
        res  = resFunc
        gLor = beta * a0 * (resS * conv_G_resG0 + (1 - resS) * conv_G_resG1)
        iLor = beta * (1 - a0) * (resS * conv_I_resG0
                                  + (1 - resS) * conv_I_resG1)

        return res, gLor, iLor

    else:
        return model





def protein_liquid_analytic_voigt_CF(X, params, dataset, D2OSignal=None,
                                     qIdx=None, scanIdx=slice(0, None),
                                     returnSubCurves=False):
    """ Fitting function for protein in liquid D2O environment.

        This makes uses of an analytic expression for convolution with
        resolution function, therefore resolution function used here should
        be a sum of two gaussians or pseudo-voigt.

        :arg params:          parameters for the model (described below),
                              usually given by scipy's routines
        :arg dataset:         dataset namedtuple containing x-axis values,
                              intensities, errors,...
        :arg D2OSignal:       D2O signal data as a 2D array with shape
                              (nbr q-values, S(q, :math:`\\omega`))
        :arg qIdx:            index of the current q-value
                              if None, a global fit is performed
                              over all q-values
        :arg scanIdx:         only for FWS (or QENS) data series,
                              index of the scan being fitted
        :arg returnSubCurves: if True, returns each individual component
                              of the model after convolution

    """

    params = np.array(params)

    g0      = params[0]     # global diffusion linewidth
    g1      = params[1]     # internal diffusion linewidth
    tau     = params[2]     # Residence time for jump diffusion model
    beta    = params[3]     # contribution factor of protein
    a0      = params[4]     # contribution factor of EISF


    if params.size != 5:
        betaSlice   = dataset.get_betaSlice()
        a0Slice     = dataset.get_a0Slice()
        beta        = params[betaSlice][:, np.newaxis]
        a0          = params[a0Slice][:, np.newaxis]


    X = dataset.data.X

    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis]

    resFunc = dataset.getResFunc()


    if D2OSignal is None:
        D2OSignal = dataset.getD2OSignal()


    if qIdx is None:
        g0 = qVals**2 * g0
        g1 = g1 * qVals**2 / (1 + g1 * qVals**2 * tau)



    # Computes the different components of the convolution
    resG0 = np.array([dataset.resData.params[qIdx][0][2]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resG1 = np.array([dataset.resData.params[qIdx][0][3]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resS  = np.array([dataset.resData.params[qIdx][0][1]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_G_resG0 = (g0 + resG0) / (np.pi * (X**2 + (g0 + resG0)**2))
        conv_G_resG1 = wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
        conv_G_resG1 /= (resG1 * np.sqrt(2 * np.pi))
        conv_I_resG0 = (g0 + g1 + resG0)
        conv_I_resG0 /= (np.pi * (X**2 + (g0 + g1 + resG0)**2))
        conv_I_resG1 = wofz((X + 1j * (g1 + g0)) / (resG1 * np.sqrt(2))).real
        conv_I_resG1 /= (resG1 * np.sqrt(2 * np.pi))


    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_G_resG0 = wofz((X + 1j * g0) / (resG0 * np.sqrt(2))).real
        conv_G_resG0 /= (resG0 * np.sqrt(2 * np.pi))
        conv_G_resG1 = wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
        conv_G_resG1 /= (resG1 * np.sqrt(2 * np.pi))
        conv_I_resG0 = wofz((X + 1j * (g1 + g0)) / (resG0 * np.sqrt(2))).real
        conv_I_resG0 /= (resG0 * np.sqrt(2 * np.pi))
        conv_I_resG1 = wofz((X + 1j * (g1 + g0)) / (resG1 * np.sqrt(2))).real
        conv_I_resG1 /= (resG1 * np.sqrt(2 * np.pi))

    else:
        print("The resolution function model used is not \
               supported by this model.\n \
               Please use either resFunc_pseudoVoigt \
               or resFunc_gaussian.\n")
        return



    model = (a0 * (resS * conv_G_resG0 + (1 - resS) * conv_G_resG1)
             + (1 - a0) * (resS * conv_I_resG0 + (1 - resS) * conv_I_resG1))


    model = beta * model + D2OSignal


    if qIdx is not None:
        model   = model[qIdx]



    if returnSubCurves:
        res  = resFunc
        gLor = beta * a0 * (resS * conv_G_resG0 + (1 - resS) * conv_G_resG1)
        iLor = beta * (1 - a0) * (resS * conv_I_resG0
                                  + (1 - resS) * conv_I_resG1)

        return res, gLor, iLor


    else:
        return model.flatten()





def protein_liquid_withImmobileFrac(params, dataset, D2OSignal=None,
                                    qIdx=None, returnCost=True,
                                    scanIdx=slice(0, None),
                                    returnSubCurves=False):
    """ Fitting function for protein in liquid D2O environment.

        This makes uses of an analytic expression for convolution
        with resolution function, therefore resolution function used
        here should be a sum of two gaussians or pseudo-voigt.

        :arg params:          parameters for the model (described below),
                              usually given by Scipy routines
        :arg dataset:         dataset namedtuple containing x-axis values,
                              intensities, errors,...
        :arg D2OSignal:       D2O signal data as a 2D array with shape
                              (nbr q-values, S(q, :math:`\\omega`) )
        :arg qIdx:            index of the current q-value
                              if None, a global fit is performed over all
                              q-values
        :arg returnCost:      if True, return the standard deviation of the
                              model to experimental data
                              if False, return only the model
        :arg scanIdx:         only for FWS (or QENS) data series, index of
                              the scan being fitted
        :arg returnSubCurves: if True, returns each individual component of
                              the model after convolution

    """


    g0      = params[0]     # Lorentzian linewidth
    g1      = params[1]     # Lorentzian linewidth
    tau     = params[2]     # Residence time for jump diffusion model
    a1      = params[3]     # Residence time for jump diffusion model
    beta    = params[4]     # scaling factor
    a0      = params[5]     # contribution factor for the Lorentzian


    if params.size != 6:
        betaSlice   = dataset.get_betaSlice()
        a0Slice     = dataset.get_a0Slice()
        beta        = params[betaSlice][:, np.newaxis]
        a0          = params[a0Slice][:, np.newaxis]


    X = dataset.data.X
    qVals = dataset.data.qVals[dataset.data.qIdx, np.newaxis]
    resFunc = dataset.getResFunc()


    if D2OSignal is None:
        D2OSignal = dataset.getD2OSignal()

    g0 = g0 * qVals**2
    g1 = g0 + g1 * qVals**2 / (1 + g1 * qVals**2 * tau)



    # Computes the different components of the convolution
    resG0 = np.array([dataset.resData.params[qIdx][0][2]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resG1 = np.array([dataset.resData.params[qIdx][0][3]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]
    resS  = np.array([dataset.resData.params[qIdx][0][1]
                      for qIdx in dataset.data.qIdx])[:, np.newaxis]

    if isinstance(dataset.resData, resFunc_pseudoVoigt.Model):
        conv_g0_resG0 = (g0 + resG0) / (np.pi * (X**2 + (g0 + resG0)**2))
        conv_g0_resG1 = wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
        conv_g0_resG1 /= (resG1 * np.sqrt(2 * np.pi))
        conv_g1_resG0 = (g1 + resG0) / (np.pi * (X**2 + (g1 + resG0)**2))
        conv_g1_resG1 = wofz((X + 1j * g1) / (resG1 * np.sqrt(2))).real
        conv_g1_resG1 /= (resG1 * np.sqrt(2 * np.pi))

    elif isinstance(dataset.resData, resFunc_gaussian.Model):
        conv_g0_resG0 = wofz((X + 1j * g0) / (resG0 * np.sqrt(2))).real
        conv_g0_resG0 /= (resG0 * np.sqrt(2 * np.pi))
        conv_g0_resG1 = wofz((X + 1j * g0) / (resG1 * np.sqrt(2))).real
        conv_g0_resG1 /= (resG1 * np.sqrt(2 * np.pi))
        conv_g1_resG0 = wofz((X + 1j * g1) / (resG0 * np.sqrt(2))).real
        conv_g1_resG0 /= (resG0 * np.sqrt(2 * np.pi))
        conv_g1_resG1 = wofz((X + 1j * g1) / (resG1 * np.sqrt(2))).real
        conv_g1_resG1 /= (resG1 * np.sqrt(2 * np.pi))

    else:
        print("The resolution function model used is not \
               supported by this model.\n \
               Please use either resFunc_pseudoVoigt \
               or resFunc_gaussian.\n")
        return



    model = beta * (a0 * resFunc + (1 - a0)
                    * (a1 * (resS * conv_g0_resG0
                             + (1 - resS) * conv_g0_resG1)
                    + (1 - a1) * (resS * conv_g1_resG0
                                  + (1 - resS) * conv_g1_resG1)))


    model = model + D2OSignal


    cost = np.sum(
        (dataset.data.intensities[scanIdx][dataset.data.qIdx] - model)**2
        / dataset.data.errors[scanIdx][dataset.data.qIdx]**2, axis=1)


    if qIdx is not None:
        cost    = cost[qIdx]
        model   = model[qIdx]
    else:
        cost = np.sum(cost)



    if returnCost:
        return cost

    elif returnSubCurves:
        res  = a0 * resFunc
        lor1 = (1 - a0) * a1 * (resS * conv_g0_resG0
                                + (1 - resS) * conv_g0_resG1)
        lor2 = (1 - a0) * (1 - a1) * (resS * conv_g1_resG0
                                      + (1 - resS) * conv_g1_resG1)
        return res, lor1, lor2

    else:
        return model
