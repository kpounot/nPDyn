import numpy as np
from collections import namedtuple
from scipy.signal import fftconvolve


def gaussian(x, scaleF, msd):
    """ This function can be used to fit elastic data. It makes use of simple gaussian approximation.

        Input:  scaleF      -> scaling factor
                msd         -> mean-squared displacement """

    return ( scaleF * np.exp( -(1/6) * x**2 * msd ) )


def q4_corrected_gaussian(x, scaleF, msd, sigma):
    """ This function can be used to fit elastic data. It makes use of a corrected gaussian using the 
        next term in autocorrelation function expansion.

        Input:  scaleF      -> scaling factor
                msd         -> mean-squared displacement
                sigma       -> extra correction term for q**4 contribution """

    return scaleF * np.exp( -x**2 * msd ) * (1 + 1/72 * (x**4) * sigma)
