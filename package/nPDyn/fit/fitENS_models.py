import numpy as np


def gaussian(x, msd, shift):
    """ This function can be used to fit elastic data.
        It makes use of simple gaussian approximation.

        :arg msd:    mean-squared displacement
        :arg shift:  shift from 0 in abscissa 

    """

    return  (np.sqrt(6 / (np.pi * msd))
            *np.exp(-(1 / 6) * (x - shift)**2 * msd))



def linearMSD(x, msd, y0):
    """ This function can be used to fit elastic data to
        exctract the msd.
        
        The elastic data are assumed to have been linearized.

        :arg msd:   mean-square displacement
        :arg y0:    intercept in the y-axis

    """

    return y0 - msd * x



def q4_corrected_gaussian(x, msd, sigma, shift):
    """ This function can be used to fit elastic data.
        It makes use of a corrected gaussian using the
        next term in autocorrelation function expansion.

        :arg msd:    mean-squared displacement
        :arg sigma:  extra correction term for q**4 contribution
        :arg shift:  shift from 0 in abscissa 

    """

    x = (x - shift)

    return np.exp(-x**2 * msd) * (1 + 1 / 72 * (x**4) * sigma**2)





def gamma(x, msd, beta, shift):
    """ This function can be used to fit elastic data.
        It makes use of the Gamma distribution approach
        developped by Kneller and Hinsen.

        :arg msd:    mean-squared displacement
        :arg beta:   beta parameter that represents motion
                     homogeneity in the sample
        :arg shift:  shift from 0 in abscissa 

    """

    return 1 / (1 + (msd * (x - shift)**2) / beta)**(beta)
