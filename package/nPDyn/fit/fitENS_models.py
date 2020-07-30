import numpy as np


def gaussian(x, msd, scaleF):
    """ This function can be used to fit elastic data.
        It makes use of simple gaussian approximation.

        :arg msd:    mean-squared displacement
        :arg scaleF: scaling factor

    """

    return scaleF * np.exp(-(1 / 6) * x**2 * msd)




def q4_corrected_gaussian(x, msd, sigma, scaleF):
    """ This function can be used to fit elastic data.
        It makes use of a corrected gaussian using the
        next term in autocorrelation function expansion.

        :arg msd:    mean-squared displacement
        :arg sigma:  extra correction term for q**4 contribution
        :arg scaleF: scaling factor

    """

    return scaleF * np.exp(-x**2 * msd) * (1 + 1 / 72 * (x**4) * sigma)





def gamma(x, msd, beta, scaleF):
    """ This function can be used to fit elastic data.
        It makes use of the Gamma distribution approach
        developped by Kneller and Hinsen.

        :arg msd:    mean-squared displacement
        :arg beta:   beta parameter that represents motion
                     homogeneity in the sample
        :arg scaleF: scaling factor

    """

    return scaleF / (1 + (msd * x**2) / beta)**(beta)
