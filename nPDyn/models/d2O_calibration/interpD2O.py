import sys
import os
import numpy as np
import scipy.interpolate as sii


def getD2Odata(volFraction=0.95):
    """This script extract D2O background signal from measurements performed
    on IN6 at the ILL.

    The D2O signal intensity is weighted by the given
    volume fraction in sample solution ('volFraction).
    A linear interpolation is performed
    and the function is returned.

    """
    path = os.path.dirname(os.path.abspath(__file__))

    # D2O data from IN6 - width of "main" (narrow) Lorentzian
    # created based on MATLAB D2O function:
    D2O_Table = np.loadtxt(path + "/D2O_interpolation_table.dat")
    D2O_T = np.loadtxt(path + "/D2O_temperatures.dat")
    D2O_q = D2O_Table[:, 0]
    D2O_S = volFraction * D2O_Table[:, 1:5]

    # Interpolation based on extracted arrays
    sD2O = sii.interp2d(D2O_T, D2O_q, D2O_S, kind="linear")

    return sD2O
