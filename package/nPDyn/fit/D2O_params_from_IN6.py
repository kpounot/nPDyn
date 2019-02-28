import sys, os
import numpy as np
import scipy.interpolate as sii

def getD2Odata(volFraction=0.95):
    """ This script extract D2O background signal from IN6 measurements data.
        Then the D2O signal intensity is weighted by the given volume fraction of it in sample solution 
        with default value being one for pure D2O.
        Eventually, linear interpolation is performed and the function is returned. """


    filePath = os.path.abspath(__file__)

    #_Platform dependent way of return path
    if 'win32' in sys.platform: 
        dataPath = filePath[:filePath.rfind('\\')] + "/D2O_data/"
    else:
        dataPath = filePath[:filePath.rfind('/')] + "/D2O_data/"



    #_D2O data from IN6 - width of "main" (narrow) Lorentzian created based on MATLAB D2O function:
    D2O_Table = np.loadtxt( dataPath + 'D2O_interpolation_table.dat')
    D2O_T     = np.loadtxt( dataPath + 'D2O_temperatures.dat')
    D2O_q     = D2O_Table[:,0] # np.loadtxt( datapath+'q_vector_D2O_table.dat')
    D2O_S     = volFraction * D2O_Table[:,1:5]

    #_Interpolation based on extracted arrays
    sD2O     = sii.interp2d(D2O_T, D2O_q, D2O_S, kind='linear') 

    return sD2O