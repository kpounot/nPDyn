import sys, os, pickle as pk
import numpy as np
import argParser
import re

from collections import namedtuple

from scipy import optimize

import h5process
import h5py as h5

 
class ECCorr():
    def __init__(self, ECFile):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.ECFitList = []

        #_Get datas from the file and store them into dataList 
        self.dataList = h5process.processData(self.dataFiles, karg['binS'], FWS=False)

        ECFitting()



#_Everything needed for the fit
    def fitFunc(self, x, normF, S, lorW, gauW, shift, bkgd):

        return  (normF * (S * lorW/(lorW**2 + (x-shift)**2) /np.pi 
                + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi))
                + bkgd))  
                
    def ECFitting(self):
    
        for i, ECData in enumerate(self.dataList):
            fitList = []
            for j, qWiseData in enumerate(ECData.intensities):
                fitList.append(optimize.curve_fit(self.fitFunc, 
                                ECData.energies,
                                ECData.intensities[j],
                                sigma=[val+0.0001 for val in ECData.errors[j]],
                                #p0 = [0.5, 1, 0.8, 50, 0], 
                                bounds=([0., 0., 0., 0., -10, 0.],  
                                        [2000, 1, 5, 5, 10, 0.5]),
                                max_nfev=10000000,
                                method='trf'))

            self.ECFitList.append(fitList)

    def getECCurves(self, scaleFactor):

        ECCurve = np.zeros_like(self.dataList[0].intensities)
        for qIdx, qVal in enumerate(self.dataList[0]):
            ECCurve[qIdx] = self.fitFunc(self.dataList[0].energies, *self.ECFitList[0][qIdx].x)

        return scaleFactor * ECCurve

