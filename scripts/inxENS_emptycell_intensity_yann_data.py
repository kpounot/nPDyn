'''
This script use a gaussian distribution to fit the data from
fixed window elastic scans of neutron diffraction.

References:
-   Zheng Yi, Yinglong Miao, Jerome Baudry et al. (2012) Derivation of Mean-Square 
    Displacements for Protein Dynamics from Elastic Incoherent Neutron Scattering.
    J. Phys. Chem. B, 116, 5028-5036
'''

import sys
import numpy as np
import inxBinENS
import argParser
import re
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple
from scipy import optimize
from PyQt5.QtWidgets import QApplication, QFileDialog
import inxQENS_extract_resFunc_analysis as resAnalysis

def getMeanIntensity():
    
    argList, kargList = argParser.argParser(sys.argv)

    dataFile = QFileDialog().getOpenFileName()[0] 

    dataList = inxBinENS.inxBin(dataFile, 1)

    qMin = float(kargList['qMin'])
    qMax = float(kargList['qMax'])

    qList = list(map(float, [i.qVal for i in dataList]))
    reducedqList = [val for val in qList if qMin < val < qMax]
    reducedIndex = [i for i, val in enumerate(qList) if qMin < val < qMax]

    dataList = [dataList[i] for i in reducedIndex]


    meanIntensities = [np.mean(qData.intensities) for qData in dataList]

    return meanIntensities




if __name__ == '__main__':

    argList, kargList = argParser.argParser(sys.argv)

     

