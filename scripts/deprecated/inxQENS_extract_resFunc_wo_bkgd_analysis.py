import sys
import numpy as np
import inxBinQENS
import argParser
import re
import matplotlib.pyplot as plt
from collections import namedtuple
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sympy.functions.special.delta_functions import DiracDelta
from scipy import optimize
from scipy.signal import fftconvolve
from scipy.special import wofz, sph_jn
from scipy.stats import chisquare, bayes_mvs
from scipy.misc import factorial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib


def getDataList():

    arg, karg = argParser.argParser(sys.argv)
    fileToOpen = QFileDialog().getOpenFileName()[0]
    try:
        resBinStep = int(karg['resBinStep'])
    except KeyError:
        resBinStep = 5

    #_Get data from the file and store them into self.dataList 
    dataList = inxBinQENS.inxBin(fileToOpen, resBinStep)

    #_Keep the data from wanted q-range
    qMin = min([data.qVal for data in dataList], 
                    key = lambda x : abs(float(karg['qMin']) - x))
    qMax = min([data.qVal for data in dataList], 
                    key = lambda x : abs(float(karg['qMax']) - x))
    dataList = [val for val in dataList if qMin <= val.qVal <= qMax]

    #_Discard the selected index
    if karg['qDiscard'] is not '':
        qDiscardPattern = re.compile(r'[ ,:;-]+')
        qDiscardList = qDiscardPattern.split(karg['qDiscard'])
        qDiscardList = [min([data.qVal for data in dataList], 
                    key = lambda x: abs(float(val) - x)) for val in qDiscardList]
        dataList = [val for val in self.dataList 
                            if val.qVal not in qDiscardList] 
    
    return dataList


def getFitList():

    arg, karg = argParser.argParser(sys.argv)
    dataList = getDataList()


    try:
        meanLowQ = float(karg['meanLowQ'])
    except KeyError:
        meanLowQ = 0.6
    try:
        meanHighQ = float(karg['meanHighQ'])
    except KeyError:
        meanHighQ = 1.8


    resFitList = resFit(dataList)
    meanWidth = []
    meanWidth.append(np.mean([resFitList[j][0][1] 
                            for j, val in enumerate(resFitList) 
                                if meanLowQ <= dataList[j].qVal <= meanHighQ]))

    return resFitList, meanWidth, dataList

#_Everything needed for the fit
def resFunc(x, lorG, gauG, lorS, normF, shift):

    return  (normF * (lorS * lorG/(lorG**2 + (x-shift)**2) /np.pi 
            + (1-lorS) * np.exp(-((x-shift)**2) / (2*gauG**2)) / (gauG*np.sqrt(2*np.pi))))  
            
def resFit(dataList):

    resFitList = []
    for j, data in enumerate(dataList):
        resFitList.append(optimize.curve_fit(resFunc, 
                        data.energies,
                        data.intensities,
                        sigma=[val + 0.0001 for val in data.errors],
                        #p0 = [0.5, 1, 0.8, 50, 0], 
                        bounds=([0., 0., 0., 0., -10],  
                                [4, 10, 1., 5000, 10]),
                        max_nfev=10000000,
                        method='trf'))

    return resFitList    

        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    sys.exit(app.exec_())
