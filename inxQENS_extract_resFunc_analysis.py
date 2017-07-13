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

    filesToOpen = QFileDialog().getOpenFileNames()[0]

    dataFiles = filesToOpen
    dataList = []

    arg, karg = argParser.argParser(sys.argv)

    for dataFile in dataFiles:
        #_Get datas from the file and store them into self.dataList 
        inxDatas = inxBinQENS.inxBin(dataFile, karg['resBinStep'])
        dataList.append(inxDatas)    


    resFitList = resFit(dataList)
    meanWidth = []
    for k, fileDatas in enumerate(dataList):
        meanWidth.append(np.mean([resFitList[k][j][0][1] 
                            for j, val in enumerate(fileDatas) 
                                if dataList[k][j].qVal >= 0.6]))

    return resFitList, meanWidth

#_Everything needed for the fit
def resFunc(x, lorG, gauG, lorS, normF, shift, bkgd):

    return  (normF * (lorS * lorG/(lorG**2 + (x-shift)**2) /np.pi 
            + (1-lorS) * np.exp(-((x-shift)**2) / (2*gauG**2)) / (gauG*np.sqrt(2*np.pi))
            + bkgd))  
            
def resFit(dataList):

    resFitList = []
    for i, resFile in enumerate(dataList):
        resList = []
        for j, datas in enumerate(resFile):
            resList.append(optimize.curve_fit(resFunc, 
                            datas.energies,
                            datas.intensities,
                            sigma=[val + 0.0001 for val in datas.errors],
                            #p0 = [0.5, 1, 0.8, 50, 0], 
                            bounds=([0., 0., 0., 0., -10, 0.],  
                                    [2, 10, 1., 5000, 10, 1]),
                            max_nfev=1000000,
                            method='trf'))
        resFitList.append(resList)

    return resFitList    

        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    sys.exit(app.exec_())
