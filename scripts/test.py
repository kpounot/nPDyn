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
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
import inxQENS_extract_resFunc_analysis as resAnalysis

def inxPlot(dataFile, binS, qMin, qMax):
    
    dataList = inxBinENS.inxBin(dataFile, binS)
    qMin = float(qMin)
    qMax = float(qMax)
    normFact = int(kargList['normFactor'])
    
    qList = list(map(float, [i.qVal for i in dataList]))
    q2List = [val**2 for val in qList]
    reducedqList = [val for val in qList if qMin < val < qMax]
    reducedIndex = [i for i, val in enumerate(qList) if qMin < val < qMax]

    if kargList['vanaNorm'] == 'True':
        message = QMessageBox.information(QWidget(), 'File selection',
                'Please select the file from which to extract the resolution function for: \n'
                + dataFile[dataFile.rfind('/'):])
        resDataList = resAnalysis.getDataList()

    #_Discard the selected index
    if kargList['qDiscard'] is not '':
        qDiscardPattern = re.compile(r'[ ,:;-]+')
        qDiscardList = qDiscardPattern.split(kargList['qDiscard'])
        qDiscardList = [min(qList, key = lambda x: abs(float(val) - x)) for val in qDiscardList]
        zipList = [(i + min(reducedIndex), val) for i, val in enumerate(reducedqList) 
                            if val not in qDiscardList] 
        reducedIndex, reducedqList = list(zip(*zipList))[0], list(zip(*zipList))[1]


    #_Plot of the intensities decay with temperature
    plt.figure()
    plt.axis([0, 300, qMin, qMax])

    normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
    cmap = matplotlib.cm.get_cmap('rainbow')

    for j, val in enumerate(dataList):
        if kargList['vanaNorm'] == 'True':
            plt.errorbar(dataList[0].temps,
                         dataList[j].intensities 
                              /np.mean(resDataList[j].intensities[np.abs(resDataList[j].energies) < 0.1]), 
                         fmt='-o', label='%.2f' % qList[j], c=cmap(normColors(qList[j])))   
        else:
            plt.errorbar(dataList[0].temps, 
                         dataList[j].intensities / np.mean(dataList[j].intensities[:normFact]),
                         #dataList[j].errors / np.mean(dataList[j].intensities[:normFact]),
                         fmt='-o', label='%.2f' % qList[j], c=cmap(normColors(qList[j])))   

    plt.grid()
    plt.legend(framealpha=0.5, loc='lower left')
    plt.xlabel('T (K)', fontsize=18)
    plt.ylabel(r'$S_{el, T}(q, 0)$', fontsize=24)
    plt.title('...' + dataFile[len(dataFile)-40:])
    plt.legend(framealpha=0.5, loc='lower left')
    plt.tight_layout()
    plt.show(block=False)

    #_Normalization of the data 
    normSList = []
    normEList = []
    normSList2 = []
    normEList2 = []
    if kargList['vanaNorm'] == 'True':
        for i, val in enumerate(dataList[0].temps):
            normSList.append([q.intensities[i]
                              /np.mean(resDataList[j].intensities[np.abs(resDataList[j].energies) < 0.1])
                              for j, q in enumerate(dataList)])
            normEList.append([q.errors[i]
                              /np.mean(resDataList[j].intensities[np.abs(resDataList[j].energies) < 0.1])
                              for j, q in enumerate(dataList)])
    else:
        for i, val in enumerate(dataList[0].temps):
            normSList.append([q.intensities[i]/np.mean(q.intensities[:normFact]) for q in dataList])
            normEList.append([q.errors[i]/np.mean(q.intensities[:normFact]) for q in dataList])
            normSList2.append([q.intensities[i]/np.mean(q.intensities[:2]) for q in dataList])
            normEList2.append([q.errors[i]/np.mean(q.intensities[:2]) for q in dataList])

    #_Definition of the functions used for the fit
    def fGauss(x, a, b): 
        return a * np.exp(-(1/6)*(x**2) * b)

    def fCorrGauss(x, a, b, c):
        return  a * np.exp(-(x**2) * b)*(1 + 1/72*(x**4)*c)

    #_Fit of the data S(q^2) with gaussian and store the tuples in fitGaussList
    fitGaussList = []
    fitGaussList2 = []
    plt.figure()
    for i, val in enumerate(dataList[0].temps):
       
        normIntensities = [normSList[i][j] for j in reducedIndex]
        sigma = [normEList[i][j] for j in reducedIndex]

        normIntensities2 = [normSList2[i][j] for j in reducedIndex]
        sigma2 = [normEList2[i][j] for j in reducedIndex]

        if i == 0:
            p0 = [1.0, 0.0]
        else:
            p0 = [fitGaussList[i-1][0][0], fitGaussList[i-1][0][1]]
        
        fitGaussList.append(optimize.curve_fit(fGauss, reducedqList, normIntensities,
                p0=p0, bounds=(-5, [4., 6.]), 
                loss='cauchy',
                sigma=sigma))
        fitGaussList2.append(optimize.curve_fit(fGauss, reducedqList, normIntensities2,
                p0=p0, bounds=(-5, [4., 6.]), 
                loss='cauchy',
                sigma=sigma2))


        #_Plot the normalized elastic curve for each temperature (y-axis)
        normColorsT = matplotlib.colors.Normalize(vmin=0, vmax=310)
        cmap = matplotlib.cm.get_cmap('jet')
        plt.errorbar(q2List, [j + val for j in normSList[i]], normEList[i], 
                     fmt='o', c=cmap(normColorsT(val)))

        #_Plot the fits to the data
        gaussianFit = [fGauss(qVal, fitGaussList[i][0][0], fitGaussList[i][0][1])
                       for qVal in qList]
        plt.plot(q2List, [fitVal + val for fitVal in gaussianFit], 'b-', label='fit')

        plt.grid()
        plt.axvspan(qMin**2, qMax**2, color='c', alpha=0.4)
        plt.xlabel(r'$q^{2} (\AA^{-2})$', fontsize=20)
        plt.ylabel(r'$S_{el, T}\ /\ S_{el, 10K} + T$', fontsize=24)
        plt.title('...' + dataFile[len(dataFile)-40:])
        plt.tight_layout()
    plt.show(block=False)

    return fitGaussList, fitGaussList2, dataList

def inxPlotMSD(dataFiles):

    #_Plot of the MSD calculated from the curve fit
    fitList = []
    fitList2 = []
    dataList = []
    for datafile in dataFiles:
        fit, fit2, data = inxPlot(datafile, kargList['binS'], 
                            kargList['qMin'], kargList['qMax'])
        fitList.append(fit)
        fitList2.append(fit2)
        dataList.append(data)

    plt.figure()
    for i, val in enumerate(fitList):
        plt.xlabel('T (K)')
        plt.ylabel(r'$MSD  (\AA^{2})\ q=%s - %s\ \AA^{-1}$' % 
                                (kargList['qMin'], kargList['qMax']))
        plt.errorbar(dataList[i][0].temps, [value[0][1] for  value in val],
                    [np.sqrt(np.diag(value[1]))[1] for value in val], 
                    fmt='-o',
                    label='...' + dataFiles[i][dataFiles[i].rfind('/'):])

        plt.errorbar(dataList[i][0].temps, [value[0][1] for  value in fitList2[i]],
                    [np.sqrt(np.diag(value[1]))[1] for value in fitList2[i]], 
                    fmt='-o',
                    label='...' + dataFiles[i][dataFiles[i].rfind('/'):])
    
    plt.grid()
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':


    app = QApplication(sys.argv)

    argList, kargList = argParser.argParser(sys.argv)
    inxPlotMSD(argList[1:]) 

    app.close()

    sys.exit(app.exec_())
