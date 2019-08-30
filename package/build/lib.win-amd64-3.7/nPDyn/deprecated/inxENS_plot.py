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
        message.done(0)
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
    plt.xlabel('T (K)', fontsize=18)
    plt.xlim(0, 320)
    plt.ylabel(r'$S_{el, T}(q, 0)$', fontsize=24)
    plt.ylim(0, 1.2)
    plt.title('...' + dataFile[dataFile.rfind('/'):])
    plt.legend(framealpha=0.5, loc='lower left')
    plt.tight_layout()
    plt.show(block=False)


    #_Normalization of data 
    normSList = []
    normEList = []
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

    #_Definition of the functions used for the fit
    def fGauss(x, a, b): 
        return a * np.exp(-(1/6)*(x**2) * b)

    def fCorrGauss(x, a, b, c):
        return  a * np.exp(-(x**2) * b)*(1 + 1/72*(x**4)*c)

    #_Fit of the data S(q^2) with gaussian and store the tuples in fitGaussList
    fitGaussList = []
    plt.figure()
    for i, val in enumerate(dataList[0].temps):
       
        normIntensities = [normSList[i][j] for j in reducedIndex]
        sigma = [normEList[i][j] for j in reducedIndex]

        if i == 0:
            p0 = [1.0, 0.0]
        else:
            p0 = fitGaussList[i-1][0][:]

        fitGaussList.append(optimize.curve_fit( fGauss, reducedqList, normIntensities,
                                                p0=p0, 
                                                bounds=(0., [10., 4.]), 
                                                sigma=sigma
                                                ))

        plt.grid()
        plt.axvspan(qMin**2, qMax**2, color='c', alpha=0.4)
        plt.xlabel(r'$q^{2} (\AA^{-2})$', fontsize=20)
        plt.ylabel(r'$S_{el, T}\ /\ S_{el, 10K} + T$', fontsize=24)
        plt.title('...' + dataFile[len(dataFile)-40:])
        plt.tight_layout()


        #_Plot the normalized elastic curve for each temperature (y-axis)
        normColorsT = matplotlib.colors.Normalize(vmin=0, vmax=310)
        cmap = matplotlib.cm.get_cmap('jet')
        plt.errorbar(q2List, [j + val for j in normSList[i]], normEList[i], 
                     fmt='o', c=cmap(normColorsT(val)))

        #_Plot the fits to the data
        gaussianFit = [fGauss(qVal, *fitGaussList[i][0][:])
                       for qVal in qList]
        plt.plot(q2List, [fitVal + val for fitVal in gaussianFit], 'b-', label='fit')
    plt.show(block=False)

    return fitGaussList, dataList

def inxPlotMSD(dataFiles):

    #_Plot of the MSD calculated from the curve fit
    fitList = []
    dataList = []


    for idx, datafile in enumerate(dataFiles):
        fit, data = inxPlot(datafile, kargList['binS'], 
                            kargList['qMin'], kargList['qMax'])
        fitList.append(fit)
        dataList.append(data)

    #_Plot the total scattering
    plt.figure()
    for idx, datafile in enumerate(dataFiles):

        qList = list(map(float, [i.qVal for i in dataList[idx]]))
        reducedIndex = [i for i, val in enumerate(qList) if float(kargList['qMin']) < val < 
                                                                                float(kargList['qMax'])]
        reducedData = [dataList[idx][k] for k in reducedIndex]
        summedIntensities = [0 for i in reducedData[idx].intensities]
        for i, val in enumerate(dataList[idx]):
            summedIntensities += val.intensities 

        summedErrors = [0 for i in dataList[idx][0].errors]
        for i, val in enumerate(dataList[idx]):
            summedErrors += val.errors 
        
        plt.errorbar(   dataList[idx][0].temps, 
                        summedIntensities / np.mean(summedIntensities[:int(kargList['normFactor'])]), 
                        summedErrors / np.mean(summedIntensities[:int(kargList['normFactor'])]),
                        label='...' + datafile[datafile.rfind('/'):],
                        fmt='-o')
    plt.grid()
    plt.xlabel('T (K)', fontsize=18)
    plt.ylabel('Total elastic scattering', fontsize=18)
    plt.legend(loc='lower left')
    plt.show(block=False)


    #_Plot the MSD
    plt.figure()
    colors = ['green', 'coral']
    for i, val in enumerate(fitList):
        plt.xlabel('T (K)', fontsize=14)
        plt.ylabel(r'$MSD  (\AA^{2})\ q=%s - %s\ \AA^{-1}$' % 
                                (kargList['qMin'], kargList['qMax']), fontsize=14)

        reducedTList = dataList[i][0].temps[:np.where(dataList[i][0].temps > 295)[0][0]]
        plt.errorbar(reducedTList, 
                    [value[0][1] for i, value in enumerate(val) if i < reducedTList.size],
                    #[np.sqrt(np.diag(value[1]))[1] for i, value in enumerate(val) if i < reducedTList.size], 
                    fmt='-o',
                    label='...' + dataFiles[i][dataFiles[i].rfind('/'):],
                    color=colors[i])
    
    plt.tick_params(labelsize=14)
    #plt.grid()
    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':


    app = QApplication(sys.argv)

    argList, kargList = argParser.argParser(sys.argv)
    inxPlotMSD(argList[1:]) 


