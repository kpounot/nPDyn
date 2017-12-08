'''
This script use a corrected gaussian distribution to fit the data from
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

def inxPlot(dataFile, binS, qMin, qMax):
    
    dataList = inxBinENS.inxBin(dataFile, binS)
    qMin = float(qMin)
    qMax = float(qMax)
    normFact = int(kargList['normFactor'])
    
    qList = list(map(float, [i.qVal for i in dataList]))
    q2List = [val**2 for val in qList]
    reducedqList = np.array([val for val in qList if qMin < val < qMax])
    reducedIndex = np.array([i for i, val in enumerate(qList) if qMin < val < qMax])

    #_Discard the selected index
    if kargList['qDiscard'] is not '':
        qDiscardPattern = re.compile(r'[ ,:;-]+')
        qDiscardList = qDiscardPattern.split(kargList['qDiscard'])
        qDiscardList = [min(qList, key = lambda x: abs(float(val) - x)) for val in qDiscardList]
        zipList = [(i + min(reducedIndex), val) for i, val in enumerate(reducedqList) 
                            if val not in qDiscardList] 

  
    #_Plot of the intensities decay with temperature
    plt.figure()
    plt.axis([0, 300, qMin, qMax])

    normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
    cmap = matplotlib.cm.get_cmap('rainbow')
    for j, val in enumerate(dataList):
        plt.errorbar(dataList[0].temps, 
                     list(map(lambda x: x/dataList[j].intensities[0], dataList[j].intensities)), 
                     list(map(lambda x: x/dataList[j].intensities[0], dataList[j].errors)), 
                     fmt='-o', label='%.2f' % qList[j], c=cmap(normColors(qList[j])))   

    plt.grid()
    plt.xlabel('T (K)', fontsize=18)
    plt.ylabel(r'$S_{el, T}(q, 0)$', fontsize=24)
    plt.title('...' + dataFile[len(dataFile)-40:])
    plt.legend(framealpha=0.5, loc='lower left')
    plt.tight_layout()
    plt.show(block=False)

    #_Extract the intensities and errors for the same temperature depending on q values 
    normSList = []
    normEList = []
    for i, val in enumerate(dataList[0].temps):
        normSList.append([q.intensities[i]/np.mean(q.intensities[:normFact]) for q in dataList])
        normEList.append([q.errors[i]/np.mean(q.intensities[:normFact]) for q in dataList])

    #_Definition of the functions used for the fit
    def fGauss(x, a, b): 
        return a * np.exp(-(1/6)*(x**2) * b)

    def cost_func(x, dataI, dataErr):
        return np.sum((dataI - fGauss(reducedqList, *x))**2 / dataErr)

    #_Fit of the data S(q^2) with gaussian and store the tuples in fitGaussList
    fitGaussList = []
    plt.figure()
    for i, val in enumerate(dataList[0].temps):
       
        normIntensities = np.array([normSList[i][j] for j in reducedIndex])
        sigma = np.array([normEList[i][j] for j in reducedIndex])
        bounds = [(0., 10), (0., 4)]
        fitGaussList.append(optimize.basinhopping(cost_func,
                                                  x0=[1, 0.5],
                                                  disp=True,
                                                  minimizer_kwargs={'args':(normIntensities, sigma),
                                                                    'bounds':bounds}))

        plt.grid()
        plt.axvspan(qMin**2, qMax**2, color='c', alpha=0.4)
        plt.xlabel(r'$q^{2} (\AA^{-2})$', fontsize=20)
        plt.ylabel(r'$S_{el, T}\ /\ S_{el, 10K} + T$', fontsize=24)
        plt.tight_layout()
        plt.title('...' + dataFile[len(dataFile)-40:])

        #_Plot the normalized elastic curve for each temperature (y-axis)
        normColorsT = matplotlib.colors.Normalize(vmin=0, vmax=310)
        cmap = matplotlib.cm.get_cmap('jet')
        plt.errorbar(q2List, [j + val for j in normSList[i]], normEList[i], 
                     fmt='o', c=cmap(normColorsT(val)))

        #_Plot the fit to the data
        gaussianFit = [fGauss(qVal, *fitGaussList[i].x) for qVal in qList]
        plt.plot(q2List, [fitVal + val for fitVal in gaussianFit], 'b-', label='fit')
    plt.show(block=False)

    return fitGaussList, dataList

def inxPlotMSD(dataFiles):

    #_Plot of the MSD calculated from the curve fit
    fitList = []
    dataList = []
    for datafile in dataFiles:
        fit, data = inxPlot(datafile, kargList['binS'], 
                            kargList['qMin'], kargList['qMax'])
        fitList.append(fit)
        dataList.append(data)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i, val in enumerate(fitList):
        ax.set_xlabel('T (K)')
        ax.set_ylabel(r'$MSD  (\AA^{2})$')
        ax.errorbar(dataList[i][0].temps, [value.x[1] for  value in val],
                    #[np.sqrt(np.diag(value[1]))[1] for value in val], 
                    fmt='-o',
                    label='...' + dataFiles[i][dataFiles[i].rfind('/'):])
    
    plt.grid()
    qRangeText = r'$q-range: %s - %s \ \AA^{-1}$' % (kargList['qMin'], kargList['qMax'])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(matplotlib.patches.Patch(color='none', label=qRangeText))
    plt.legend(handles=handles, loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    argList, kargList = argParser.argParser(sys.argv)
    inxPlotMSD(argList[1:]) 

