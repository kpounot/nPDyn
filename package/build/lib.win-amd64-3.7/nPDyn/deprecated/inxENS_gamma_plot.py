'''
This script use the Laplace transform of a Gamma distribution to fit the data from
fixed window elastic scans of neutron diffraction.
Different plots are generated and in the gammaModel function, you can choose 
either a standard Gamma model or a shifted one by commenting the right line.

References:
-   Judith Peters and Gerald R. Kneller (2013) Motional heterogeneity in human 
    acetylcholinesterase revealed by a non-Gaussian model for elastic incoherent 
    neutron scattering. The Journal of Chemical Physics 139, 165102

-   Gerald R. Kneller and Konrad Hinsen (2009) Quantitative model for the 
    heterogeneity of atomic position fluctuations in proteins: A simulation study.
    The Journal of Chemical Physics, 131, 045104
'''

import sys
import numpy as np
import inxBinENS
import argParser
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    reducedqList = [val for val in qList if qMin < val < qMax]
    reducedIndex = [i for i, val in enumerate(qList) if qMin < val < qMax]

    #_Discard the selected index
    if kargList['qDiscard'] is not '':
        qDiscardPattern = re.compile(r'[ ,:;-]+')
        qDiscardList = qDiscardPattern.split(kargList['qDiscard'])
        qDiscardList = [min(qList, key = lambda x: abs(float(val) - x)) for val in qDiscardList]
        zipList = [(i + min(reducedIndex), val) for i, val in enumerate(reducedqList) 
                            if val not in qDiscardList] 
        reducedIndex, reducedqList = list(zip(*zipList))[0], list(zip(*zipList))[1]


    #_Plot of the intensities decay with temperature
    normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
    cmap = matplotlib.cm.get_cmap('rainbow')
    plt.figure()
    plt.axis([0, 320, qMin, qMax])
    for j, val in enumerate(dataList):
        plt.errorbar(dataList[0].temps, 
                     list(map(lambda x: x/np.mean(dataList[j].intensities[:normFact]), 
                                            dataList[j].intensities)), 
                     list(map(lambda x: x/np.mean(dataList[j].intensities[:normFact]), 
                                            dataList[j].errors)), 
                     fmt='-o', label='%.2f' % qList[j], c=cmap(normColors(qList[j])))   

    plt.grid()
    plt.legend(framealpha=0.5, loc='lower left')
    plt.xlabel('T (K)', fontsize=18)
    plt.ylabel(r'$S_{el, T}(q, 0)$', fontsize=24)
    plt.title('...' + dataFile[len(dataFile)-40:])
    plt.tight_layout()
    plt.show(block=False)

    #_Extract the intensities and errors for the same temperature depending on q values 
    normSList = []
    normEList = []
    for i, val in enumerate(dataList[0].temps):
        normSList.append([q.intensities[i]/np.mean(q.intensities[:normFact]) for q in dataList])
        normEList.append([q.errors[i]/np.mean(q.intensities[:normFact]) for q in dataList])

    #_Definition of the functions used for the fit
    def gammaModel(x, b, msd, a): 

        #return np.exp(-shift*(x*msd)**2) / (1 + ((x*msd)**2/b))**b  #_Shifted Gamma distribution
        return a / (1 + ((x*msd)**2/b))**b  #_Standard Gamma distribution

    #_Fit of the data S(q^2) with gaussian and store the tuples in fitGaussList
    fitGaussList = []
    plt.figure()
    for i, val in enumerate(dataList[0].temps):
       
        normIntensities = [normSList[i][j] for j in reducedIndex]
        normSigma = [normEList[i][j] for j in reducedIndex]
        fitGaussList.append(optimize.curve_fit(gammaModel, reducedqList, normIntensities,
                                               p0 = [8, 0.5, 1], 
                                               bounds=([0., 0., 0.], [20, 4, 5]), 
                                               sigma=normSigma, 
                                               #loss='cauchy',
                                               method='trf'))

        print('Parameters for T = %.2f' % val, flush=True)
        print('    {0:<5}= {1:.3f}'.format('MSD', fitGaussList[i][0][1]), flush=True)
        #print('    {0:<5}= {1:.3f}'.format('Shift', fitGaussList[i][0][2]), flush=True)
        print('    {0:<5}= {1:.3f} \n'.format('Beta', fitGaussList[i][0][0]), flush=True)

        #_Plots parameters
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

        #_Plot the fit to the data
        gammaFit = [gammaModel(qVal, *fitGaussList[i][0][:])
                       for qVal in qList]
        plt.plot(q2List, [fitVal + val for fitVal in gammaFit], 'b-', label='fit')
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

    figure = plt.figure()
    mplGrid = gridspec.GridSpec(1, 2)
    for i, val in enumerate(fitList):

        #_Plot of the fitted MSD
        ax = figure.add_subplot(mplGrid[:,0])
        ax.grid(True)
        ax.set_xlabel('T (K)', fontsize=20)
        ax.set_ylabel(r'$MSD  (\AA^{2})$', fontsize=20)
        ax.errorbar(dataList[i][0].temps,
                    [value[0][1]**2 for value in val],
                    #[np.sqrt(np.diag(value[1])[1]) for value in val], 
                    fmt='-o')
        ax.legend(['...' + dataFile[dataFile.rfind('/'):] for dataFile in dataFiles], 
                  framealpha=0.5)

        #_Plot of the beta parameter
        ax = figure.add_subplot(mplGrid[:,1])
        ax.grid(True)        
        ax.set_xlabel('T (K)', fontsize=20)
        ax.set_ylabel(r'$\beta \ (a.u.)$', fontsize=20)
        ax.set_ylim(0, 20)
        ax.plot(dataList[i][0].temps, [value[0][0] for value in val], ls='--')
        ax.legend(['...' + dataFile[dataFile.rfind('/'):] for dataFile in dataFiles], 
                  framealpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    argList, kargList = argParser.argParser(sys.argv)
    inxPlotMSD(argList[1:]) 

