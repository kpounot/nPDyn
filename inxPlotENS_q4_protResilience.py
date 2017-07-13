'''
This script use a gaussian model corrected to Q**4 to fit the data from
fixed window elastic scans of neutron diffraction.

References:
-   Zheng Yi, Yinglong Miao, Jerome Baudry et al. (2012) Derivation of Mean-Square 
    Displacements for Protein Dynamics from Elastic Incoherent Neutron Scattering.
    J. Phys. Chem. B, 116, 5028-5036

-   Zaccai Giuseppe (2000) How soft is a protein? A protein dynamics force constant 
    measured by neutron scattering. Science, 288. 
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
from scipy.stats import linregress

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
    plt.legend(framealpha=0.5, loc='lower left')
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

    def fCorrGauss(x, a, b, c):
        return  a * np.exp(-(x**2) * b)*(1 + 1/72*(x**4)*c)

    #_Fit of the data S(q^2) with gaussian and store the tuples in fitGaussList
    fitGaussList = []
    plt.figure()
    for i, val in enumerate(dataList[0].temps):
       
        normIntensities = [normSList[i][j] for j in reducedIndex]
        sigma = [normEList[i][j] for j in reducedIndex]
        fitGaussList.append(optimize.curve_fit(fCorrGauss, reducedqList, normIntensities,
                p0 = [1.0, 0.5, 4.0], bounds =(0. , [2., 4., 50.]), 
                #sigma=sigma,
                #loss='cauchy',
                method='trf'))

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
        gaussianFit = [fCorrGauss(qVal, fitGaussList[i][0][0], 
                       fitGaussList[i][0][1], fitGaussList[i][0][2]) for qVal in qList]
        plt.plot(q2List, [fitVal + val for fitVal in gaussianFit], 'b-', label='fit')
    plt.show(block=False)

    return fitGaussList, dataList

def inxPlotMSD(dataFiles):

    fitList = []
    dataList = []
    for datafile in dataFiles:
        fit, data = inxPlot(datafile, kargList['binS'], 
                            kargList['qMin'], kargList['qMax'])
        fitList.append(fit)
        dataList.append(data)

    #_Parameters for spring constant fitting
    resilTMin = float(kargList['resilTMin'])
    resilTMax = float(kargList['resilTMax'])
    fitFunc = lambda x, k, c: 0.00138/k * x + c

    plt.figure()
    for i, val in enumerate(fitList):

        #_Fit of the spring constant above dynamical transition - protein resilience
        fitTRange = [value for value in dataList[i][0].temps if resilTMin < value < resilTMax]
        fitYdata = [val[j][0][1] for j, value in enumerate(dataList[i][0].temps) 
                    if resilTMin < value < resilTMax]
        fitYerrors = [np.sqrt(np.diag(val[j][1]))[1] for j, value 
                      in enumerate(dataList[i][0].temps) if resilTMin < value < resilTMax]
        popt, pcov = optimize.curve_fit(fitFunc, fitTRange, fitYdata, 
                                        sigma=fitYerrors, method='trf', loss='cauchy')

        #_Plot of the MSD calculated from the first curve fit
        plt.grid()
        plt.xlabel('T (K)')
        plt.ylabel(r'$MSD  (\AA^{2})$')
        plt.errorbar(dataList[i][0].temps, [value[0][1] for value in val],
                    [np.sqrt(np.diag(value[1]))[1] for value in val], fmt='-o',
                    label=dataFiles[i][dataFiles[i].rfind('/'):])

        #_Plot of the fitted spring constant
        fitYList = [fitFunc(x, popt[0], popt[1]) for x in dataList[i][0].temps]
        plt.plot(dataList[i][0].temps, fitYList, lw=4, ls='--',
                 label=r'$\langle k \rangle = %.3f \ \pm \ %.3f \ N/m$' 
                            % (popt[0], np.sqrt(np.diag(pcov))[0])) 

    plt.grid()
    plt.ylim(-0.5, 2.5)
    plt.legend(framealpha=0.5, loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':


    argList, kargList = argParser.argParser(sys.argv)
    try:
        resilTMin = kargList['resilTMin'] 
        resilTMax = kargList['resilTMax']
    
        if resilTMin > resilTMax:
            raise ValueError('ERROR !\n The resilTMin value is greater than resilTMax.')

        inxPlotMSD(argList[1:]) 

    except KeyError:
        print('ERROR !\n' +
              'This script needs additional parameters resilTMin and resilTMax\n' + 
              'to be defined by the user in the corresponding text box !')

    except ValueError as error:
        print(error)

