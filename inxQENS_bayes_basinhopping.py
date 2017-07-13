'''
This script is used to perform a bayesian analysis of the data and thus, determine the best
number of Lorentzians to be used to fit them. The 'fit' button use automaticaly the best
model found for the fit.

Reference :
-   D.S. Sivia, S König, C.J. Carlile and W.S. Howells (1992) Bayesian analysis of 
    quasi-elastic neutron scattering data. Physica B, 182, 341-348 
'''

import sys
import numpy as np
import inxBinQENS
import argParser
import re
import matplotlib.pyplot as plt
from collections import namedtuple
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame, QMessageBox)
from PyQt5 import QtGui
import mpl_toolkits.mplot3d.axes3d as ax3d
from sympy.functions.special.delta_functions import DiracDelta
from scipy import optimize, linalg
from scipy.signal import fftconvolve
from scipy.special import wofz
from scipy.stats import chisquare, bayes_mvs
from scipy.misc import factorial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib

class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []
        self.scatFitList1 = []
        self.probList = []
        self.maxProbIndex = []
        for i, dataFile in enumerate(self.dataFiles):
        
            #_Get the corresponding file at low temperature and integrate the curves
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the corresponding low temperature file for :\n ...' + 
                    dataFile[dataFile.rfind('/'):])
            resFile = QFileDialog().getOpenFileName()[0]
            self.resList.append(inxBinQENS.inxBin(resFile, karg['binS']))
        
            #_Get datas from the file and store them into self.dataList 
            inxDatas = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxDatas)    

        #_Keep the datas from wanted q-range
        qMin = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMin']) - x))
        qMax = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMax']) - x))
        for i, fileDatas in enumerate(self.dataFiles):
            self.dataList[i] = [val for val in self.dataList[i] if qMin <= val.qVal <= qMax]
            self.resList[i] = [val for val in self.resList[i] if qMin <= val.qVal <= qMax]

        #_Discard the selected index
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,.:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])
            for j, fileDatas in enumerate(self.dataFiles):
                for val in qDiscardList:
                    self.dataList[j].pop(int(val))
                    self.resList[j].pop(int(val)) 
        

        self.resFit()
        self.scatFit()
 
#_Construction of the GUI
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        

        #_Add some interactive elements
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        self.fitButton = QPushButton('Fit')
        self.fitButton.clicked.connect(self.fitPlot)

        self.analysisButton = QPushButton('Analysis')
        self.analysisButton.clicked.connect(self.analysisPlot)

        self.resButton = QPushButton('Resolution')
        self.resButton.clicked.connect(self.resPlot)

        self.plot3DButton = QPushButton('plot 3D')
        self.plot3DButton.clicked.connect(self.plot3D)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.label = QLabel('Q value to plot', self)
        self.lineEdit = QLineEdit(self) 
        self.lineEdit.setText('0.8')

        self.labelLor = QLabel('Number of Lorentzians for fitting', self)
        self.lineEditLor = QLineEdit(self) 
        self.lineEditLor.setText('2')

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.labelLor)
        layout.addWidget(self.lineEditLor)
        layout.addWidget(self.button)
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.fitButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.resButton)
        self.setLayout(layout)

#_Everything needed for the fit
    def resFunc(self, x, lorG, gauG, lorS, normF, shift, bkgd):

        return  (normF * (lorS * lorG/(lorG**2 + (x-shift)**2) /np.pi 
                + (1-lorS) * np.exp(-((x-shift)**2) / (2*gauG**2)) / (gauG*np.sqrt(2*np.pi))
                + bkgd))  
                


    def resFit(self):
    
        for i, resFile in enumerate(self.resList):
            resList = []
            for j, datas in enumerate(resFile):
                resList.append(optimize.curve_fit(self.resFunc, 
                                datas.energies,
                                datas.intensities,
                                sigma=[val + 0.0001 for val in datas.errors],
                                #p0 = [0.1, 1, 0.95, 50, 0, 0.01], 
                                bounds=([0., 0., 0., 0., -10, 0],  
                                        [2, 10, 1., 5000, 10, 1]),
                                max_nfev=1000000,
                                method='trf'))
            self.resFitList.append(resList)


    def fitFunc(self, x, fileDatas, i, N):

        bkgdL = []

        funToFit = 0
        s0 = x[0]
        sList = []
        gList = []
        for numLor in range(0, N):
            exec('s%s = x[2*numLor+1]' % (numLor+1))
            exec('sList.append(s%s)' % (numLor+1)) 
            exec('g%s = x[2*numLor+2]' % (numLor+1)) 
            exec('gList.append(g%s)' % (numLor+1)) 

        for j, datas in enumerate(fileDatas):
    
            resShift = self.resFitList[i][j][0][4]
            normFact = self.resFitList[i][j][0][3]
            lorS = self.resFitList[i][j][0][2]
            GauR = self.resFitList[i][j][0][1]
            lorR = self.resFitList[i][j][0][0]
            bkgd = self.resFitList[i][j][0][5] 
            sigma = GauR/np.sqrt(2*np.log(2))

            if N > 0:
                chiSquared = sum([(datas.intensities[k]/normFact
                    - s0 * self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd)

                    - sum([sList[numLor] * (lorS * (lorR + gList[numLor])
                                            / (np.pi * ((lorR + gList[numLor])**2 + val**2)) 
                           + (1-lorS) * np.real(wofz(val + 1j*gList[numLor] 
                           / (sigma*np.sqrt(2)))) 
                           / (sigma*np.sqrt(np.pi*2)) + bkgd) for numLor in range(0, N)]))**2

                    / (datas.intensities[k] / normFact)**2 
                    * (sum(datas.intensities) / sum(datas.errors)) 
                    for k, val in enumerate(datas.energies)])

            else:
                chiSquared = sum([(datas.intensities[k]/normFact
                    - s0 * self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd))**2
                    /datas.errors[k]**2 * (sum(datas.intensities) / sum(datas.errors))**2 
                    for k, val in enumerate(datas.energies)])

            #funToFit += np.log(1 + chiSquared) #_Cauchy loss function 
            #funToFit += 2 * ((1 + chiSquared)**0.5 - 1) #_Soft_l1 loss function
            funToFit += chiSquared

        return funToFit


    def scatFit(self):
    
        for i, fileDatas in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            
            scatList1 = []
            meanResWidth = np.mean([self.resFitList[i][j][0][1] 
                                    for j, val in enumerate(fileDatas)])
            for N in range(0, 6):
                #_Minimization 
                x0 = [0.1, 1, 
                      0.4, 30, 
                      0.1, 300, 
                      0.3, 100, 
                      0.2, 2]
                boundaries = [(0., 1), (meanResWidth, 1000),
                              (0., 1), (meanResWidth, 1000),
                              (0., 1), (meanResWidth, 1000),
                              (0., 1), (meanResWidth, 1000),
                              (0., 1), (meanResWidth, 1000)]

                scatList1.append(optimize.basinhopping(lambda x:
                        self.fitFunc(x, fileDatas, i, N),
                        [0.5] + x0[:2*N], 
                        niter=10,
                        minimizer_kwargs={'method':'L-BFGS-B'}))

                print('> Final result for %d lorentzian(s): ' % (N), flush=True)
                if N > 0:
                    print('Fit s0 : ' + str(scatList1[N].x[0]), flush=True)
                    for k in range(0, N):
                        print('Fit g%d : ' % (k+1) + str(scatList1[N].x[2*k+2]), flush=True)
                        print('Fit s%d : ' % (k+1) + str(scatList1[N].x[2*k+1]), flush=True)
                else:
                    print('Fit s0 : ' + str(scatList1[0].x[0]), flush=True)
                print('\n', flush=True)

            self.scatFitList1.append(scatList1)

            #_This part compute the posterior probability for each number of Lorentzians
            #_and store it into self.probList[i] where i is the data file.
            #_The self.maxProbIndex store the indices for which the prob is  
            #_the highest for each data file. 
            probList = []
            sMax = max([max([val for k, val in enumerate(scatList1[numLor].x) 
                        if k > 0 and k % 2 != 0]) for numLor in range(1, 6)]) 
            gMax = max([max([val for k, val in enumerate(scatList1[numLor].x) 
                        if k > 0 and k % 2 == 0]) for numLor in range(1, 6)])
            for numLor in range(1, 6):
                probList.append(np.log10(factorial(numLor) / (sMax*gMax)**numLor
                    * (4*np.pi)**numLor * np.exp(-scatList1[numLor].fun/2)
                    / np.sqrt(1 / 
                    linalg.det(scatList1[numLor].lowest_optimization_result.hess_inv.todense()))))

            self.probList.append(probList)
            self.maxProbIndex.append(np.argmax(self.probList[i]) + 1)

    #_Function used to produce the plot of the fit
    def fittedFunc(self, datas, resShift, normFact, lorS, GauR, lorR, sigma, s0, bkgd, k, N):

        sList = []
        gList = []
        for numLor in range(0, N):
            exec('s%s = self.scatFitList1[k][N].x[2*numLor+1]' % (numLor+1))
            exec('sList.append(s%s)' % (numLor+1)) 
            exec('g%s = self.scatFitList1[k][N].x[2*numLor+2]' % (numLor+1)) 
            exec('gList.append(g%s)' % (numLor+1)) 

        Model = [s0 * self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) +
 
                sum([sList[numLor] * (lorS * (lorR + gList[numLor])
                                        / (np.pi * ((lorR + gList[numLor])**2 + val**2)) 
                + (1-lorS) * np.real(wofz(val + 1j*gList[numLor]/(sigma*np.sqrt(2)))) /
                (sigma*np.sqrt(np.pi*2)) + bkgd) for numLor in range(0, N)])

                for k, val in enumerate(datas.energies)] 

        return Model        


#_Definitions of the slots for the plot window
    def plot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - x))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
                    ax.errorbar(qDatas.energies, 
                           [x/self.resFitList[k][j][0][3] for x in qDatas.intensities],
                           [x/self.resFitList[k][j][0][3] for x in qDatas.errors], fmt='o')
            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.legend(['...' + dataFile[dataFile.rfind('/'):] for dataFile in self.dataFiles], 
                   loc='upper left', framealpha=0.5)
            
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot3D(self):

        plt.gcf().clear()     
        #_3D plots of the datas for each selected datafiles
        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))

        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k], projection='3d')

            for i, datas in enumerate(fileDatas):
                normFact = self.resFitList[k][i][0][3]

                ax.plot(datas.energies, 
                                  [val/normFact for val in datas.intensities], 
                                  datas.qVal, zdir='y', c=cmap(normColors(datas.qVal)))

            ax.set_xlabel(r'$\hslash \omega (\mu eV)$')
            ax.set_ylabel(r'$q$')
            ax.set_zlabel(r'$S_{300K}(q, \omega)$')
            ax.set_ylim((0, 2))
            ax.set_zlim((0, 1))
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()

        plt.tight_layout()
        self.canvas.draw()
 
    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(2, len(self.dataList)) 

        for k, fileDatas in enumerate(self.dataList):
           
            ax = self.figure.add_subplot(mplGrid[:1,k])
            ax.plot(range(1,6), self.probList[k], marker='o', 
                    label=self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_xlabel(r'$Number\ of\ lorentzians$', fontsize=18)
            ax.set_ylabel(r'$Log_{10}(P\ (N\ |\ Data))$', fontsize=18)
            ax.grid(True)

            #_Plot of the parameters of the fit
            gList = []
            gErrList = []
            sList = []
            sErrList = []
            lorNum = self.maxProbIndex[k]
            for nParam in range(0, self.maxProbIndex[k]):
                gList.append(self.scatFitList1[k][self.maxProbIndex[k]].x[2*nParam+2])
                gErrList.append(np.sqrt(np.diag(
        self.scatFitList1[k][lorNum].lowest_optimization_result.hess_inv.todense())[2*nParam+2]))
                sList.append(self.scatFitList1[k][self.maxProbIndex[k]].x[2*nParam+1])
                sErrList.append(np.sqrt(np.diag(
        self.scatFitList1[k][lorNum].lowest_optimization_result.hess_inv.todense())[2*nParam+1]))
            
            ax = self.figure.add_subplot(mplGrid[1:,k])
            ax.errorbar(gList, sList, xerr=gErrList, yerr=sErrList, 
                    marker='o', linestyle='None', 
                    label=self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_xlabel(r'$\Gamma \ parameter \ (\mu eV)$', fontsize=18)
            ax.set_ylabel(r'$Amplitude$', fontsize=18)
            plt.legend(framealpha=0.5)
            ax.grid(True)

        plt.tight_layout()
        self.canvas.draw()

    def fitPlot(self):
                
        plt.gcf().clear()     
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
                            key = lambda x : abs(float(self.lineEdit.text()) - x))


        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k])
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
                    resShift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][3]
                    lorS = self.resFitList[k][j][0][2]
                    GauR = self.resFitList[k][j][0][1]
                    lorR = self.resFitList[k][j][0][0]
                    bkgd = self.resFitList[k][j][0][5] 
                    sigma = GauR/np.sqrt(2*np.log(2))

                    #_Plot of the probability for the models
                    s0 = self.scatFitList1[k][self.maxProbIndex[k]].x[0]
                    gList = []
                    sList = []


                    if self.lineEditLor.text() == '':
                        nbrLor = self.maxProbIndex[k]
                        for nParam in range(0, self.maxProbIndex[k]):
                            gList.append(self.scatFitList1[k][self.maxProbIndex[k]].x[2*nParam+2])
                            sList.append(self.scatFitList1[k][self.maxProbIndex[k]].x[2*nParam+1])
                    else:
                        try:
                            nbrLor = int(self.lineEditLor.text())
                            assert 0 <= nbrLor < 6
                        except:
                            errorDiag = QMessageBox.critical(self, 'Error',
                        'Error, the number of Lorentzians should be a integer between 0 and 5.')
                                
                        for nParam in range(0, nbrLor):
                            gList.append(self.scatFitList1[k][nbrLor].x[2*nParam+2])
                            sList.append(self.scatFitList1[k][nbrLor].x[2*nParam+1])
                    
                    #_Plot of the measured QENS signal
                    ax.errorbar(qDatas.energies, 
                           [x/normFact for x in qDatas.intensities],
                           [x/normFact for x in qDatas.errors], fmt='o')

                    #_Plot of the resolution function
                    ax.plot(qDatas.energies, 
                            [self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) 
                                             for val in qDatas.energies], label='resolution')

                    #_Plot of the fitted lorentzian
                    for index, val in enumerate(gList):
                        exec('g%s = val' % (index+1))
                        exec('s%s = sList[index]' % (index+1))

                        ax.plot(qDatas.energies, 
                                        [gList[index]/(gList[index]**2+val**2) * sList[index]
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian %s' % (index+1))


                    #_Plot of the fitted incoherent structure factor
                    ax.plot(qDatas.energies, self.fittedFunc(qDatas, resShift, normFact, lorS, 
                                    GauR, lorR, sigma, s0, bkgd, k, nbrLor), 
                            label='convolution', linewidth=2, color='orangered')
                    

            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.set_ylim(1e-3, 1.2)
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            plt.legend(framealpha=0.5)
            ax.grid()
        
        plt.tight_layout()
        self.canvas.draw()

    
    def resPlot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - x))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.resList):
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
    
                    resShift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][3]
                    lorS = self.resFitList[k][j][0][2]
                    GauR = self.resFitList[k][j][0][1]
                    lorR = self.resFitList[k][j][0][0]
                    bkgd = self.resFitList[k][j][0][5] 

                    ax.plot(qDatas.energies, [val/normFact for val in qDatas.intensities])
    
                    ax.plot(qDatas.energies, 
                            [self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd)
                                             for val in qDatas.energies])


            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S_{10K}(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.legend(['...' + dataFile[dataFile.rfind('/'):] for dataFile in self.dataFiles], 
                   loc='upper left', framealpha=0.5)
            
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)
    subW = Window() 
    subW.show()

    sys.exit(app.exec_())
