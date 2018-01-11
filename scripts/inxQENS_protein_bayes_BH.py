'''
This script is used to perform a bayesian analysis of the data and thus, determine the best
number of Lorentzians to be used to fit them. The 'fit' button use automaticaly the best
model found for the fit.

Reference :
-   D.S. Sivia, S König, C.J. Carlile and W.S. Howells (1992) Bayesian analysis of 
    quasi-elastic neutron scattering data. Physica B, 182, 341-348 
'''

import sys, os
import pickle as pk
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
from scipy import optimize, linalg
from scipy.misc import factorial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib
import inxQENS_extract_resFunc_analysis as resFuncAnalysis
import inxENS_extract_normF as elasticNormF

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
        self.meanResGauWidth = []
        self.normF = []

        #_Parsing the additional parameters
        try:
            self.BH_iter = int(karg['BH_iter'])
        except ValueError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'Error: the "BH_iter" parameter can\'t be converted to an' +
                            ' integer.')
            sys.exit()
        except KeyError:
            self.BH_iter = 300
            print('No value for the number of basinhopping iterations to be' +
                            ' used was given. \n' +
                            'Using BH_iter=300 as default value.\n', flush=True)

        #_Get the resolution function parameters
        with open(os.path.dirname(os.path.abspath(__file__)).rstrip('scripts') + 'params/resFunc_params', 
                  'rb') as resFit:
            resFitData = pk.Unpickler(resFit).load()
            self.resFitList = resFitData[0]
            #_If there is only of fit for the resolution function, we assume the same is used
            #_for all data files
            if len(self.resFitList) == 1:
                self.resFitList = np.tile(self.resFitList, len(self.dataFiles))
            self.meanResGauWidth = resFitData[1]

        #_Get datas from the file and store them into self.dataList 
        for i, dataFile in enumerate(self.dataFiles):
            inxDatas = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxDatas)    


        #_Discard the selected index
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,.:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])
            for j, fileDatas in enumerate(self.dataFiles):
                for val in qDiscardList:
                    self.dataList[j].pop(int(val))
                    self.resFitList[j].pop(int(val))
        

        #_Keep the datas from wanted q-range
        qMin = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMin']) - x))
        qMax = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMax']) - x))
        for i, fileDatas in enumerate(self.dataFiles):
            self.resFitList[i] = [self.resFitList[i][qIdx] for qIdx, val in enumerate(self.dataList[i]) 
                                                                    if qMin <= val.qVal <= qMax]
            self.dataList[i] = [val for val in self.dataList[i] if qMin <= val.qVal <= qMax]


        #_Get values for normalization
        for i, dataFile in enumerate(self.dataFiles):
            if karg['elasticNormF']=='True':
                self.normF.append(elasticNormF.get_elastic_normF())
            else:
                self.normF.append([val[0][0] for val in self.resFitList[i]])


        if karg['new_fit']=='True':
            #_Get the file in which the fitted parameters are to be saved
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file in which to save the fitted parameters...') 
            self.paramsFile = QFileDialog().getSaveFileName()[0]
            self.scatFit()
            self.getProba()
        else:
            #_Get the file containing the fitted parameters
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file containing the fitted parameters...') 
            paramsFile = QFileDialog().getOpenFileName()[0]
            with open(paramsFile, 'rb') as params:
                self.scatFitList1 = pk.Unpickler(params).load()
            self.getProba()
 
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
        self.setLayout(layout)

#_Everything needed for the fit
    def fitFunc(self, x, fileData, i, N):

        cost = 0
        s0 = x[0]
        sList = x[1:2*N+3:2]
        sList = sList.reshape(sList.shape[0], 1)
        gList = x[2:2*N+3:2]
        gList = gList.reshape(gList.shape[0], 1)

        for j, data in enumerate(fileData):
    
            shift = self.resFitList[i][j][0][4]
            normF = self.resFitList[i][j][0][0]
            S = self.resFitList[i][j][0][1]
            gauW = self.resFitList[i][j][0][3]
            lorW = self.resFitList[i][j][0][2]
            bkgd = 0

            pBkgd = x[j + 2*N+3]
            msd = x[-1]

            X = data.energies
            
            #_Resolution function
            f_res = ((S * lorW / (lorW**2 + (X - shift)**2) / np.pi 
                    + (1-S) * np.exp(-(X - shift)**2 / (2*gauW**2)) / (gauW * np.sqrt(2*np.pi)) 
                    + bkgd))

            #_Lorentzians
            f_lor = gList / (np.pi * (X**2 + gList**2))

            convolutions = np.array([np.convolve(val, f_res, mode='same') for val in f_lor])

            f = np.exp(-data.qVal**2*msd/3) * (s0 * f_res + np.sum(sList * convolutions, axis=0) + pBkgd)

            cost += np.sum((data.intensities / self.normF[i][j] - f)**2 / (data.errors / self.normF[i][j])**2)


        return cost


    def scatFit(self):
    
        for i, fileData in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            
            scatList1 = []
            for N in range(0, 5):
                #_Minimization 
                x0_lor = [0.3, 1, 0.3, 30, 0.2, 300, 0.1, 100, 0.1, 2]
                bounds_lor = [(0., 1), (0., 200), (0., 1), (0., 200), (0., 1), 
                              (0., 200), (0., 1), (0., 200), (0., 1), (0., 200)]

                x0 = [0.5] + x0_lor[:2*N+3] + [0.005 for val in fileData] + [0.5]
                bounds = [(0., 1)] + bounds_lor[:2*N+3] + [(0., 0.5) for val in fileData] + [(0., 5)]

                scatList1.append(optimize.basinhopping(self.fitFunc,
                                    x0, 
                                    niter=self.BH_iter,
                                    niter_success=0.5*self.BH_iter,
                                    disp=True,
                                    minimizer_kwargs={'args': (fileData, i, N),
                                                      'bounds': bounds}))

                print('\n> Final result for %d lorentzian(s): ' % (N+1), flush=True)
                print('Fit s0 : ' + str(scatList1[N].x[0]), flush=True)
                print('Fit msd : ' + str(scatList1[N].x[-1]), flush=True)
                for k in range(0, N+1):
                    print('Fit g%d : ' % (k+1) + str(scatList1[N].x[2*k+2]), flush=True)
                    print('Fit s%d : ' % (k+1) + str(scatList1[N].x[2*k+1]), flush=True)
                print('\n', flush=True)

            self.scatFitList1.append(scatList1)
        
        with open(self.paramsFile, 'wb') as fittedParamsFile:
            myFile = pk.Pickler(fittedParamsFile)
            modelFitList = self.scatFitList1
            myFile.dump(modelFitList)


    def getProba(self):
        ''' This part compute the posterior probability for each number of Lorentzians
            and store it into self.probList[i] where i is the data file.
            The self.maxProbIndex store the indices for which the prob is  
            the highest for each data file. '''
        for i, fitList in enumerate(self.scatFitList1):
            probList = []

            sMax = max([max([val for k, val in enumerate(fitList[numLor].x) 
                        if 0 < k <= 2*numLor+3 and k % 2 != 0]) for numLor in range(0, 5)]) 
            gMax = max([max([val for k, val in enumerate(fitList[numLor].x) 
                        if 0 < k <= 2*numLor+3 and k % 2 == 0]) for numLor in range(0, 5)])

            for numLor in range(0, 5):
                probList.append(np.log10(factorial(numLor) / (sMax*gMax)**numLor
                    * (4*np.pi)**numLor * np.exp(-fitList[numLor].fun/2)
                    / np.sqrt(1/linalg.det(fitList[numLor].lowest_optimization_result.hess_inv.todense()))))

            self.probList.append(probList)
            self.maxProbIndex.append(np.argmax(self.probList[i]))

    #_Function used to produce the plot of the fit
    def fittedFunc(self, data, shift, normFact, S, gauW, lorW, bkgd, j, N, x):

        s0 = x[0]
        sList = x[1:2*N+3:2]
        sList = sList.reshape(sList.shape[0], 1)
        gList = x[2:2*N+3:2]
        gList = gList.reshape(gList.shape[0], 1)
        pBkgd = x[j + 2*N+3]
        msd = x[-1]

        X = data.energies
        
        #_Resolution function
        f_res = ((S * lorW / (lorW**2 + (X - shift)**2) / np.pi 
                + (1-S) * np.exp(-(X - shift)**2 / (2*gauW**2)) / (gauW * np.sqrt(2*np.pi)) 
                + bkgd))

        #_Lorentzians
        f_lor = gList / (np.pi * (X**2 + gList**2))

        convolutions = np.array([np.convolve(val, f_res, mode='same') for val in f_lor])

        f = np.exp(-data.qVal**2*msd/3) * (s0 * f_res + np.sum(sList * convolutions, axis=0) + pBkgd)

        return f



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
                           [x/self.normF[k][j] for x in qDatas.intensities],
                           [x/self.normF[k][j] for x in qDatas.errors], fmt='o')
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
        #_3D plots of the data for each selected datafiles
        if len(self.dataList) <= 5:
            ax = [self.figure.add_subplot(1, len(self.dataList), i+1, projection='3d') 
                                        for i, val in enumerate(self.dataList)]
        else:
            ax = [self.figure.add_subplot(2, 5, i+1, projection='3d') 
                                        for i, val in enumerate(self.dataList)]

        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for k, filedata in enumerate(self.dataList):
            for i, data in enumerate(filedata):

                ax[k].plot(data.energies, [val / self.normF[k][i] for val in data.intensities], 
                                  data.qVal, zdir='y', c=cmap(normColors(data.qVal)))

            ax[k].set_xlabel(r'$\hslash \omega (\mu eV)$')
            ax[k].set_ylabel(r'$q$')
            ax[k].set_zlabel(r'$S_{300K}(q, \omega)$')
            ax[k].set_ylim((0, 2))
            ax[k].set_zlim((0, 1))
            ax[k].set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax[k].grid()

        plt.tight_layout()
        self.canvas.draw()
 
    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     
        mplGrid = gridspec.GridSpec(2, len(self.dataList)) 

        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:1,k])
            ax.plot(range(1, 6), self.probList[k], marker='o')
            ax.set_title(self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_xlabel(r'$Number\ of\ lorentzians$', fontsize=18)
            ax.set_ylabel(r'$Log_{10}(P\ (N\ |\ Data))$', fontsize=18)
            ax.grid(True)

            #_Plot of the parameters of the fit
            gList = []
            gErrList = []
            sList = []
            sErrList = []
            lorNum = self.maxProbIndex[k]
            for nParam in range(0, self.maxProbIndex[k]+1):
                gList.append(self.scatFitList1[k][lorNum].x[2*nParam+2])
                gErrList.append(np.sqrt(np.diag(
                    self.scatFitList1[k][lorNum].lowest_optimization_result.hess_inv.todense())[2*nParam+2]))
                sList = np.append(sList, self.scatFitList1[k][lorNum].x[2*nParam+1])
                sList = np.array(sList)
                sList = sList / (np.sum(sList) + self.scatFitList1[k][lorNum].x[0])
                sErrList.append(np.sqrt(np.diag(
                    self.scatFitList1[k][lorNum].lowest_optimization_result.hess_inv.todense())[2*nParam+1]))

            
            ax = self.figure.add_subplot(mplGrid[1:,k])
            ax.errorbar(gList, sList, xerr=gErrList, yerr=sErrList, 
                    marker='o', linestyle='None')
            ax.set_xlabel(r'$\Gamma \ parameter \ (\mu eV)$', fontsize=18)
            ax.set_ylabel(r'$Amplitude$', fontsize=18)
            ax.grid(True)

        plt.tight_layout()
        self.canvas.draw()


    def fitPlot(self):
                
        plt.gcf().clear()     
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
                            key = lambda x : abs(float(self.lineEdit.text()) - x))

        if len(self.dataList) <= 5:
            ax = [self.figure.add_subplot(1, len(self.dataList), i+1) 
                                        for i, val in enumerate(self.dataList)]
        else:
            ax = [self.figure.add_subplot(2, 5, i+1) 
                                        for i, val in enumerate(self.dataList)]

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
                    resShift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][0]
                    lorS = self.resFitList[k][j][0][1]
                    GauR = self.resFitList[k][j][0][3]
                    lorR = self.resFitList[k][j][0][2]
                    bkgd = 0

                    #_Plot of the probability for the models
                    s0 = self.scatFitList1[k][self.maxProbIndex[k]].x[0]
                    pBkgd = self.scatFitList1[k][self.maxProbIndex[k]].x[j + 2*self.maxProbIndex[k]+3]
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
                            assert 0 < nbrLor < 6
                        except:
                            errorDiag = QMessageBox.critical(self, 'Error',
                        'Error, the number of Lorentzians should be a integer between 1 and 5.')
                                
                        for nParam in range(0, nbrLor):
                            gList.append(self.scatFitList1[k][nbrLor].x[2*nParam+2])
                            sList.append(self.scatFitList1[k][nbrLor].x[2*nParam+1])
                    
                    #_Plot of the measured QENS signal
                    ax[k].errorbar(qDatas.energies, 
                           [x/self.normF[k][j] for x in qDatas.intensities],
                           [x/self.normF[k][j] for x in qDatas.errors], fmt='o')

                    #_Plot of the resolution function
                    ax[k].plot(qDatas.energies, 
                            [resFuncAnalysis.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) 
                                             for val in qDatas.energies], label='resolution')

                    ax[k].axhline(y=pBkgd, label='background')                    

                    #_Plot of the fitted lorentzian
                    for index, val in enumerate(gList):
                        exec('g%s = val' % (index+1))
                        exec('s%s = sList[index]' % (index+1))

                        ax[k].plot(qDatas.energies, 
                                        [gList[index]/(gList[index]**2+val**2) * sList[index]
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian %s' % (index+1))


                    #_Plot of the fitted incoherent structure factor
                    ax[k].plot(qDatas.energies, self.fittedFunc(qDatas, resShift, normFact, lorS, GauR, lorR, 
                                        bkgd, j, nbrLor, self.scatFitList1[k][nbrLor].x), 
                            label='convolution', linewidth=2, color='orangered', zorder=5)
            
                    ax[k].legend(framealpha=0.5, loc='upper right')
                    

            ax[k].set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax[k].set_yscale('log')
            ax[k].set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax[k].set_ylim(1e-3, 1.2)
            ax[k].set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax[k].grid()
        
        plt.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)
    subW = Window() 
    subW.show()

    sys.exit(app.exec_())
