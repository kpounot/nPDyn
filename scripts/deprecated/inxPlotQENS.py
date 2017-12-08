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
from scipy.special import wofz
from scipy.stats import chisquare
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib

class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)
        #_Delete the unwanted q values from the lists
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,.:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []
        self.scatFitList = []
        self.lorFitList = []

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

        self.plot3DButton = QPushButton('3D Plot')
        self.plot3DButton.clicked.connect(self.plot3D)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.label = QLabel('Q value to plot', self)
        self.lineEdit = QLineEdit(self) 
        self.lineEdit.setText('0.8')

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button)
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.fitButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.resButton)
        self.setLayout(layout)

#_Everything needed for the fit
    def resFunc(self, x, a, b, c, d, e):
        return  d * (c * a/(a**2 + (x-e)**2) + (1-c) * np.exp(-((x-e)**2) / (b**2)))  

    def resFit(self):
    
        for i, resFile in enumerate(self.resList):
            resList = []
            for j, datas in enumerate(resFile):
                resList.append(optimize.curve_fit(self.resFunc, 
                                datas.energies,
                                datas.intensities,
                                sigma=[val + 0.0001 for val in datas.errors],
                                #p0 = [1, 1, 0.8, 20, 0], 
                                bounds=([0., 0., 0., 0., -10], [1, 10, 1., 10000, 10]), 
                                method='trf'))
            self.resFitList.append(resList)


    def scatFunc(self, x, g0, s0, s1, resShift, normFact, lorS, GauR, lorR, sigma, bkgd):

        Model = (s0 * self.resFunc(x, lorR, GauR, lorS, normFact, resShift) + 
                s1 * (lorS * fftconvolve(lorR/(x**2 + lorR**2), 
                g0/(x**2 + g0**2), mode='same') + 
                (1-lorS) * np.real(wofz(x + 1j*g0/(sigma*np.sqrt(2))) / 
                (sigma*np.sqrt(np.pi*2))))) + bkgd 

        return Model        


    def scatFit(self):
    
        for i, fileDatas in enumerate(self.dataList):
            scatList = []
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):])
            for j, datas in enumerate(fileDatas):
                resShift = self.resFitList[i][j][0][4]
                normFact = self.resFitList[i][j][0][3]
                lorS = self.resFitList[i][j][0][2]
                GauR = self.resFitList[i][j][0][1]
                lorR = self.resFitList[i][j][0][0]

                sigma = GauR/np.sqrt(2*np.log(2))

                scatList.append(optimize.curve_fit(lambda x, g0, s0, s1, bkgd:
                            self.scatFunc(x, g0, s0, s1, resShift, 1, 
                            lorS, GauR, lorR, sigma, bkgd), 
                            datas.energies,
                            [val/normFact for val in datas.intensities],
                            sigma=datas.errors,
                            p0 = [8, 0.5, 0.5, 0.001], 
                            bounds=([0., 0., 0., 0.], 
                            [100, 1, 1, 0.01]), method='trf'))

                print('qVal : ' + str(datas.qVal) +':')
                print('Gamma : ' + str(scatList[j][0][0]))
                print('Scale resolution : ' + str(scatList[j][0][1]))
                print('Scale lorentzian : ' + str(scatList[j][0][2]))
                print('Background : ' + str(scatList[j][0][3]))
                print('\n')
            self.scatFitList.append(scatList)


#_Definitions of the slots for the plot window
    def plot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - float(x)))
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

    
    def analysisPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(2, len(self.dataFiles))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:1,k])
            qList = []
            gammaList = []
            chi2List = []
            for j, qDatas in enumerate(fileDatas):
                resShift = self.resFitList[k][j][0][4]
                normFact = self.resFitList[k][j][0][3]
                lorS = self.resFitList[k][j][0][2]
                GauR = self.resFitList[k][j][0][1]
                lorR = self.resFitList[k][j][0][0]

                g0 = self.scatFitList[k][j][0][0]
                s0 = self.scatFitList[k][j][0][1]
                s1 = self.scatFitList[k][j][0][2]
                bkgd = self.scatFitList[k][j][0][3]

                sigma = GauR/np.sqrt(2*np.log(2))

                qList.append(qDatas.qVal)
                gammaList.append(self.scatFitList[k][j][0][0])
                chi2List.append(chisquare([self.scatFunc(x, g0, s0, s1, resShift, 1, lorS,
                                          GauR, lorR, sigma, bkgd) for x in qDatas.energies],
                                          [val/normFact for val in qDatas.intensities])[0])
                

            #_Plot of the chi squared of the fits
            ax = self.figure.add_subplot(mplGrid[:1,k])
            ax.plot(qList, chi2List, label=r'$\chi^{2}$', marker='o')
            ax.set_xlabel(r'$q$', fontsize=18)
            ax.set_ylabel(r'$\chi^{2}$', fontsize=18)
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()

            #_Plot of the gammas of the system lorentzian
            ax = self.figure.add_subplot(mplGrid[1:,k])
            ax.errorbar(qList, gammaList, 
                        [np.sqrt(np.diag(val[1])[0]) for val in self.scatFitList[k]],
                        label='gamma', marker='o')
            ax.set_xlabel(r'$q$', fontsize=18)
            ax.set_ylim([0., 5])
            ax.set_ylabel(r'$\Gamma$', fontsize=18)
            ax.grid()

        plt.tight_layout()
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
 

    def fitPlot(self):
            
        plt.gcf().clear()     
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - float(x)))


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
                    
                    #_Plot of the measured QENS signal
                    ax.errorbar(qDatas.energies, 
                           [x/normFact for x in qDatas.intensities],
                           [x/normFact for x in qDatas.errors], fmt='o')

                    #_Plot of the resolution function
                    ax.plot(qDatas.energies, 
                            [self.resFunc(val, lorR, GauR, lorS, 1, resShift) 
                                             for val in qDatas.energies], label='resolution')

                    #_Plot of the fitted lorentzian
                    g0 = self.scatFitList[k][j][0][0]
                    s0 = self.scatFitList[k][j][0][1]
                    s1 = self.scatFitList[k][j][0][2]
                    bkgd = self.scatFitList[k][j][0][3]

                    sigma = GauR/np.sqrt(2*np.log(2))

                    ax.plot(qDatas.energies, [g0/(g0**2+val**2)
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian')

                    #_Plot of the background
                    ax.plot(qDatas.energies, [bkgd for val in qDatas.energies], ls='--',
                            label='background')
                    
                    #_Plot of the fitted incoherent structure factor
                    scatCurve = [self.scatFunc(val, g0, s0, s1, resShift,
                                 1, lorS, GauR, lorR, sigma, bkgd) 
                                 for val in qDatas.energies]

                    ax.plot(qDatas.energies, scatCurve, label='convolution', linewidth=2)
                    

            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()
        
        plt.legend(framealpha=0.5)
        plt.tight_layout()
        self.canvas.draw()

    
    def resPlot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - float(x)))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.resList):
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
    
                    resShift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][3]
                    lorS = self.resFitList[k][j][0][2]
                    GauR = self.resFitList[k][j][0][1]
                    lorR = self.resFitList[k][j][0][0]

                    ax.plot(qDatas.energies, [val/normFact for val in qDatas.intensities])
    
                    ax.plot(qDatas.energies, 
                            [self.resFunc(val, lorR, GauR, lorS, 1, resShift)
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
