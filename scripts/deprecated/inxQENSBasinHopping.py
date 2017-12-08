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
import mpl_toolkits.mplot3d.axes3d as ax3d
from sympy.functions.special.delta_functions import DiracDelta
from scipy import optimize
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
        #_Delete the unwanted q values from the lists
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,.:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []
        self.scatFitList1 = []
        self.tempStats = []
        self.fitStats = []
        self.BHIter = 20

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
 
        #_3D plots of the datas for each selected datafiles
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')
        for k, fileDatas in enumerate(self.dataList):
            fig3d = plt.figure()
            qensPlot = ax3d.Axes3D(fig3d)

            for i, datas in enumerate(fileDatas):
                normFact = self.resFitList[k][i][0][3]

                qensPlot.plot(datas.energies, 
                                  [val/normFact for val in datas.intensities], 
                                  float(datas.qVal), zdir='y', c=cmap(normColors(datas.qVal)))

            qensPlot.set_xlabel(r'$\hslash \omega (\mu eV)$')
            qensPlot.set_ylabel(r'$q$')
            qensPlot.set_zlabel(r'$S_{300K}(q, \omega)$')
            qensPlot.set_ylim((0, 2))
            qensPlot.set_zlim((0, 1))
            qensPlot.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            plt.show(block=False)

        plt.grid()
        plt.yscale('log')
        plt.show(block=False)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        

        #_Add some interactive elements
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        self.fitButton = QPushButton('Fit parameters')
        self.fitButton.clicked.connect(self.fitPlot)

        self.analysisButton = QPushButton('Analysis')
        self.analysisButton.clicked.connect(self.analysisPlot)

        self.resButton = QPushButton('Resolution')
        self.resButton.clicked.connect(self.resPlot)

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
        layout.addWidget(self.fitButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.resButton)
        self.setLayout(layout)

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


    def fitFunc(self, x, datas, resShift, normFact, lorS, GauR, lorR, sigma):

        chiSquared = sum([(datas.intensities[k]/normFact - 
                (x[1] * self.resFunc(val, lorR, GauR, lorS, 1, resShift) + 
                x[2] * (lorS * fftconvolve(lorR/(val**2 + lorR**2), 
                x[0]/(val**2 + x[0]**2), mode='same') + 
                (1-lorS) * np.real(wofz(val + 1j*x[0]/(sigma*np.sqrt(2))) / 
                (sigma*np.sqrt(np.pi*2)))) + x[3]))**2 
                /datas.errors[k]**2
                for k, val in enumerate(datas.energies)]) 

        return chiSquared        

    def scatFit(self):
    
        for i, fileDatas in enumerate(self.dataList):
            scatList1 = []
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            for j, datas in enumerate(fileDatas):
                resShift = self.resFitList[i][j][0][4]
                normFact = self.resFitList[i][j][0][3]
                lorS = self.resFitList[i][j][0][2]
                GauR = self.resFitList[i][j][0][1]
                lorR = self.resFitList[i][j][0][0]

                sigma = GauR/np.sqrt(2*np.log(2))

                scatList1.append(optimize.basinhopping(lambda x:
                            self.fitFunc(x, datas, resShift, normFact, lorS, GauR, lorR, sigma),
                            [1, 0.4, 0.06, 0.003], callback=self.fitState,
                            niter=self.BHIter, minimizer_kwargs={'method':'BFGS'}))


                print('Fit q-value : ' + str(datas.qVal), flush=True)
                print('Fit Gamma : ' + str(scatList1[j].x[0]), flush=True) 
                print('Fit res S : ' + str(scatList1[j].x[1]), flush=True)
                print('Fit lor S: ' + str(scatList1[j].x[2]), flush=True)
                print('Fit bkgd : ' + str(scatList1[j].x[3]), flush=True)
                print('', flush=True)


            self.scatFitList1.append(scatList1)

    #_Callback function for the basinHopping algorithm
    def fitState(self, x, f, accepted):
        print('at minimum   %.4f     accepted : %d' % (f, int(accepted)), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Gamma', x[0]), flush=True)

    
    #_Function used to plot the data
    def fittedFunc(self, datas, resShift, normFact, lorS, GauR, lorR, sigma, g0, s0, s1, bkgd):

        Model = [s0 * self.resFunc(val, lorR, GauR, lorS, 1, resShift) + 
                s1 * (lorS * fftconvolve(lorR/(val**2 + lorR**2), 
                g0/(val**2 + g0**2), mode='same') + 
                (1-lorS) * np.real(wofz(val + 1j*g0/(sigma*np.sqrt(2))) / 
                (sigma*np.sqrt(np.pi*2)))) + bkgd
                for k, val in enumerate(datas.energies)] 

        return Model        

    def fitPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))
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

                g0 = self.scatFitList1[k][j].x[0]
                s0 = self.scatFitList1[k][j].x[1]
                s1 = self.scatFitList1[k][j].x[2]
                bkgd = self.scatFitList1[k][j].x[3]

                sigma = GauR/np.sqrt(2*np.log(2))

                qList.append(qDatas.qVal)
                gammaList.append(self.scatFitList1[k][j].x[0])
                chi2List.append(chisquare(self.fittedFunc(qDatas, resShift, normFact, lorS,
                                            GauR, lorR, sigma, g0, s0, s1, bkgd),
                                          [val/normFact for val in qDatas.intensities])[0])
                
            
            #_Plot of the chi squared of the fits
            ax = self.figure.add_subplot(mplGrid[:,0])
            ax.plot(qList, chi2List, label=r'$\chi^{2}$', marker='o')
            ax.set_xlabel(r'$q$', fontsize=18)
            ax.set_ylabel(r'$\chi^{2}$', fontsize=18)
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()

            #_Plot of the gammas of the system lorentzian
            ax = self.figure.add_subplot(mplGrid[:,1])
            ax.plot(qList, gammaList, 
            label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], marker='o')
            ax.set_xlabel(r'$q$', fontsize=18)
            ax.set_ylim([0., 5])
            ax.set_ylabel(r'$\Gamma$', fontsize=18)
            plt.legend(framealpha=0.5)
            ax.grid()
        

        plt.tight_layout()
        self.canvas.draw()

    def analysisPlot(self):
            
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
                    g0 = self.scatFitList1[k][j].x[0]
                    s0 = self.scatFitList1[k][j].x[1]
                    sigma = GauR/np.sqrt(2*np.log(2))
                    s1 = self.scatFitList1[k][j].x[2]
                    bkgd = self.scatFitList1[k][j].x[3]

                    ax.plot(qDatas.energies, [g0/(g0**2+val**2)*s1/g0
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian')

                    #_Plot of the background
                    ax.plot(qDatas.energies, [bkgd for val in qDatas.energies], ls='--',
                            label='background') 
                    

                    #_Plot of the fitted incoherent structure factor
                    ax.plot(qDatas.energies, self.fittedFunc(qDatas, resShift, normFact, lorS, 
                                                            GauR, lorR, sigma, g0, s0, s1, bkgd), 
                            label='convolution', linewidth=2)
                    

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
