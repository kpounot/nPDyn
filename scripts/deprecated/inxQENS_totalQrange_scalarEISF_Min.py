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

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []
        self.scatFitList1 = []
        self.scatFitList2 = []
        self.tempBkgd = []
        self.bkgdList = []
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

        self.fitButton = QPushButton('Fit parameters')
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


    def fitFunc(self, x, fileDatas, i):

        bkgdL = []

        chiSquared = 0
        g0 = x[0]
        s0 = x[1]
        s1 = x[2]
        for j, datas in enumerate(fileDatas):

            resShift = self.resFitList[i][j][0][4]
            normFact = self.resFitList[i][j][0][3]
            lorS = self.resFitList[i][j][0][2]
            GauR = self.resFitList[i][j][0][1]
            lorR = self.resFitList[i][j][0][0]
            sigma = GauR/np.sqrt(2*np.log(2))

            conv1 = [fftconvolve(lorR/(val**2 + lorR**2), g0/(val**2 + g0**2), mode='same')
                            for val in datas.energies]

            chiSquared += sum([(datas.intensities[k]/normFact - 
                    (s0 * self.resFunc(val, lorR, GauR, lorS, 1, resShift) + 
                    s1 * (lorS * conv1[k] + 
                    (1-lorS) * np.real(wofz(val + 1j*g0/(sigma*np.sqrt(2)))) / 
                    (sigma*np.sqrt(np.pi*2))) + x[j + 3]))**2 
                    /datas.errors[k]**2 * sum(datas.errors)**2 
                    for k, val in enumerate(datas.energies)])

        return chiSquared


    def scatFit(self):
    
        scatList1 = []
        for i, fileDatas in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            
            bkgdList = [0.008 for i, val in enumerate(fileDatas)]
            bkgdBounds = [(0., 0.4) for i, val in enumerate(fileDatas)]

            #_Minimization 
            scatList1.append(optimize.minimize(lambda x:
                    np.log(1 + self.fitFunc(x, fileDatas, i)),
                    [1, 0.5, 0.2] + bkgdList, 
                    bounds = [(0., 5), (0., 1), (0., 1)] + bkgdBounds,
                    callback=self.fitState))


            print('> Final result : ', flush=True)
            print('Fit g0 : ' + str(scatList1[i].x[0]), flush=True)
            print('Fit s0 : ' + str(scatList1[i].x[1]), flush=True)
            print('Fit s1 : ' + str(scatList1[i].x[2]), flush=True)


            self.scatFitList1 = scatList1

    #_Callback function for the algorithm
    def fitState(self, x):
        print('Step results :', flush=True)
        print('        {0:<50}= {1:.4f}'.format('g0', x[0]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('s0', x[1]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('s1', x[2]), flush=True)


    #_Function used to produce the plot of the fit
    def fittedFunc(self, datas, resShift, normFact, lorS, GauR, lorR, sigma, 
                                                g0, s0, s1, bkgd):
        

        conv1 = [fftconvolve(lorR/(val**2 + lorR**2), g0/(val**2 + g0**2), mode='same')
                        for val in datas.energies]

        Model = [s0 * self.resFunc(val, lorR, GauR, lorS, 1, resShift) + 
                s1 * (lorS * conv1[k] + 
                (1-lorS) * np.real(wofz(val + 1j*g0/(sigma*np.sqrt(2)))) /
                (sigma*np.sqrt(np.pi*2))) + bkgd
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
    def fitPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(1, 2)
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        xList = [x+1 for x in range(len(self.dataFiles))]

        g0List = []
        s0List = []
        s1List = []

        g0Err = []
        s0Err = []
        s1Err = []


        for k, fileDatas in enumerate(self.dataList):
            g0List.append(self.scatFitList1[k].x[0])
            g0Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[0])
            s0List.append(self.scatFitList1[k].x[1])
            s0Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[1])
            s1List.append(self.scatFitList1[k].x[2])
            s1Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[2])
            
            #_Plot of the g0 parameter of the fits
            ax = self.figure.add_subplot(mplGrid[:,0])
            ax.errorbar(xList[k], g0List[k], g0Err[k], marker='o', 
                    linestyle='None', label='g0')
            ax.set_ylabel(r'$\Gamma 0$', fontsize=18)
            plt.xticks(
                [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
                [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
                rotation=45, ha='right')
            ax.grid()

            #_Plot of the s0 parameter of the fits
            ax = self.figure.add_subplot(mplGrid[:,1])
            ax.errorbar(xList[k],s0List[k], s0Err[k], marker='o', 
                    linestyle='None', label='s0')
            ax.errorbar(xList[k],s1List[k], s1Err[k], marker='o', 
                    linestyle='None', label='s1')
            ax.set_ylabel(r'$Contributions$', fontsize=18)
            plt.xticks(
                [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
                [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
                rotation=45, ha='right')
            plt.legend(framealpha=0.5)
            ax.grid()

        plt.tight_layout()
        self.canvas.draw()

    def analysisPlot(self):
                
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
                    sigma = GauR/np.sqrt(2*np.log(2))
                    
                    #_Plot of the measured QENS signal
                    ax.errorbar(qDatas.energies, 
                           [x/normFact for x in qDatas.intensities],
                           [x/normFact for x in qDatas.errors], fmt='o')

                    #_Plot of the resolution function
                    ax.plot(qDatas.energies, 
                            [self.resFunc(val, lorR, GauR, lorS, 1, resShift) 
                                             for val in qDatas.energies], label='resolution')

                    #_Plot of the fitted lorentzian
                    g0 = self.scatFitList1[k].x[0]
                    s0 = self.scatFitList1[k].x[1]
                    s1 = self.scatFitList1[k].x[2]
                    bkgd = self.scatFitList1[k].x[j + 3]

                    ax.plot(qDatas.energies, [g0/(g0**2+val**2) * s1
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
            ax.set_ylim(1e-3, 1.2)
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()
        
        plt.legend(framealpha=0.5)
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
