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
        self.BHIter = 25
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
        p1 = x[1]
        d = x[2]
        msd = x[3]
        for j, datas in enumerate(fileDatas):

            s0 = 1 - 2 * (p1 * (1-p1)) * (1 - (1 - 
                                      datas.qVal**2 * d**2 / factorial(3) +
                                      datas.qVal**4 * d**4 / factorial(5) -
                                      datas.qVal**6 * d**6 / factorial(7) +
                                      datas.qVal**8 * d**8 / factorial(9) -
                                      datas.qVal**10 * d**10 / factorial(11)))
            s1 = 1 - s0

            resShift = self.resFitList[i][j][0][4]
            normFact = self.resFitList[i][j][0][3]
            lorS = self.resFitList[i][j][0][2]
            GauR = self.resFitList[i][j][0][1]
            lorR = self.resFitList[i][j][0][0]
            sigma = GauR/np.sqrt(2*np.log(2))

            
            chiSquared += sum([(datas.intensities[k]/normFact - 
                    np.exp(-datas.qVal**2 * msd) * (s0 * self.resFunc(val, lorR, GauR, 
                                                            lorS, 1, resShift) + 
                    s1 * (lorS * fftconvolve(lorR/(val**2 + lorR**2), 
                    g0/(val**2 + g0**2), mode='same') + 
                    (1-lorS) * np.real(wofz(val + 1j*g0/(sigma*np.sqrt(2)))) / 
                    (sigma*np.sqrt(np.pi*2))) + x[j + 4]))**2 
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
            scatList1.append(optimize.basinhopping(lambda x:
                    np.log(1 + self.fitFunc(x, fileDatas, i)),
                    [1, 0.5, 1, 1.1] + bkgdList, 
                    niter = self.BHIter,
                    minimizer_kwargs = {'method':'L-BFGS-B'},
                            #'bounds':'[(0., 20), (0., 1), (0., 20), (0., 10)] + bkgdBounds'},
                    accept_test = self.fitAccept,
                    callback=self.fitState))


            print('> Final result : ', flush=True)
            print('Fit Gamma : ' + str(scatList1[i].x[0]), flush=True)
            print('Fit p1 : ' + str(scatList1[i].x[1]), flush=True)
            print('Fit distance between energy minima : ' + str(scatList1[i].x[2]), flush=True)
            print('Fit MSD : ' + str(scatList1[i].x[3]) + '\n', flush=True)


            self.scatFitList1 = scatList1

    #_Accept test for the basinHopping algorithm
    def fitAccept(self, f_new, x_new, f_old, x_old):
        testList = [False if val < 0 else True for val in x_new]
        
        return False if False in testList else True
            

    #_Callback function for the algorithm
    def fitState(self, x, f, success):
        print('at minimum  %.4f  accepted : %d' % (f, int(success)), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Gamma', x[0]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('p1', x[1]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Vibrational MSD', x[3]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Distance btwn sites', x[2]), flush=True)


    #_Function used to produce the plot of the fit
    def fittedFunc(self, datas, resShift, normFact, lorS, GauR, lorR, sigma, 
                                                g0, p1, d, msd, bkgd):
        
        s0 = 1 - 2 * (p1 * (1-p1)) * (1 - (1- datas.qVal**2 * d**2 / factorial(3) +
                                  datas.qVal**4 * d**4 / factorial(5) -
                                  datas.qVal**6 * d**6 / factorial(7) +
                                  datas.qVal**8 * d**8 / factorial(9)))
        s1 = 1 - s0

        Model = [np.exp(-datas.qVal**2 * msd) * (s0 * self.resFunc(val, lorR, GauR, lorS, 
                                                                            1, resShift) + 
                s1 * (lorS * fftconvolve(lorR/(val**2 + lorR**2), 
                g0/(val**2 + g0**2), mode='same') + 
                (1-lorS) * np.real(wofz(val + 1j*g0/(sigma*np.sqrt(2))) / 
                (sigma*np.sqrt(np.pi*2)))) + bkgd)
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

        mplGrid = gridspec.GridSpec(2, 2)
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        xList = [x+1 for x in range(len(self.dataFiles))]

        gammaList = []
        p1List = []
        msdList = []
        distList = []

        gammaErr = []
        p1Err = []
        msdErr = []
        distErr = []


        for k, fileDatas in enumerate(self.dataList):
            gammaList.append(self.scatFitList1[k].x[0])
            gammaErr.append(np.sqrt(np.diag(
                        self.scatFitList1[k].lowest_optimization_result.hess_inv.todense()))[0])
            p1List.append(self.scatFitList1[k].x[1])
            p1Err.append(np.sqrt(np.diag(
                        self.scatFitList1[k].lowest_optimization_result.hess_inv.todense()))[1])
            distList.append(self.scatFitList1[k].x[2])
            distErr.append(np.sqrt(np.diag(
                        self.scatFitList1[k].lowest_optimization_result.hess_inv.todense()))[2])
            msdList.append(self.scatFitList1[k].x[3])
            msdErr.append(np.sqrt(np.diag(
                        self.scatFitList1[k].lowest_optimization_result.hess_inv.todense()))[3])

            
            #_Plot of the gamma parameter of the fits
            ax = self.figure.add_subplot(mplGrid[:1,0])
            ax.errorbar(xList[k], gammaList[k], gammaErr[k],  marker='o', 
                    linestyle='None', label=self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_ylabel(r'$\Gamma$', fontsize=18)
            ax.set_xlim([0, len(self.dataFiles) +1])
            ax.grid()

            #_Plot of the gamma parameter of the fits
            ax = self.figure.add_subplot(mplGrid[:1,1])
            ax.errorbar(xList[k], p1List[k], p1Err[k], marker='o', 
                    linestyle='None', label=self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_ylabel(r'$p1$', fontsize=18)
            ax.set_xlim([0, len(self.dataFiles) +1])
            ax.grid()

            #_Plot of the msd parameter of the fits
            ax = self.figure.add_subplot(mplGrid[1:,0])
            ax.errorbar(xList[k], msdList[k], msdErr[k],  marker='o', 
                    linestyle='None', label=self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_ylabel(r'$vib \ MSD$', fontsize=18)
            ax.set_xlim([0, len(self.dataFiles) +1])
            plt.xticks(
                [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
                [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
                rotation=45, ha='right')
            ax.grid()

            #_Plot of the d parameter of the fits
            ax = self.figure.add_subplot(mplGrid[1:,1])
            ax.errorbar(xList[k], distList[k], distErr[k], marker='o', 
                    linestyle='None', label=self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_ylabel(r'$d$', fontsize=18)
            ax.set_xlim([0, len(self.dataFiles) +1])
            plt.xticks(
                [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
                [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
                rotation=45, ha='right')
            ax.grid()

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
                    p1 = self.scatFitList1[k].x[1]
                    d = self.scatFitList1[k].x[2]
                    msd = self.scatFitList1[k].x[3]
                    bkgd = self.scatFitList1[k].x[j + 4]

                    s1 = 2 * p1 * (1-p1) * (1 - (1 - qDatas.qVal**2 * d**2 / factorial(3) +
                                  qDatas.qVal**4 * d**4 / factorial(5) -
                                  qDatas.qVal**6 * d**6 / factorial(7) +
                                  qDatas.qVal**8 * d**8 / factorial(9))/(qDatas.qVal*d))

                    ax.plot(qDatas.energies, [g0/(g0**2+val**2) * s1
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian')

                    #_Plot of the background
                    ax.plot(qDatas.energies, [bkgd for val in qDatas.energies], ls='--',
                            label='background') 
                    

                    #_Plot of the fitted incoherent structure factor
                    ax.plot(qDatas.energies, self.fittedFunc(qDatas, resShift, normFact, lorS, 
                                                    GauR, lorR, sigma, g0, p1, d, msd, bkgd), 
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
