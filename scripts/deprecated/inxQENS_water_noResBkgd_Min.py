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
from scipy.special import wofz, sph_jn
from scipy.stats import chisquare, bayes_mvs
from scipy.misc import factorial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib
import inxQENS_extract_resFunc_analysis as resAnalysis

class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resDataList = []
        self.resFitList = []
        self.meanResGauWidth = []
        self.scatFitList1 = []
        self.HO_dist = 0.96

        try:
            self.maxBesselOrder = int(karg['maxBesselOrder'])
        except ValueError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'Error: the "maxBesselOrder" parameter can\'t be converted to an' +
                            ' integer.')
            sys.exit()
        except KeyError:
            self.maxBesselOrder = 2
            infoMessage = QMessageBox.information(QWidget(), 'Parameter error',
                            'No value for the maximum order of Bessel function to be' +
                            ' used was given. \n\n' +
                            'Using maxBesselOrder=2 as default value.')

        try:
            self.ratioPower = float(karg['ratioPower'])
        except KeyError:
            self.ratioPower = 0.6
                    

        for i, dataFile in enumerate(self.dataFiles):
        
            #_Get the corresponding file at low temperature and integrate the curves
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the corresponding low temperature file for :\n ...' + 
                    dataFile[dataFile.rfind('/'):])
            resData = resAnalysis.getFitList()
            self.resFitList.append(resData[0])
            self.meanResGauWidth.append(resData[1])
            self.resDataList.append(resData[2])
        
            #_Get data from the file and store them into self.dataList 
            inxdata = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxdata)    

        #_Keep the data from wanted q-range
        qMin = min([data.qVal for data in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMin']) - x))
        qMax = min([data.qVal for data in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMax']) - x))
        for i, filedata in enumerate(self.dataFiles):
            self.dataList[i] = [val for val in self.dataList[i] if qMin <= val.qVal <= qMax]

        #_Discard the selected index
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])
            qDiscardList = [min([data.qVal for data in self.dataList[0]], 
	                key = lambda x: abs(float(val) - x)) for val in qDiscardList]
            for i, filedata in enumerate(self.dataFiles):
                self.dataList[i] = [val for val in self.dataList[i] 
                                    if val.qVal not in qDiscardList] 

        

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

        self.plot3DButton = QPushButton('3D Plot')
        self.plot3DButton.clicked.connect(self.plot3D)

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
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.fitButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.resButton)
        self.setLayout(layout)




#_Everything needed for the fit
    def fitFunc(self, x, filedata, i):

        bkgdL = []

        chiSquared = 0
        gt = x[0] 
        gr = x[1]
        s0 = x[2]
        st = x[3]
        sr = x[4]
        msd = x[5]
        diffFactor = x[6]

        for j, data in enumerate(filedata):

            resShift = self.resFitList[i][j][0][4]
            normFact = self.resFitList[i][j][0][3]
            lorS = self.resFitList[i][j][0][2]
            GauR = self.resFitList[i][j][0][1]
            lorR = self.resFitList[i][j][0][0]
            bkgd = 0
            sigma = GauR/np.sqrt(2*np.log(2))
            pBkgd = x[j + 7]

            bessel = sph_jn(self.maxBesselOrder, self.HO_dist * data.qVal)
            
            funSquared = sum([(data.intensities[k] / normFact - 
                    np.exp(-data.qVal**2 * msd) * 
                    ((s0 + sr*bessel[0][0]**2)  
                    * resAnalysis.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) + 
 
                    st * (lorS * (lorR + gt*data.qVal**diffFactor) 
                    / (np.pi * ((lorR + gt*data.qVal**diffFactor)**2 + val**2))
                    + (1-lorS) * np.real(wofz((val + 1j*gt*data.qVal**diffFactor)
                    /(sigma*np.sqrt(2)))) / 
                    (sigma*np.sqrt(np.pi*2)) + bkgd) + 

                    sr * (np.sum([(2*l + 1) * bessel[0][l]**2 * 
                    (lorS * (lorR + l*(l+1)*gr) / (np.pi * ((lorR + l*(l+1)*gr)**2 + val**2))
                    + (1-lorS) * np.real((wofz(val + 1j*l*(l+1)*gr)/(sigma*np.sqrt(2)))) / 
                    (sigma*np.sqrt(np.pi*2)) + bkgd) 
                    for l in range(1, self.maxBesselOrder + 1)])) + pBkgd))**2 

                    / (data.errors[k]/normFact)**2
                    #/ (data.intensities[k] / normFact)**2 
                    #/ (sum(data.intensities) / sum(data.errors))**2 
                    for k, val in enumerate(data.energies)])

            #chiSquared += np.log(1 + funSquared) #_Cauchy loss function 
            #chiSquared += 2 * ((1 + funSquared)**0.5 - 1) #_Soft_l1 loss function
            chiSquared += funSquared

        return chiSquared


    def scatFit(self):
    
        scatList1 = []
        for i, filedata in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            
            #_Minimization 
            scatList1.append(optimize.minimize(lambda x:
                    self.fitFunc(x, filedata, i),
                    [1, 4, 0.1, 0.4, 0.5, 1, 2] + [0.005 for val in range(len(filedata))], 
                    bounds = [(0.0, 30), (0.0, 30), (0., 1), (0., 1), (0., 1), (0., 10), (0., 2.5)] 
                              + [(0., 0.5) for val in range(len(filedata))],
                    options={'eps':1e-10, 'maxcor':250, 'maxls':100, 
                             'maxfun':100000, 'maxiter':100000},
                    callback=self.fitState))

            print('> Final result : ', flush=True)
            print('Fit gt : ' + str(scatList1[i].x[0]), flush=True)
            print('Fit gr : ' + str(scatList1[i].x[1]), flush=True)
            print('Fit s0 : ' + str(scatList1[i].x[2]), flush=True)
            print('Fit st : ' + str(scatList1[i].x[3]), flush=True)
            print('Fit sr : ' + str(scatList1[i].x[4]), flush=True)
            print('Fit msd : ' + str(scatList1[i].x[5]), flush=True)
            print('Fit Lambda : ' + str(scatList1[i].x[6]), flush=True)


            self.scatFitList1 = scatList1

    #_Callback function for the algorithm
    def fitState(self, x):
        print('Step results :', flush=True)
        print('        {0:<50}= {1:.4f}'.format('gt', x[0]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('gr', x[1]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('s0', x[2]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('st', x[3]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('sr', x[4]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('msd', x[5]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Lambda', x[6]), flush=True)


    #_Function used to produce the plot of the fit
    def fittedFunc(self, data, i, resShift, normFact, lorS, GauR, lorR, sigma, 
                                            gt, gr, s0, st, sr, bkgd, msd, pBkgd, diffFactor):


        bessel = sph_jn(self.maxBesselOrder, self.HO_dist * data.qVal)

        Model = [np.exp(-data.qVal**2 * msd) *
                ((s0 + sr * bessel[0][0]**2)   
                * resAnalysis.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) + 

                st * (lorS * (lorR + gt*data.qVal**diffFactor) 
                / (np.pi * ((lorR + gt*data.qVal**diffFactor)**2 + val**2))
                + (1-lorS) * np.real(wofz(val + 1j*gt*data.qVal**diffFactor
                / (sigma*np.sqrt(2)))) /
                (sigma*np.sqrt(np.pi*2)) + bkgd) +

                sr * (np.sum([(2*l + 1) * bessel[0][l]**2 *
                (lorS * (lorR + l*(l+1)*gr) / (np.pi * ((lorR + l*(l+1)*gr)**2 + val**2))
                + (1-lorS) * np.real(wofz(val + 1j*l*(l+1)*gr/(sigma*np.sqrt(2)))) /
                (sigma*np.sqrt(np.pi*2)) + bkgd) 
                for l in range(1, self.maxBesselOrder + 1)])) + pBkgd) 
 
                for k, val in enumerate(data.energies)] 

        return Model        



#_Definitions of the slots for the plot window
    def plot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([data.qVal for data in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - x))
        #_Plot the data for selected q value normalized with integrated curves at low temperature
        for k, filedata in enumerate(self.dataList):
            for j, qdata in enumerate(filedata):
                if qdata.qVal == qValToShow:
                    normFact = self.resFitList[k][j][0][3]
                    GauR = self.resFitList[k][j][0][1]
                    ax.errorbar(qdata.energies, 
                           [x / normFact for x in qdata.intensities],
                           [x/self.resFitList[k][j][0][3] for x in qdata.errors], fmt='o')
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
        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))

        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for k, filedata in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k], projection='3d')


            for i, data in enumerate(filedata):
                normFact = self.resFitList[k][i][0][3]
                GauR = self.resFitList[k][i][0][1]

                ax.plot(data.energies, 
                                  [val/normFact 
                                            for val in data.intensities], 
                                  data.qVal, zdir='y', c=cmap(normColors(data.qVal)))

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

        mplGrid = gridspec.GridSpec(1, 2)
        #_Plot the data for selected q value normalized with integrated curves at low temperature
        xList = np.array([x+1 for x in range(len(self.dataFiles))])

        gtList = []
        grList = []
        s0List = []
        stList = []
        srList = []

        gtErr = []
        grErr = []
        s0Err = []
        stErr = []
        srErr = []


        for k, filedata in enumerate(self.dataList):
            gtList.append(self.scatFitList1[k].x[0])
            gtErr.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[0])
            grList.append(self.scatFitList1[k].x[1])
            grErr.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[1])
            s0List.append(self.scatFitList1[k].x[2])
            s0Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[2])
            stList.append(self.scatFitList1[k].x[3])
            stErr.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[3])
            srList.append(self.scatFitList1[k].x[4])
            srErr.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[4])

        #_Plot of the gamma parameters of the fits
        ax = self.figure.add_subplot(mplGrid[:,0])
        ax.errorbar(xList - 0.1, gtList, gtErr, marker='o', 
                linestyle='-', label='translational')
        ax.errorbar(xList, grList, grErr, marker='^', 
                linestyle='-', label='rotational')
        ax.set_ylabel(r'$\Gamma$', fontsize=18)
        plt.xticks(
            [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
            [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
            rotation=45, ha='right')
        ax.grid(True)

        #_Plot of the a parameters
        ax = self.figure.add_subplot(mplGrid[:,1])
        ax.errorbar(xList - 0.1, stList, marker='o', 
                linestyle='-', label='st')
        ax.errorbar(xList, srList, marker='^', 
                linestyle='-', label='sr')
        ax.errorbar(xList + 0.1, s0List, marker='x', 
                linestyle='-', label='s0')
        ax.set_ylabel(r'$ Lorentzians \ contributions$', fontsize=14)
        ax.set_ylim(0, 1)
        plt.xticks(
            [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
            [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
            rotation=45, ha='right')
        plt.legend(framealpha=0.5)
        ax.grid(True)

        plt.tight_layout()
        self.canvas.draw()

    def fitPlot(self):
                
        plt.gcf().clear()     
        qValToShow = min([data.qVal for data in self.dataList[0]], 
                            key = lambda x : abs(float(self.lineEdit.text()) - x))


        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))
        #_Plot the data for selected q value normalized with integrated curves at low temperature
        for k, filedata in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k])
            for j, qdata in enumerate(filedata):
                if qdata.qVal == qValToShow:
                    resShift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][3]
                    lorS = self.resFitList[k][j][0][2]
                    GauR = self.resFitList[k][j][0][1]
                    lorR = self.resFitList[k][j][0][0]
                    bkgd = 0
                    sigma = GauR/np.sqrt(2*np.log(2))
                    
                    #_Plot of the measured QENS signal
                    ax.errorbar(qdata.energies, 
                           [(GauR/self.meanResGauWidth[k])**self.ratioPower * 
                                                x/normFact for x in qdata.intensities],
                           [x/normFact for x in qdata.errors], fmt='o')

                    #_Plot of the resolution function
                    ax.plot(qdata.energies, 
                            [resAnalysis.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) 
                                             for val in qdata.energies], label='resolution')

                    #_Plot of the fitted lorentzian
                    gt = self.scatFitList1[k].x[0] 
                    gr = self.scatFitList1[k].x[1]
                    s0 = self.scatFitList1[k].x[2]
                    st = self.scatFitList1[k].x[3]
                    sr = self.scatFitList1[k].x[4]
                    msd = self.scatFitList1[k].x[5]
                    diffFactor = self.scatFitList1[k].x[6]
                    pBkgd = self.scatFitList1[k].x[j + 7]


                    ax.plot(qdata.energies, [gt/(gt**2+val**2) * st
                                        for val in qdata.energies], 
                                        ls='--', label='translational')
                    ax.plot(qdata.energies, [gr/(gr**2+val**2) * sr
                                        for val in qdata.energies], 
                                        ls='--', label='rotational')
                    ax.axhline(y=pBkgd, label='background')                    


                    #_Plot of the fitted incoherent structure factor
                    ax.plot(qdata.energies, self.fittedFunc(qdata, k, resShift, normFact, lorS, 
                            GauR, lorR, sigma, gt, gr, s0, st, sr, bkgd, msd, pBkgd, diffFactor), 
                            label='convolution', linewidth=2, color='orangered', zorder=5)
                    

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
        
        qValToShow = min([data.qVal for data in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - x))
        #_Plot the data for selected q value normalized with integrated curves at low temperature
        for k, filedata in enumerate(self.resDataList):
            for j, qdata in enumerate(filedata):
                if qdata.qVal == qValToShow:
    
                    resShift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][3]
                    lorS = self.resFitList[k][j][0][2]
                    GauR = self.resFitList[k][j][0][1]
                    lorR = self.resFitList[k][j][0][0]
                    bkgd = self.resFitList[k][j][0][5]

                    ax.plot(qdata.energies, [val/normFact for val in qdata.intensities])
    
                    ax.plot(qdata.energies, 
                            [resAnalysis.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd)
                                             for val in qdata.energies])


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
