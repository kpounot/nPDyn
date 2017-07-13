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

class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []
        self.scatFitList1 = []
        self.HO_dist = 0.98

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
            qDiscardPattern = re.compile(r'[ ,:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])
            qDiscardList = [min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x: abs(float(val) - x)) for val in qDiscardList]
            for i, fileDatas in enumerate(self.dataFiles):
                self.dataList[i] = [val for val in self.dataList[i] 
                                    if val.qVal not in qDiscardList] 
                self.resList[i] = [val for val in self.resList[i] 
                                    if val.qVal not in qDiscardList] 

        

        self.resFit()
        self.meanResGauWidth = []
        for k, fileDatas in enumerate(self.dataFiles):
            self.meanResGauWidth.append(np.mean([val[0][1] 
                for i, val in enumerate(self.resFitList[k]) if self.dataList[k][i].qVal >= 0.6]))

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
                                #p0 = [0.5, 1, 0.8, 50, 0], 
                                bounds=([0., 0., 0., 0., -10, 0.],  
                                        [2, 10, 1., 5000, 10, 1]),
                                max_nfev=1000000,
                                method='trf'))
            self.resFitList.append(resList)

        
    def fitFunc(self, x, datas, i, j):

        bkgdL = []

        chiSquared = 0
        gt = x[0] 
        gr = x[1]
        s0 = x[2]
        st = x[3]
        sr = x[4]
        msd = x[5]
        pBkgd =x[6]
        fLambda = x[7]

        resShift = self.resFitList[i][j][0][4]
        normFact = self.resFitList[i][j][0][3]
        lorS = self.resFitList[i][j][0][2]
        GauR = self.resFitList[i][j][0][1]
        lorR = self.resFitList[i][j][0][0]
        bkgd = self.resFitList[i][j][0][5]
        sigma = GauR/np.sqrt(2*np.log(2))

        bessel = sph_jn(self.maxBesselOrder, self.HO_dist * datas.qVal)
        
        funSquared = sum([((GauR/self.meanResGauWidth[i] * datas.intensities[k]/normFact) - 
                np.exp(-datas.qVal**2 * msd**2) * 
                ((s0 + sr * bessel[0][0]**2) *  
                self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) + 

                st * (lorS * (lorR + gt*datas.qVal**fLambda) / 
                (np.pi * ((lorR + gt*datas.qVal**fLambda)**2 + val**2))
                + (1-lorS) * np.real(wofz(val + 1j*gt*datas.qVal**fLambda/(sigma*np.sqrt(2)))) / 
                (sigma*np.sqrt(np.pi*2)) + bkgd) + 

                sr * (np.sum([(2*l + 1) * bessel[0][l]**2 * 
                (lorS * (lorR + l*(l+1)*gr) / (np.pi * ((lorR + l*(l+1)*gr)**2 + val**2))
                + (1-lorS) * np.real(wofz(val + 1j*l*(l+1)*gr/(sigma*np.sqrt(2)))) / 
                (sigma*np.sqrt(np.pi*2)) + bkgd) 
                for l in range(1, self.maxBesselOrder + 1)])) + pBkgd))**2  

                / (datas.errors[k]/normFact)**2
                #/ (datas.intensities[k] / normFact)**2 
                #* (sum(datas.intensities) / sum(datas.errors))**2 
                for k, val in enumerate(datas.energies)])

        #chiSquared += np.log(1 + funSquared) #_Cauchy loss function 
        #chiSquared += 2 * ((1 + funSquared)**0.5 - 1) #_Soft_l1 loss function
        chiSquared += funSquared

        return chiSquared


    def scatFit(self):
    
        scatList1 = []
        for i, fileDatas in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
           
            
            #_Minimization 
            scatList = []
            for j, datas in enumerate(fileDatas):
                scatList.append(optimize.minimize(lambda x:
                    self.fitFunc(x, datas, i, j),
                    [1, 4, 0.5, 0.2, 0.3, 1, 0.001, 2], 
                    bounds = [  (0., 30), (0., 30), 
                                (0., 1), (0., 1), (0., 1), (0., 10), (0., 0.5), (0., 3)]))

                print('\n> Final result for qVal = %s: ' % datas.qVal, flush=True)
                print('Fit gt : ' + str(scatList[j].x[0]), flush=True)
                print('Fit gr : ' + str(scatList[j].x[1]), flush=True)
                print('Fit s0 : ' + str(scatList[j].x[2]), flush=True)
                print('Fit st : ' + str(scatList[j].x[3]), flush=True)
                print('Fit sr : ' + str(scatList[j].x[4]), flush=True)
                print('Fit msd : ' + str(scatList[j].x[5]), flush=True)
                print('Fit pBkgd : ' + str(scatList[j].x[6]), flush=True)
                print('Fit fLambda : ' + str(scatList[j].x[7]), flush=True)


            self.scatFitList1.append(scatList)


    #_Function used to produce the plot of the fit
    def fittedFunc(self, datas, i, resShift, normFact, lorS, GauR, lorR, sigma, 
                                                gt, gr, s0, st, sr, bkgd, msd, pBkgd, fLambda):


        bessel = sph_jn(self.maxBesselOrder, self.HO_dist * datas.qVal)

        Model = [np.exp(-datas.qVal**2 * msd**2) *
                ((s0 + sr * bessel[0][0]**2)  * 
                self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) + 

                st * (lorS * (lorR + gt*datas.qVal**fLambda) 
                / (np.pi * ((lorR + gt*datas.qVal**fLambda)**2 + val**2))
                + (1-lorS) * np.real(wofz(val + 1j*gt*datas.qVal**fLambda/(sigma*np.sqrt(2)))) /
                (sigma*np.sqrt(np.pi*2)) + bkgd) +

                sr * (np.sum([(2*l + 1) * bessel[0][l]**2 *
                (lorS * (lorR + l*(l+1)*gr) / (np.pi * ((lorR + l*(l+1)*gr)**2 + val**2))
                + (1-lorS) * np.real(wofz(val + 1j*l*(l+1)*gr/(sigma*np.sqrt(2)))) /
                (sigma*np.sqrt(np.pi*2)) + bkgd) 
                for l in range(1, self.maxBesselOrder + 1)])) + pBkgd) 
 
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
                GauR = self.resFitList[k][i][0][1]

                ax.plot(datas.energies, 
                                  [val*(GauR/self.meanResGauWidth)
                                    /normFact for val in datas.intensities], 
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

        mplGrid = gridspec.GridSpec(1, 2)
        qList = [val.qVal for val in self.dataList[0]]

        for k, fileDatas in enumerate(self.dataList):
            gtList = []
            grList = []
            gtErr = []
            grErr = []
            
            for j, data in enumerate(fileDatas):    
                gtList.append(self.scatFitList1[k][j].x[0])
                gtErr.append(np.sqrt(np.diag(self.scatFitList1[k][j].hess_inv.todense()))[0])
                grList.append(self.scatFitList1[k][j].x[1])
                grErr.append(np.sqrt(np.diag(self.scatFitList1[k][j].hess_inv.todense()))[1])

            
            #_Plot of the gt parameter of the fits
            ax1 = self.figure.add_subplot(mplGrid[:,0])
            ax1.plot(qList, gtList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax1.set_ylabel(r'$\Gamma $', fontsize=18)
            ax1.set_xlabel(r'$Translational $', fontsize=14)
            ax1.grid(True)

            #_Plot of the gr parameter of the fits
            ax2 = self.figure.add_subplot(mplGrid[:,1])
            ax2.plot(qList, grList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax2.set_xlabel(r'$Rotational $', fontsize=14)
            ax2.grid(True)


        plt.legend()
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
                    
                    #_Plot of the measured QENS signal
                    ax.errorbar(qDatas.energies, 
                           [x/normFact for x in qDatas.intensities],
                           [x/normFact for x in qDatas.errors], fmt='o')

                    #_Plot of the resolution function
                    ax.plot(qDatas.energies, 
                            [self.resFunc(val, lorR, GauR, lorS, 1, resShift, bkgd) 
                                             for val in qDatas.energies], label='resolution')

                    #_Plot of the fitted lorentzian
                    gt = self.scatFitList1[k][j].x[0] 
                    gr = self.scatFitList1[k][j].x[1]
                    s0 = self.scatFitList1[k][j].x[2]
                    st = self.scatFitList1[k][j].x[3]
                    sr = self.scatFitList1[k][j].x[4]
                    msd = self.scatFitList1[k][j].x[5]
                    pBkgd = self.scatFitList1[k][j].x[6]
                    fLambda = self.scatFitList1[k][j].x[7]


                    ax.plot(qDatas.energies, [gt/(gt**2+val**2) * st
                                        for val in qDatas.energies], 
                                        ls='--', label='translational')
                    ax.plot(qDatas.energies, [gr/(gr**2+val**2) * sr
                                        for val in qDatas.energies], 
                                        ls='--', label='rotational')
                    ax.axhline(y=pBkgd, label='background')


                    #_Plot of the fitted incoherent structure factor
                    ax.plot(qDatas.energies, self.fittedFunc(qDatas, k, resShift, normFact, lorS, 
                                        GauR, lorR, sigma, gt, gr, s0, 
                                        st, sr, bkgd, msd, pBkgd, fLambda), 
                            label='convolution', linewidth=2, color='orangered')
                    

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
