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
                                #sigma=[val for val in datas.errors],
                                #p0 = [1, 1, 0.8, 20, 0], 
                                bounds=([0., 0., 0., 0., -10], [1, 10, 1., 10000, 10]), 
                                method='trf'))
            self.resFitList.append(resList)


    def fitFunc(self, x, datas, resShift, normFact, lorS, GauR, lorR, sigma):


        g0 = x[0]
        s0 = 1 - 2 * x[1] * (1 - (1 - (datas.qVal**2 * x[2]**2 / factorial(3) +
                                  datas.qVal**4 * x[2]**4 / factorial(5) -
                                  datas.qVal**6 * x[2]**6 / factorial(7) +
                                  datas.qVal**8 * x[2]**8 / factorial(9))/datas.qVal*x[2]))
        s1 = 1 - s0
        msd = x[3]
        bkgd = x[4]

        chiSquared = sum([(datas.intensities[k]/normFact - 
                np.exp(-datas.qVal**2 * msd) * (s0 * self.resFunc(val, lorR, GauR, 
                                                        lorS, 1, resShift) + 
                s1 * (lorS * fftconvolve(lorR/(val**2 + lorR**2), 
                g0/(val**2 + g0**2), mode='same') + 
                (1-lorS) * np.real(wofz(val + 1j*g0/(sigma*np.sqrt(2))) / 
                (sigma*np.sqrt(np.pi*2)))) + bkgd))**2 
                /datas.errors[k]**2
                for k, val in enumerate(datas.energies)]) 

        return chiSquared        

    def scatFit(self):
    
        for i, fileDatas in enumerate(self.dataList):
            scatList1 = []
            scatList2 = []
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
                            [1, 0.1, 1, 0.1, 0.003], callback=self.fitState,
                            niter=self.BHIter, minimizer_kwargs={'method':'BFGS'}))

                print('> Final result : ', flush=True)
                print('Fit q-value : ' + str(datas.qVal), flush=True)
                print('Fit Gamma : ' + str(scatList1[j].x[0]), flush=True)
                print('Fit p1*p2 : ' + str(scatList1[j].x[1]), flush=True)
                print('Fit distance btwn minima: ' + str(scatList1[j].x[2]), flush=True)
                print('Fit MSD : ' + str(scatList1[j].x[3]), flush=True)
                print('Fit background : ' + str(scatList1[j].x[4]) + '\n', flush=True)


            self.scatFitList1.append(scatList1)

    #_Callback function for the basinHopping algorithm
    def fitState(self, x, f, accepted):
        print('at minimum   %.4f     accepted : %d' % (f, int(accepted)), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Gamma', x[0]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('p1p2', x[1]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Vibrational MSD', x[3]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('Distance btwn sites', x[2]), flush=True)

        self.tempStats.append((x[0], x[3], x[2], f))
        
        if len(self.tempStats) == self.BHIter:
            self.tempStats = self.tempStats[8:]

            gammaStats = bayes_mvs([val[0] for val in self.tempStats if int(accepted) == 1])
            vibMSDStats = bayes_mvs([val[1] for val in self.tempStats if int(accepted) == 1])
            distStats = bayes_mvs([val[2] for val in self.tempStats if int(accepted) == 1])
            self.fitStats.append((gammaStats, vibMSDStats, distStats))

            self.tempStats = []

    #_Function used to produce the plot of the fit
    def fittedFunc(self, datas, resShift, normFact, lorS, GauR, lorR, sigma, 
                                                g0, p1p2, d, msd, bkgd):
        
        s0 = 1 - 2 * p1p2 * (1 - (1 - datas.qVal**2 * d**2 / factorial(3) +
                                  datas.qVal**4 * d**4 / factorial(5) -
                                  datas.qVal**6 * d**6 / factorial(7) +
                                  datas.qVal**8 * d**8 / factorial(9)))
        s1 = 1 - s0

        Model = [np.exp(-datas.qVal**2 * msd) * (s0 * self.resFunc(val, lorR, GauR, lorS, 
                                                                            1, resShift) + 
                s1 * (lorS * fftconvolve(lorR/(val**2 + lorR**2), 
                g0/(k**2 + g0**2), mode='same') + 
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
        mplGrid = gridspec.GridSpec(1, len(self.dataFiles)) #création d'une grille pour des subplots (ligne, colonnes)

        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2) 
        cmap = matplotlib.cm.get_cmap('winter')
        #juste pour ajouter de jolies couleurs

        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k], projection='3d')
            #création du subplot avec la 3d

            for i, datas in enumerate(fileDatas):
                normFact = self.resFitList[k][i][0][3]

                ax.plot(datas.energies, 
                                  [val/normFact for val in datas.intensities], 
                                  datas.qVal, zdir='y', c=cmap(normColors(datas.qVal)))
                #plot avec les trois axes comprenant les données et zdir définissant l'axe vertical

            ax.set_xlabel(r'$\hslash \omega (\mu eV)$')
            ax.set_ylabel(r'$q$')
            ax.set_zlabel(r'$S_{300K}(q, \omega)$')
            ax.set_ylim((0, 2))
            ax.set_zlim((0, 1))
            #définition des titres, on peut ajouter des paramètres sur les axes avec set_xlim ou set_xticks
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()

        plt.tight_layout()
        self.canvas.draw()
 

    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(2, 2)
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:1])
            qList = [val.qVal for val in self.dataList[0]]
            gammaList = []
            gammaErrors = []
            vibMSDList = []
            vibMSDErrors = []
            distList = []
            distErrors = []
            chi2List = []
            dataFilesNbr = range(len(self.dataFiles))

            for j, qDatas in enumerate(fileDatas):
                resShift = self.resFitList[k][j][0][4]
                normFact = self.resFitList[k][j][0][3]
                lorS = self.resFitList[k][j][0][2]
                GauR = self.resFitList[k][j][0][1]
                lorR = self.resFitList[k][j][0][0]

                g0 = self.scatFitList1[k][j].x[0]
                p1p2 = self.scatFitList1[k][j].x[1]
                d = self.scatFitList1[k][j].x[2]
                msd = self.scatFitList1[k][j].x[3]
                bkgd = self.scatFitList1[k][j].x[4]

                sigma = GauR/np.sqrt(2*np.log(2))

                gammaList.append(self.scatFitList1[k][j].x[0])
                gammaErrors.append(self.fitStats[(k+1)*j][0][2].statistic)

                vibMSDList.append(self.scatFitList1[k][j].x[3])
                vibMSDErrors.append(self.fitStats[(k+1)*j][1][2].statistic)

                distList.append(self.scatFitList1[k][j].x[2])
                distErrors.append(self.fitStats[(k+1)*j][2][2].statistic)

                chi2List.append(chisquare(self.fittedFunc(qDatas, resShift, normFact, lorS,
                                            GauR, lorR, sigma, g0, p1p2, d, msd, bkgd),
                                          [val/normFact for val in qDatas.intensities])[0])
                
            
            #_Plot of the chi squared of the fits
            ax = self.figure.add_subplot(mplGrid[:1,0])
            ax.plot(qList, chi2List, 
            label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], marker='o')
            ax.set_ylabel(r'$\chi^{2}$', fontsize=18)
            plt.legend(framealpha=0.5)
            ax.grid()

            #_Plot of the gammas of the system lorentzian
            ax = self.figure.add_subplot(mplGrid[:1,1])
            ax.errorbar(qList, gammaList, gammaErrors, 
            label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], marker='o')
            ax.set_ylim([0., 5])
            ax.set_ylabel(r'$\Gamma$', fontsize=18)
            ax.grid()
        
            #_Plot of the vibrational MSD of the system 
            ax = self.figure.add_subplot(mplGrid[1:,0])
            ax.errorbar(qList, vibMSDList, vibMSDErrors,  
            label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], marker='o')
            ax.set_xlabel(r'$q$', fontsize=18)
            ax.set_ylim([0., 5])
            ax.set_ylabel(r'$vib MSD(\AA^{2})$', fontsize=18)
            ax.grid()

            #_Plot of the distance between the two sites  
            ax = self.figure.add_subplot(mplGrid[1:,1])
            ax.errorbar(qList, distList, distErrors, 
            label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], marker='o')
            ax.set_xlabel(r'$q$', fontsize=18)
            ax.set_ylim([0., 5])
            ax.set_ylabel(r'$d (\AA)$', fontsize=18)
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
                    g0 = self.scatFitList1[k][j].x[0]
                    p1p2 = self.scatFitList1[k][j].x[1]
                    d = self.scatFitList1[k][j].x[2]
                    msd = self.scatFitList1[k][j].x[3]
                    bkgd = self.scatFitList1[k][j].x[4]

                    s1 = 2 * p1p2 * (1 - (1 - qDatas.qVal**2 * d**2 / factorial(3) +
                                  qDatas.qVal**4 * d**4 / factorial(5) -
                                  qDatas.qVal**6 * d**6 / factorial(7) +
                                  qDatas.qVal**8 * d**8 / factorial(9)))

                    ax.plot(qDatas.energies, [g0/(g0**2+val**2)*s1
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian')

                    #_Plot of the background
                    ax.plot(qDatas.energies, [bkgd for val in qDatas.energies], ls='--',
                            label='background') 
                    

                    #_Plot of the fitted incoherent structure factor
                    ax.plot(qDatas.energies, self.fittedFunc(qDatas, resShift, normFact, lorS, 
                                                    GauR, lorR, sigma, g0, p1p2, d, msd, bkgd), 
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
