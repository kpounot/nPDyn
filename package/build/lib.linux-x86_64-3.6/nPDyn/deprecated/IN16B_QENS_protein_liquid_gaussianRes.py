'''
    This script is meant to be used with QENS data from IN16B.
    The script will first ask for the resolution file (usually vanadium QENS data) that will be
    used to fit a resolution function using 'IN16B_QENS_resFunc.py' script.
    Then, the file containing the fitted parameters for D2O background is asked.
    Eventually, the fitting procedure is runned if 'new fit' is set to True and the 
    plotting window is showed.
'''

import sys, os, pickle as pk
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
from scipy import optimize
from scipy.special import wofz
from scipy.signal import fftconvolve
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib

import h5process

import IN16B_QENS_resFunc_fullGaussian as resFuncAnalysis
import D2O_params_from_IN6 as D2O_IN6
import IN16B_QENS_fittedEC_substraction as fitECCorr

class Window(QDialog):
    def __init__(self, dataFiles, resFitList, D2OParams):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        if type(dataFiles) == str:
            self.dataFiles  = [dataFiles]
        else:
            self.dataFiles = dataFiles

        self.resFitList = resFitList
        self.D2OParams = D2OParams
        self.resFunc    = resFuncAnalysis.Window().resFunc
        self.protFitList = []

        #_Get datas from the file and store them into dataList 
        self.dataList = h5process.processData(dataFiles, karg['binS'], averageFiles=False, FWS=False)

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

        #_Get D2O linewidth tabled values from IN6
        try:
            self.volFraction = float(karg['volFraction'])
            self.sD2O        = D2O_IN6.getD2Odata(self.volFraction)

        except ValueError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'Error: the "volFraction" parameter can\'t be converted to an' +
                            ' integer.')
            sys.exit()

        except KeyError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'No value for volume fraction of D2O was given.\n' +
                            'Please use volFraction=[yourValue] in additional parameters pane.\n')
            sys.exit()
        

        #_Empty cell substraction, if ECCorrection is set to True by the user
        try:
            EC_scaleFactor = float(karg['ECCorr'])

            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file containing the QENS of empty cell...') 
            fileToOpen = QFileDialog().getOpenFileName()[0]
            ECData = h5process.processData(fileToOpen, karg['binS'])

            ECCorr = fitECCorr.ECCorr(self.dataList, self.resFitList, ECData)
            self.dataList = ECCorr.substract_ECData(EC_scaleFactor)


        except KeyError:
            print('No value for "ECCorr" parameter was set, no empty cell correction was performed.\n',
                    flush=True)
         


        if karg['new_fit']=='True':
            #_Get the file in which the fitted parameters are to be saved
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file in which to save the fitted parameters...') 
            self.paramsFile = QFileDialog().getSaveFileName()[0]
            self.basinHopping_fit()
        else:
            #_Get the file containing the fitted parameters
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file containing the fitted parameters...') 
            paramsFile = QFileDialog().getOpenFileName()[0]
            with open(paramsFile, 'rb') as params:
                self.protFitList = pk.Unpickler(params).load()
 


#_Construction of the GUI
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        

        #_Add some interactive elements
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        self.analysisButton = QPushButton('Analysis')
        self.analysisButton.clicked.connect(self.analysisPlot)

        self.plot3DButton = QPushButton('3D Plot')
        self.plot3DButton.clicked.connect(self.plot3D)

        self.fitButton = QPushButton('Fit')
        self.fitButton.clicked.connect(self.fitPlot)

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
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.fitButton)
        self.setLayout(layout)

#_Everything needed for the fit
    def fitFunc(self, params, dataSet, fileIdx, qIdx, returnCost=True):

        resParams     = self.resFitList[fileIdx][qIdx][0]
        S             = resParams[1]
        normF         = resParams[0]
        temp          = dataSet.temp
        X = dataSet.energies
        aD2O = self.D2OParams[fileIdx][qIdx].x[1]

        gGlo = params[1] #* dataSet.qVals[qIdx]**2
        gInt = params[2] #* dataSet.qVals[qIdx]**2 / (1 + params[2] * dataSet.qVals[qIdx]**2 * params[-1])
        gD2O = self.sD2O(temp, dataSet.qVals[qIdx])

        cost = 0

        voigt_G_g0 =    (wofz((X + 1j*gGlo) / (resParams[2] * np.sqrt(2))).real 
                            / (resParams[2] * np.sqrt(2*np.pi)))

        voigt_G_g1 =    (wofz((X + 1j*gGlo) / (resParams[3] * np.sqrt(2))).real 
                            / (resParams[3] * np.sqrt(2*np.pi)))

        voigt_I_g0 =    (wofz((X + 1j*(gInt+gGlo)) / (resParams[2] * np.sqrt(2))).real 
                            / (resParams[2] * np.sqrt(2*np.pi)))

        voigt_I_g1 =    (wofz((X + 1j*(gInt+gGlo)) / (resParams[3] * np.sqrt(2))).real 
                            / (resParams[3] * np.sqrt(2*np.pi)))

        voigt_D2O_g0 =  (wofz((X + 1j*gD2O) / (resParams[2] * np.sqrt(2))).real 
                            / (resParams[2] * np.sqrt(2*np.pi)))

        voigt_D2O_g1 =  (wofz((X + 1j*gD2O) / (resParams[3] * np.sqrt(2))).real 
                            / (resParams[3] * np.sqrt(2*np.pi)))

        model   = ( params[3] * ( params[0] * (S*voigt_G_g0 + (1-S)*voigt_G_g1)
                                  + (1-params[0]) * (S*voigt_I_g0 + (1-S)*voigt_I_g1)) 
                    + self.volFraction * aD2O * (S*voigt_D2O_g0 + (1-S)*voigt_D2O_g1)
                    + resParams[-1]) 

        cost += np.sum( (dataSet.intensities[qIdx] / normF - model)**2  
                        / (dataSet.errors[qIdx] / normF)**2 )  

        if returnCost:
            return cost
        else:
            return model

                
    def basinHopping_fit(self):
    
        for i, dataSet in enumerate(self.dataList):

            print("\nFitting data for file: " + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(50*"_", flush=True)

            fitList = []
            for qIdx, qWiseData in enumerate(dataSet.intensities):

                bounds      = [(0., 1), (0., np.inf), (0., np.inf), (0., np.inf)] 
                if qIdx == 0:
                    params_init = [0.8, 1, 10, 0.3]
                else:
                    params_init = fitList[qIdx-1].x

                fitList.append(optimize.basinhopping(self.fitFunc, 
                                params_init,
                                niter=self.BH_iter,
                                niter_success=0.25*self.BH_iter,
                                disp=True,
                                minimizer_kwargs = { 'args':(dataSet, i, qIdx, True),
                                                     'bounds':bounds} ))

                r  = "\nFinal result for q-value %.2f:\n" % dataSet.qVals[qIdx]
                r += "    a0   = %.2f\n" % fitList[qIdx].x[0]
                r += "    g0   = %.2f\n" % fitList[qIdx].x[1]
                r += "    g1   = %.2f\n" % fitList[qIdx].x[2]
                r += "    beta = %.2f\n" % fitList[qIdx].x[3]

                print(r, flush=True)

            self.protFitList.append(fitList)

        with open(self.paramsFile, 'wb') as fittedParamsFile:
            myFile = pk.Pickler(fittedParamsFile)
            modelFitList = self.protFitList
            myFile.dump(modelFitList)

        return self.protFitList 

        
#_Definitions of the slots for the plot window
    def plot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min(self.dataList[0].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(self.dataList[0].qVals == qValToShow)[0])

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, dataSet in enumerate(self.dataList):
            normF = self.resFitList[k][qValIdx][0][0]

            ax.errorbar(dataSet.energies, 
                        dataSet.intensities[qValIdx] / normF,
                        dataSet.errors[qValIdx] / normF, 
                        fmt='o')
            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
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

        for k, dataSet in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k], projection='3d')

            for i, qWiseData in enumerate(dataSet.intensities):
                normF = self.resFitList[k][i][0][0]

                ax.plot(dataSet.energies, 
                        qWiseData / normF,
                        dataSet.qVals[i], 
                        zdir='y', 
                        c=cmap(normColors(dataSet.qVals[i])))

            ax.set_xlabel(r'$\hslash \omega (\mu eV)$')
            ax.set_ylabel(r'$q$')
            ax.set_zlabel(r'$S_{300K}(q, \omega)$')
            ax.set_ylim((0, 2))
            ax.set_zlim((0, 1))
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], y=1.1)
            ax.grid()

        plt.tight_layout()
        self.canvas.draw()
    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(len(self.dataList), 2)

        for k, dataSet in enumerate(self.dataList):
            
            qList = dataSet.qVals 
            a0List   = []
            g0List   = []
            g1List   = []            
            betaList = []            

            for j, qWiseData in enumerate(dataSet.intensities):    
                a0List.append(self.protFitList[k][j].x[0])
                g0List.append(self.protFitList[k][j].x[1])
                g1List.append(self.protFitList[k][j].x[2])
                betaList.append(self.protFitList[k][j].x[3])

            
            #_Plot of the lorentzian width parameter of the fits
            ax1 = self.figure.add_subplot(mplGrid[:,0])
            ax1.plot(dataSet.qVals, a0List, marker='o', label='a0 - global diffusion contribution')
            ax1.plot(dataSet.qVals, betaList, marker='o', label='beta - protein contribution')
            ax1.set_ylabel(r'$Weight$', fontsize=14)
            ax1.set_xlabel(r'$q \ (\AA^{-1})$', fontsize=18)
            ax1.grid(True)
            ax1.legend(loc='upper center', framealpha=0.5)

            #_Plot of the lorentzian width parameter of the fits
            ax2 = self.figure.add_subplot(mplGrid[:,1])
            ax2.plot(dataSet.qVals**2, g0List, marker='o', label='g0 - global diffusion')
            ax2.plot(dataSet.qVals**2, g1List, marker='o', label='g1 - internal dynamics')
            ax2.set_ylabel(r'$Lorentzian \ width$', fontsize=14)
            ax2.set_xlabel(r'$q \ (\AA^{-2})$', fontsize=18)
            ax2.grid(True)
            ax2.legend(loc='upper center', framealpha=0.5)

        plt.tight_layout()
        self.canvas.draw()

    def fitPlot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  

        qValToShow = min(self.dataList[0].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(self.dataList[0].qVals == qValToShow)[0])

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, dataSet in enumerate(self.dataList):

            normF   = self.resFitList[k][qValIdx][0][0]
            bkgd    = self.resFitList[k][qValIdx][0][-1]

            ax.errorbar(dataSet.energies, 
                        dataSet.intensities[qValIdx] / normF,
                        dataSet.errors[qValIdx] / normF, 
                        fmt='o',
                        zorder=1)

            #_Plot resolution function
            ax.plot(dataSet.energies, 
                    self.resFunc(dataSet.energies, 1, *self.resFitList[k][qValIdx][0][1:-1], 0),
                    zorder=2,
                    ls=':',
                    label="Resolution function")

            #_Plot first lorentzian
            a0   = self.protFitList[k][qValIdx].x[0]
            g0   = self.protFitList[k][qValIdx].x[1]
            beta = self.protFitList[k][qValIdx].x[3]
            ax.plot(dataSet.energies, 
                    beta * a0 * g0**2 
                    / (g0**2 + dataSet.energies**2),
                    ls='--',
                    zorder=3,
                    label="Global diffusion")

            #_Plot second lorentzian
            a0 = self.protFitList[k][qValIdx].x[0]
            g1 = self.protFitList[k][qValIdx].x[2]
            beta = self.protFitList[k][qValIdx].x[3]
            ax.plot(dataSet.energies, 
                    beta * (1-a0) * (g0 + g1) / ( (g0 + g1)**2 + dataSet.energies**2 ),
                    ls=':',
                    zorder=4,
                    label="Internal dynamics")

            #_Plot D2O contribution
            temp    = dataSet.temp
            gD2O    = self.sD2O(temp, dataSet.qVals[qValIdx]) 
            ax.plot(dataSet.energies, 
                    self.D2OParams[k][qValIdx].x[1] * self.volFraction 
                    * gD2O / ( gD2O**2 + dataSet.energies**2 ),
                    ls='-.',
                    zorder=5,
                    label="D2O solvent")


            #_Compute and plot model function
            model = self.fitFunc(self.protFitList[k][qValIdx].x, dataSet, k, qValIdx, False)  
            ax.plot( dataSet.energies, model, zorder=6, label="Model" )


            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_ylim([1e-4, 1.2])
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            ax.legend(loc='upper left', framealpha=0.5)
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)

    message = QMessageBox.information(QWidget(), 'File selection',
            'Please select the file to be used for resolution function fitting...') 
    fileToOpen = QFileDialog().getOpenFileName()[0]
    resFuncObj = resFuncAnalysis.Window(fileToOpen)
    resFitList = resFuncObj.resFit()

    message = QMessageBox.information(QWidget(), 'File selection',
            'Please select the file containing D2O fitted parameters...') 
    paramsFile = QFileDialog().getOpenFileName()[0]
    with open(paramsFile, 'rb') as params:
        D2OParams = pk.Unpickler(params).load()
 

    subW = Window(arg[1:], resFitList, D2OParams, ECData, float(karg['ECCorr'])) 
    subW.show()

    sys.exit(app.exec_())
