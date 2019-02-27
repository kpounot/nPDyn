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
from scipy.signal import fftconvolve
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib

import h5process

import IN16B_QENS_resFunc as resFuncAnalysis
import D2O_params_from_IN6 as D2OParams
import IN16B_QENS_fittedEC_substraction as fitECCorr

class Window(QDialog):
    def __init__(self, dataFiles, resFitList, ECCurves=None):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        if type(dataFiles) == str:
            self.dataFiles  = [dataFiles]
        else:
            self.dataFiles = dataFiles

        #_Obtaining data for resolution function
        message = QMessageBox.information(QWidget(), 'File selection',
                'Please select the file to be used for resolution function fitting...') 
        fileToOpen = QFileDialog().getOpenFileName()[0]
        resFuncObj = resFuncAnalysis.Window(fileToOpen)


        self.resFitList = resFitList
        self.resFunc    = resFuncAnalysis.Window().resFunc
        self.sD2O        = D2OParams.getD2Odata(1)
        self.D2OFitList = []

        #_Get datas from the file and store them into dataList 
        self.dataList = h5process.processData(dataFiles, karg['binS'], FWS=False)

        #_Discarding data outside of user-defined q-range
        qMin = float(karg['qMin'])
        qMax = float(karg['qMax'])
        for fileIdx, dataSet in enumerate(self.dataList):
            qIdxRange = [qIdx for qIdx, qVal in enumerate(dataSet.qVals) 
                                    if qMin < qVal and qVal < qMax]
            self.dataList[fileIdx] = dataSet._replace( 
                        qVals = dataSet.qVals[qIdxRange[0]:qIdxRange[-1] + 1],
                        intensities = dataSet.intensities[qIdxRange[0]:qIdxRange[-1] + 1],
                        errors = dataSet.errors[qIdxRange[0]:qIdxRange[-1] + 1] )


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
                self.D2OFitList = pk.Unpickler(params).load()
 


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
    def modelFunc(self, x, qVal, temp, a1):

        gD2O = self.sD2O(temp, qVal)

        return a1 * gD2O / (gD2O**2 + x**2)


    def fitFunc(self, params, dataSet, fileIdx, qIdx, returnCost=True):

        resParams     = self.resFitList[fileIdx][qIdx][0]
        normF         = resParams[0]
        temp          = dataSet.temp

        cost = 0

        model   = self.modelFunc(dataSet.energies, dataSet.qVals[qIdx], temp, *params[1:])
        resFunc = self.resFunc(dataSet.energies, 1, *resParams[1:-1], 0)

        model = params[0] * resFunc + np.convolve(model, resFunc, mode='same') + resParams[-1]

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


            bounds      = [(0., 1), (0.,1.)] 
            params_init = [0.2, 0.5]

            fitList = []
            for qIdx, qWiseData in enumerate(dataSet.intensities):

                fitList.append(optimize.basinhopping(self.fitFunc, 
                                params_init,
                                niter = self.BH_iter,
                                niter_success = 0.25*self.BH_iter,
                                disp=True,
                                minimizer_kwargs={  'args': (dataSet, i, qIdx, True),
                                                    'bounds': bounds    }))

                r  = "\nFinal result for q-value %.2f:\n" % dataSet.qVals[qIdx]
                r += "    a1   = %.2f\n" % fitList[qIdx].x[0]
                r += "    a2 = %.2f\n" % fitList[qIdx].x[1]

                print(r, flush=True)

            self.D2OFitList.append(fitList)

        with open(self.paramsFile, 'wb') as fittedParamsFile:
            myFile = pk.Pickler(fittedParamsFile)
            modelFitList = self.D2OFitList
            myFile.dump(modelFitList)

        return self.D2OFitList 

        
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
            a0List = []
            a1List = []            

            for j, qWiseData in enumerate(dataSet.intensities):    
                a0List.append(self.D2OFitList[k][j].x[0])
                a1List.append(self.D2OFitList[k][j].x[1])

            
            #_Plot of the lorentzian width parameter of the fits
            ax1 = self.figure.add_subplot(mplGrid[:,0])
            ax1.plot(dataSet.qVals, a0List, marker='o', label='a0 - sample cell')
            ax1.plot(dataSet.qVals, a1List, marker='o', label='a1 - D2O')
            ax1.set_ylabel(r'$Weight$', fontsize=14)
            ax1.set_xlabel(r'$q \ (\AA^{-1})$', fontsize=18)
            ax1.grid(True)
            ax1.legend(loc='upper center', framealpha=0.5)

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


            #_Compute and plot model function
            model = self.fitFunc(self.D2OFitList[k][qValIdx].x, dataSet, k, qValIdx, False)  
            ax.plot( dataSet.energies, model, zorder=6, label="Model" )


            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
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

    try:
        EC_scaleFactor = float(karg['ECCorr'])

        message = QMessageBox.information(QWidget(), 'File selection',
                'Please select the file containing the QENS of empty cell...') 
        fileToOpen = QFileDialog().getOpenFileName()[0]
        ECCorr = fitECCorr.ECCorr(fileToOpen)
        ECCurve = ECCorr.getECCurves(EC_scaleFactor)

    except KeyError:
        print('No value for "ECCorr" parameter was set, no empty cell correction was performed.\n',
                flush=True)
 

    resFitList = resFuncObj.resFit()

    subW = Window(arg[1:]) 
    subW.show()

    sys.exit(app.exec_())
