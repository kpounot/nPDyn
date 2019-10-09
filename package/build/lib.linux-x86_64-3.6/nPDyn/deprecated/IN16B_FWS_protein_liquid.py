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
import D2O_params_from_IN6 as D2O_IN6

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
        self.dataList = h5process.processData(dataFiles, karg['binS'], FWS=True)

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

        self.QENS3DButton = QPushButton('QENS 3D')
        self.QENS3DButton.clicked.connect(self.QENS_3D)

        self.fitButton = QPushButton('Fit')
        self.fitButton.clicked.connect(self.fitPlot)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.label = QLabel('Q value to plot', self)
        self.lineEdit = QLineEdit(self) 
        self.lineEdit.setText('0.8')

        self.scanNbrLabel = QLabel('Scan number to plot', self)
        self.scanLineEdit = QLineEdit(self) 
        self.scanLineEdit.setText('0')

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.scanNbrLabel)
        layout.addWidget(self.scanLineEdit)
        layout.addWidget(self.button)
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.QENS3DButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.fitButton)
        self.setLayout(layout)

#_Everything needed for the fit
    def modelFunc(self, x, D2OParams, qVal, temp, a0, g0, g1, beta):

        gD2O = self.sD2O(temp, qVal)

        return ( beta * (a0 * g0 / (g0**2 + x**2) 
                 + (1-a0) * (g0 + g1) / ( (g0 + g1)**2 + x**2 )) 
                 + self.volFraction * D2OParams[1] * gD2O / (gD2O**2 + x**2))



    def fitFunc(self, params, dataSet, fileIdx, qIdx, dEList, dEData, dEErrors, returnCost=True):

        resParams     = self.resFitList[fileIdx][qIdx][0]
        normF         = resParams[0]
        temp          = dataSet[0].temp
        D2OParams     = self.D2OParams[fileIdx][qIdx].x

        cost = 0

        model   = self.modelFunc(dEList, D2OParams, dataSet[0].qVals[qIdx], temp, *params)
        resFunc = self.resFunc(dEList, 1, *resParams[1:-1], 0)

        model = np.convolve(model, resFunc, mode='same') + resParams[-1] 

        cost += np.sum( (dEData / normF - model)**2  
                        / (dEErrors / normF)**2 )  


        if returnCost:
            return cost
        else:
            return model

                

    def basinHopping_fit(self):
    
        for i, dataSet in enumerate(self.dataList):

            print("\nFitting data for file: " + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(50*"_", flush=True)

            bounds  = [(0., 1), (0.9, 100), (0.9, 100), (0., 1)] 
            params_init = [0.2, 5, 30, 0.04]

            fileFitList = []
            for qIdx, qWiseData in enumerate(dataSet[0].intensities):

                dEList   = np.array([dEData.deltaE for dEData in dataSet]).ravel()
                dEData   = np.row_stack((dEData.intensities[qIdx] for dEData in dataSet))
                dEErrors = np.row_stack((dEData.errors[qIdx] for dEData in dataSet))
                
                fitList = []
                for scanIdx, scanValues in enumerate(qWiseData): 

                    fitList.append(optimize.basinhopping(self.fitFunc, 
                                    params_init,
                                    niter = self.BH_iter,
                                    niter_success = 0.20*self.BH_iter,
                                    disp=True,
                                    minimizer_kwargs={ 'args':(dataSet, i, qIdx, dEList,
                                                               dEData[:,scanIdx], dEErrors[:,scanIdx], True),
                                                       'bounds':bounds }))

                    r  = "\nFinal result for q-value %.2f and scan %i:\n" % (dataSet[0].qVals[qIdx],
                                                                             scanIdx)
                    r += "    a0   = %.2f\n" % fitList[scanIdx].x[0]
                    r += "    g0   = %.2f\n" % fitList[scanIdx].x[1]
                    r += "    g1   = %.2f\n" % fitList[scanIdx].x[2]
                    r += "    beta = %.2f\n" % fitList[scanIdx].x[3]

                    print(r, flush=True)

                fileFitList.append(fitList)
            self.protFitList.append(fileFitList)

        with open(self.paramsFile, 'wb') as fittedParamsFile:
            myFile = pk.Pickler(fittedParamsFile)
            modelFitList = self.protFitList
            myFile.dump(modelFitList)

        return self.protFitList 


#_Definitions of the slots for the plot window
    def plot(self):
	   
        plt.gcf().clear()     
        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))
        
        qValToShow = min(self.dataList[0][0].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(self.dataList[0][0].qVals == qValToShow)[0])

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, dataSet in enumerate(self.dataList):
            normF = self.resFitList[k][qValIdx][0][0]

            ax = self.figure.add_subplot(mplGrid[:,k])

            for dEIdx, dEData in enumerate(dataSet):
                ax.errorbar(np.arange(dEData.intensities.shape[1]), 
                            dEData.intensities[qValIdx] / normF,
                            dEData.errors[qValIdx] / normF, 
                            label='$\\Delta E$ = %.2f $\mu eV$' % dEData.deltaE,
                            fmt='o')

            ax.set_xlabel(r'$Scan \ number$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):]) 
            ax.legend(loc='upper left', framealpha=0.5)
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot3D(self):

        plt.gcf().clear()     

        cmapList =  [ matplotlib.cm.get_cmap('winter'),
                      matplotlib.cm.get_cmap('spring'),
                      matplotlib.cm.get_cmap('summer'),
                      matplotlib.cm.get_cmap('autumn'),
                      matplotlib.cm.get_cmap('cool'),
                      matplotlib.cm.get_cmap('Wistia'),
                    ]


        for k, dataSet in enumerate(self.dataList):
            normColors = matplotlib.colors.Normalize(vmin=0, vmax=len(dataSet[0].intensities[0]))

            #_Generate a column vector containing normalization factor for each q-value
            normFList = np.transpose([[val[0][0] for val in self.resFitList[k][1:]]])

            for dEIdx, dEData in enumerate(dataSet):
                ax = self.figure.add_subplot( len(self.dataList), 
                                              len(dataSet), 
                                              k * len(dataSet) + dEIdx + 1, projection='3d' )

                xx, yy = np.meshgrid(dEData.qVals[1:], np.arange(dEData.intensities.shape[1]))

                ax.plot_wireframe( xx, 
                                   yy,
                                   np.transpose(dEData.intensities[1:] / normFList),
                                   label = '$\\Delta E$ = %.2f $\mu eV$' % dEData.deltaE, 
                                   colors=cmapList[dEIdx](normColors(np.arange(dEData.intensities.shape[1]))) )

                ax.set_xlabel(r'$q\ (\AA^{-1})$')
                ax.set_ylabel(r'$Scan \ number$')
                ax.set_zlabel(r'$S(q, \Delta E)$')
                ax.legend(loc='upper right', framealpha=0.5)
                ax.grid()

        plt.title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], y=1.1)
        plt.tight_layout()
        self.canvas.draw()

    #_3D plot of data as QENS spectra reconstructed from energies offsets as a function of scan number
    def QENS_3D(self):

        plt.gcf().clear()     


        qValToShow = min(self.dataList[0][0].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(self.dataList[0][0].qVals == qValToShow)[0])

        for k, dataSet in enumerate(self.dataList):
                normColors = matplotlib.colors.Normalize(vmin=0, vmax=len(dataSet[0].intensities))
                cmap = matplotlib.cm.get_cmap('winter')

                ax = self.figure.add_subplot( 1, len(self.dataList), k + 1, projection='3d' )

                dEList = np.array([dEData.deltaE for dEData in dataSet]).ravel()
                xx, yy = np.meshgrid(dEList, np.arange(dataSet[0].intensities.shape[1]))

                dEData   = np.row_stack((dEData.intensities[qValIdx] for dEData in dataSet))
                dEErrors = np.row_stack((dEData.errors[qValIdx] for dEData in dataSet))

                normF = self.resFitList[k][qValIdx][0][0]

                ax.scatter( xx, 
                            yy,
                            np.transpose(dEData / normF),
                            label = '...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], 
                            c='blue' )

                for scanNbr, scanValue in enumerate(dataSet[0].intensities[0]):
                    #_Compute and plot model function
                    model = self.fitFunc(self.protFitList[k][qValIdx][scanNbr].x, dataSet, k, qValIdx, 
                            dEList, dEData[:,scanNbr], dEErrors[:,scanNbr], False)  
                    ax.plot(dEList, np.full(dEList.shape, scanNbr), model,
                            c=cmap(normColors(scanNbr)) )

                ax.set_xlabel(r'$\Delta E \ (\mu eV)$')
                ax.set_ylabel(r'$Scan \ number$')
                ax.set_zlabel(r'$S(q, \Delta E)$')
                ax.legend(loc='upper right', framealpha=0.5)
                ax.grid()

        plt.title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], y=1.1)
        plt.tight_layout()

        self.canvas.draw()

    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(len(self.dataList), 2)

        for k, dataSet in enumerate(self.dataList):

            qValToShow = min(dataSet[0].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
            qValIdx = int(np.argwhere(dataSet[0].qVals == qValToShow)[0])
            
            scanList = np.arange(len(self.protFitList[k][qValIdx]))
            a0List   = [val.x[0] for val in self.protFitList[k][qValIdx]]
            g0List   = [val.x[1] for val in self.protFitList[k][qValIdx]]
            g1List   = [val.x[2] for val in self.protFitList[k][qValIdx]]
            betaList = [val.x[3] for val in self.protFitList[k][qValIdx]]

            
            #_Plot of the lorentzian width parameter of the fits
            ax1 = self.figure.add_subplot(mplGrid[:,0])
            ax1.plot(scanList, a0List, marker='o', label='a0 - global diffusion contribution')
            ax1.plot(scanList, betaList, marker='o', label='beta - protein contribution')
            ax1.set_ylabel(r'$Weight$', fontsize=14)
            ax1.set_xlabel(r'$Scan \ number$', fontsize=18)
            ax1.grid(True)
            ax1.legend(loc='upper center', framealpha=0.5)

            #_Plot of the lorentzian width parameter of the fits
            ax2 = self.figure.add_subplot(mplGrid[:,1])
            ax2.plot(scanList, g0List, marker='o', label='g0 - global diffusion')
            ax2.plot(scanList, g1List, marker='o', label='g1 - internal dynamics')
            ax2.set_ylabel(r'$Lorentzian \ width$', fontsize=14)
            ax2.set_xlabel(r'$Scan \ number$', fontsize=18)
            ax2.grid(True)
            ax2.legend(loc='upper center', framealpha=0.5)

        plt.tight_layout()
        self.canvas.draw()

    def fitPlot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  


        scanNbr = int(self.scanLineEdit.text())

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, dataSet in enumerate(self.dataList):

            qValToShow = min(dataSet[0].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
            qValIdx = int(np.argwhere(dataSet[0].qVals == qValToShow)[0])

            normF   = self.resFitList[k][qValIdx][0][0]
            bkgd    = self.resFitList[k][qValIdx][0][-1]
    
            dEList   = np.array([dEData.deltaE for dEData in dataSet]).ravel()
            dEData   = np.row_stack((dEData.intensities[qValIdx] for dEData in dataSet))
            dEErrors = np.row_stack((dEData.errors[qValIdx] for dEData in dataSet))

            #_Plot processed data
            ax.errorbar(dEList, 
                        dEData[:,scanNbr] / normF,
                        dEErrors[:,scanNbr] / normF, 
                        fmt='o',
                        zorder=1)

            #_Plot resolution function
            ax.plot(dEList, 
                    self.resFunc(dEList, 1, *self.resFitList[k][qValIdx][0][1:]),
                    zorder=2,
                    ls=':',
                    label="Resolution function")

            #_Plot first lorentzian
            a0   = self.protFitList[k][qValIdx][scanNbr].x[0]
            g0   = self.protFitList[k][qValIdx][scanNbr].x[1]
            beta = self.protFitList[k][qValIdx][scanNbr].x[3]
            ax.plot(dEList, 
                    beta * a0 * g0**2 
                    / (g0**2 + dEList**2),
                    ls='--',
                    zorder=3,
                    label="Global diffusion")

            #_Plot second lorentzian
            a0 = self.protFitList[k][qValIdx][scanNbr].x[0]
            g1 = self.protFitList[k][qValIdx][scanNbr].x[2]
            beta = self.protFitList[k][qValIdx][scanNbr].x[3]
            ax.plot(dEList, 
                    beta * (1-a0) * (g0 + g1) / ( (g0 + g1)**2 + dEList**2 ),
                    ls=':',
                    zorder=4,
                    label="Internal dynamics")

            #_Plot D2O contribution
            temp    = dataSet[0].temp
            gD2O    = self.sD2O(temp, qValIdx) 
            ax.plot(dEList, 
                    self.D2OParams[k][qValIdx].x[1] * self.volFraction 
                    * gD2O / ( gD2O**2 + dataSet.energies**2 ),
                    ls='-.',
                    zorder=5,
                    label="D2O solvent")


            #_Compute and plot model function
            model = self.fitFunc(self.protFitList[k][qValIdx][scanNbr].x, dataSet, k, qValIdx, 
                    dEList, dEData[:,scanNbr], dEErrors[:,scanNbr], False)  
            ax.plot( dEList, model, zorder=6, label="Model" )


            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            ax.legend(loc='upper right', framealpha=0.5)
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

    subW = Window(arg[1:], resFitList, D2OParams) 
    subW.show()

    sys.exit(app.exec_())
