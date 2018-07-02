import sys, os, pickle as pk
import numpy as np
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
import h5py as h5

 
class Window(QDialog):
    def __init__(self, dataFiles=None):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.resFitList = []

        if dataFiles is not None:
            self.dataFiles = dataFiles

            #_Get datas from the file and store them into dataList 
            self.dataList = h5process.processData(self.dataFiles, karg['binS'], averageFiles=False, FWS=False)

            self.resFit()

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
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.resButton)
        self.setLayout(layout)


#_Everything needed for the fit
    def resFunc(self, x, normF, S, lorW, gauW, shift, bkgd):

        return  (normF * (S * lorW/(lorW**2 + (x-shift)**2) /np.pi 
                + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi))
                + bkgd))  
                
    def resFit(self):
    
        for i, resFile in enumerate(self.dataList):
            resList = []
            for j, qWiseData in enumerate(resFile.intensities):
                resList.append(optimize.curve_fit(self.resFunc, 
                                resFile.energies,
                                resFile.intensities[j],
                                sigma=[val+0.0001 for val in resFile.errors[j]],
                                #p0 = [0.5, 1, 0.8, 50, 0], 
                                bounds=([0., 0., 0., 0., -10, 0.],  
                                        [2000, 1, 5, 5, 10, 0.5]),
                                max_nfev=10000000,
                                method='trf'))
            self.resFitList.append(resList)

        return self.resFitList 

        
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

        mplGrid = gridspec.GridSpec(3, 1)

        for k, dataSet in enumerate(self.dataList):
            
            qList = dataSet.qVals 
            lauWList = []
            gauWList = []
            bkgdList = []            

            for j, qWiseData in enumerate(dataSet.intensities):    
                lauWList.append(self.resFitList[k][j][0][2])
                gauWList.append(self.resFitList[k][j][0][3])
                bkgdList.append(self.resFitList[k][j][0][5])

            
            #_Plot of the lorentzian width parameter of the fits
            ax1 = self.figure.add_subplot(mplGrid[0])
            ax1.plot(qList, lauWList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax1.set_ylabel(r'$Lorentzian \ Width $', fontsize=14)
            ax1.set_xlabel(r'$q$', fontsize=18)
            ax1.grid(True)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2)).get_frame().set_alpha(0.5)

            #_Plot of the gaussian width parameter of the fits
            ax2 = self.figure.add_subplot(mplGrid[1])
            ax2.errorbar(qList, gauWList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax2.set_ylabel(r'$Gaussian \ Width $', fontsize=14)
            ax2.set_xlabel(r'$q$', fontsize=18)
            ax2.grid(True)

            #_Plot of the background parameter of the fits
            ax3 = self.figure.add_subplot(mplGrid[2])
            ax3.errorbar(qList, bkgdList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax3.set_ylabel(r'$background $', fontsize=14)
            ax3.set_xlabel(r'$q$', fontsize=18)
            ax3.grid(True)


        plt.tight_layout()
        self.canvas.draw()

    def resPlot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  

        qValToShow = min(self.dataList[0].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(self.dataList[0].qVals == qValToShow)[0])

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, dataSet in enumerate(self.dataList):

            normF    = self.resFitList[k][qValIdx][0][0]
            S           = self.resFitList[k][qValIdx][0][1]
            lorW        = self.resFitList[k][qValIdx][0][2]
            gauW        = self.resFitList[k][qValIdx][0][3]
            shift       = self.resFitList[k][qValIdx][0][4]
            bkgd        = self.resFitList[k][qValIdx][0][5]

            ax.errorbar(dataSet.energies, 
                        dataSet.intensities[qValIdx] / normF,
                        dataSet.errors[qValIdx] / normF, 
                        fmt='o',
                        zorder=1)

            ax.plot(dataSet.energies, [self.resFunc(val, 1, S, lorW, gauW, shift, bkgd)
                                     for val in dataSet.energies], zorder=2)

            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            ax.legend(['...' + dataFile[dataFile.rfind('/'):] for dataFile in self.dataFiles], 
                   loc='upper left', framealpha=0.5)
            
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()

       
if __name__ == '__main__':

    app = QApplication(sys.argv)
    arg, karg = argParser.argParser(sys.argv)

    dataFiles = arg[1:]

    subW = Window(dataFiles) 
    subW.show()

    sys.exit(app.exec_())
