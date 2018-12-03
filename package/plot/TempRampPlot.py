import sys, os
import numpy as np

from collections import namedtuple

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib

from .subPlotsFormat import subplotsFormat

class TempRampPlot(QWidget):
    """ This class created a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

        Total scattering    -> plot the total scattering (sum for all q-values)
        q-Wise scattering   -> plot the scattering for each q-value
        Fit                 -> plot the fitted model on data for all temperatures
        MSD                 -> plot the fitted MSD as a function of temperature """

    def __init__(self, parent, fileIdxList):

        super().__init__()

        self.parent         = parent
        self.fileIdxList    = fileIdxList

        try:
            self.initChecks()
        except Exception as e:
            print(e)
            return


        #_Dataset related attributes
        self.dataSetList    = [parent.dataSetList[i] for i in fileIdxList]
        self.dataFiles      = [parent.dataFiles[i] for i in fileIdxList]
        self.paramsList     = [parent.paramsList[i] for i in fileIdxList]
        self.paramsNames    = [parent.paramsNames[i] for i in fileIdxList]
        self.modelList      = [parent.modelList[i] for i in fileIdxList]

        #--------------------------------------------------
        #_Construction of the GUI
        #--------------------------------------------------

        #_A figure instance to plot on
        self.figure = plt.figure()

        #_This is the Canvas Widget that displays the `figure`
        #_it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        #_Add some interactive elements
        self.totalScatButton = QPushButton('Total scattering')
        self.totalScatButton.clicked.connect(self.totalScat)

        self.qWiseScatButton = QPushButton('q-Wise scattering')
        self.qWiseScatButton.clicked.connect(self.qWiseScat)

        self.fitButton = QPushButton('Fit')
        self.fitButton.clicked.connect(self.fit)

        self.msdButton = QPushButton('MSD')
        self.msdButton.clicked.connect(self.MSD)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.label = QLabel('Temperature to plot (for fit, max temp for MSD)', self)
        self.lineEdit = QLineEdit(self) 
        self.lineEdit.setText('300')

        #_Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.totalScatButton)
        layout.addWidget(self.qWiseScatButton)
        layout.addWidget(self.fitButton)
        layout.addWidget(self.msdButton)
        self.setLayout(layout)


    #--------------------------------------------------
    #_Definitions of the slots for the plot window
    #--------------------------------------------------
    def totalScat(self):
        """ This is used to plot the experimental data, without any fit. """

        plt.gcf().clear()       
        ax = self.figure.add_subplot(111)  

        for i in range(len(self.fileIdxList)):
            ax.plot(self.dataSetList[i].X, 
                    np.sum(self.dataSetList[i].intensities, axis=0),
                    label=self.dataFiles[i])

            ax.set_xlabel(r'$Temperature (K)$')
            ax.set_ylabel(r'$Scattering$')
            ax.legend(framealpha=0.5, fontsize=12)
            ax.grid()

        self.canvas.draw()
    

    def qWiseScat(self):
        """ For each file, plots the temperature of elastic scattering intensity for each q-value. """

        plt.gcf().clear()     
        ax = subplotsFormat(self, True, True) 

        #_Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('rainbow')

        for i, subplot in enumerate(ax):
            for qIdx in self.dataSetList[i].qIdx:
                subplot.plot(self.dataSetList[i].X, 
                             self.dataSetList[i].intensities[qIdx],
                             label=self.dataSetList[i].qVals[qIdx],
                             c=cmap(normColors(self.dataSetList[i].qVals[qIdx])))

            subplot.set_title(self.dataFiles[i], fontsize=10)
            subplot.set_xlabel(r'$Temperature (K)$')
            subplot.set_ylabel(r'$Scattering$')
            subplot.legend(framealpha=0.5, fontsize=12)
            subplot.grid()

        self.canvas.draw()
    

    #_Plot of the parameters resulting from the fit procedure
    def fit(self):

        plt.gcf().clear()     
        ax = subplotsFormat(self, True, True)

        for i, subplot in enumerate(ax):
            #_Obtaining the temperature to plot as being the closest one to the number entered by the user 
            tempToShow = min(self.dataSetList[i].X, key = lambda x : abs(float(self.lineEdit.text()) - x))
            tempIdx = int(np.argwhere(self.dataSetList[i].X == tempToShow)[0])


            #_Plotting experimental data
            subplot.errorbar(   self.dataSetList[i].qVals**2, 
                                self.dataSetList[i].intensities[:,tempIdx],
                                self.dataSetList[i].errors[:,tempIdx],
                                label="Experimental"   )
    
            #_Adding a transparent blue region to easily locate the fitted data points
            #_(provided that no detectors were discarded between fitting and plotting)
            qMin = self.dataSetList[i].qVals[self.dataSetList[i].qIdx[0]]
            qMax = self.dataSetList[i].qVals[self.dataSetList[i].qIdx[-1]]
            subplot.axvspan(qMin**2, qMax**2, color='c', alpha=0.4)

            #_Plotting the model and adding the parameters values to the label
            label = "Model\n"
            if len(self.paramsList[i]) == len(self.paramsNames[i]): #_Formatting the legend output
                paramVals = np.round(self.paramsList[i][tempIdx][0], 2).astype(str)
                for pIdx, val in enumerate(paramVals): 
                    label += self.paramsNames[i][pIdx] + ": " + val + "\n"

            subplot.plot(   self.dataSetList[i].qVals**2,
                            self.modelList[i](self.dataSetList[i].qVals, *self.paramsList[i][tempIdx][0]),
                            label=label)

            subplot.set_title(self.dataFiles[i], fontsize=10)
            subplot.set_xlabel(r'$Scattering \ vector \ q \ (\AA^{-2})$')
            subplot.set_ylabel(r'EISF at %d K' % tempToShow)
            subplot.legend(framealpha=0.5, fontsize=12)
            subplot.grid()

        self.canvas.draw()
 

    def MSD(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  

        #_Plot the mean-squared displacement as a function of temperature for each file
        for i in range(len(self.fileIdxList)):
            #_Obtaining the temperature to plot as being the closest one to the number entered by the user 
            tempToShow = min(self.dataSetList[i].X, key = lambda x : abs(float(self.lineEdit.text()) - x))
            tempIdx = int(np.argwhere(self.dataSetList[i].X == tempToShow)[0])

            #_Extracting the MSD from parameters for each temperature
            msdList = [self.paramsList[i][tempIdx][0][1] for tempIdx, temp in enumerate(self.dataSetList[i].X)]
            qMin = self.dataSetList[i].qVals[self.dataSetList[i].qIdx[0]]
            qMax = self.dataSetList[i].qVals[self.dataSetList[i].qIdx[-1]]

            #_Computing the errors for each temperature from covariant matrix
            errList = [np.sqrt(np.diag(self.paramsList[i][tempIdx][1]))[1] for tempIdx, temp 
                                                                in enumerate(self.dataSetList[i].X)]

            #_Plotting the MSD
            ax.errorbar(self.dataSetList[i].X[:tempIdx], 
                        msdList[:tempIdx],
                        errList[:tempIdx], 
                        label = self.dataFiles[i])

            ax.set_xlabel(r'$Temperature (K)$')
            ax.set_ylabel(r'$MSD \ (\AA)$ q=%.2f to %.2f ($\AA^{-1})$' % ( qMin, qMax ))
            ax.legend(framealpha=0.5, fontsize=12, loc='upper left')
            ax.grid()

        self.canvas.draw()


    def initChecks(self):
        """ This methods is used to perform some checks before finishing class initialization. """


        for i in self.fileIdxList:
            if not self.parent.dataSetList[i]:
                raise Exception("ERROR: Index error, no data were found in dataSetList for index %i.\n" % i)

            if not self.parent.paramsList[i]:
                print("WARNING: no fitted parameters were found for data at index %i.\n" % i    
                      + "Some plotting methods might not work properly.\n")

