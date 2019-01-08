import sys, os
import numpy as np

from collections import namedtuple

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib

from .subPlotsFormat import subplotsFormat

class plotMSDSeries(QWidget):
    """ This class created a PyQt widget containing a matplotlib canvas to draw the plots.

        Init:   parent      -> parent class instance, the nPDyn dataset instance
                msdSeries   -> MSD obtained from MD simulations
                tempList    -> temperature list for MSD from simulation
                fileIdxList -> indices of experimental data's MSD to be compared with msdSeries """


    def __init__(self, parent, msdIdxList, tempList, fileIdxList):

        super().__init__()

        self.parent         = parent
        self.tempList       = tempList
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

        self.msdSeries = [parent.msdSeriesList[idx] for idx in msdIdxList] 

        #--------------------------------------------------
        #_Construction of the GUI
        #--------------------------------------------------

        #_A figure instance to plot on
        self.figure = plt.figure()

        #_This is the Canvas Widget that displays the `figure`
        #_it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        #_Add some interactive elements
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.label = QLabel('Max temperature for plotting', self)
        self.lineEdit = QLineEdit(self) 
        self.lineEdit.setText('300')

        self.msdButton = QPushButton('Replot')
        self.msdButton.clicked.connect(self.MSD)

        #_Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.msdButton)
        self.setLayout(layout)


        #_Calling method to plot MSD
        self.MSD()


    #--------------------------------------------------
    #_Definitions of the slots for the plot window
    #--------------------------------------------------

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
            #_Plotting the experimental MSD
            ax.errorbar(self.dataSetList[i].X[:tempIdx+1], 
                        msdList[:tempIdx+1],
                        errList[:tempIdx+1], 
                        label = self.dataFiles[i])

        for i in range(len(self.msdSeries)):
            
            #_Plotting the MSD from simulation
            ax.errorbar(self.tempList, self.msdSeries[i][:,0], self.msdSeries[i][:,1], label="Simulated MSD")

            ax.set_xlabel(r'$Temperature (K)$')
            ax.set_ylabel(r'$MSD \ (\AA^{2})$')
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

