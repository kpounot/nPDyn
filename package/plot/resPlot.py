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

 
class ResPlot(QWidget):
    """ This class created a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

        Plot        -> plot the normalized experimental data for the selected q-value
        3D Plot     -> plot the whole normalized dataSet
        Analysis    -> plot the different model parameters as a function of q-value
        Resolution  -> plot the fitted model on top of the experimental data for the selected q-value """

    def __init__(self, parent):

        super().__init__()

        self.resFiles       = parent.resFiles
        self.resData        = parent.resData
        self.resFunc        = parent.resFunc
        self.resParams      = parent.resParams
        self.paramsNames    = parent.resPNames

        try:
            self.initChecks()
        except Exception as e:
            print(e)
            return

        #--------------------------------------------------
        #_Construction of the GUI
        #--------------------------------------------------

        #_A figure instance to plot on
        self.figure = plt.figure()

        #_This is the Canvas Widget that displays the `figure`
        #_it takes the `figure` instance as a parameter to __init__
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

        #_Set the layout
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


    #--------------------------------------------------
    #_Definitions of the slots for the plot window
    #--------------------------------------------------
    def plot(self):
        """ This is used to plot the experimental data, without any fit. """
	   
        plt.gcf().clear()     
        ax = subplotsFormat(self, False, True, res=True)  
        
        for idx, subplot in enumerate(ax):
            #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
            qValToShow = min(self.resData[idx].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
            qValIdx = int(np.argwhere(self.resData[idx].qVals == qValToShow)[0])

            #_Plot the datas for selected q value normalized with integrated curves at low temperature
            normF = self.resParams[idx][qValIdx][0][0]

            subplot.errorbar(self.resData[idx].X, 
                        self.resData[idx].intensities[qValIdx] / normF,
                        self.resData[idx].errors[qValIdx] / normF, 
                        fmt='o')

            subplot.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            subplot.set_title(self.resFiles[idx], fontsize=10)   
            subplot.grid()
            
        self.canvas.draw()

    def plot3D(self):
        """ 3D plot of the whole dataset. """

        plt.gcf().clear()     

        #_Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        ax = subplotsFormat(self, projection='3d', res=True)

        for idx, subplot in enumerate(ax):
            for i, qWiseData in enumerate(self.resData[idx].intensities):
                normF = self.resParams[idx][i][0][0]

                subplot.plot(self.resData[idx].X, 
                        qWiseData / normF,
                        self.resData[idx].qVals[i], 
                        zdir='y', 
                        c=cmap(normColors(self.resData[idx].qVals[i])))

            subplot.set_xlabel(r'$\hslash \omega \ (\mu eV)$')
            subplot.set_ylabel(r'$q$')
            subplot.set_zlabel(r'$S \ (q, \omega)$')
            subplot.set_title(self.resFiles[idx], fontsize=10)   
            subplot.set_ylim((0, 2))
            subplot.set_zlim((0, 1))
            subplot.grid()

        self.canvas.draw()
    

    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        #_Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, resParams=True)

        #_Plot the parameters of the fits
        for fileIdx, fileName in enumerate(self.resFiles):
            #_Create 2D numpy array to easily access the q dependance of each parameter
            paramsList = np.column_stack( (params[0] for params in self.resParams[fileIdx]) )

            qList = self.resData[fileIdx].qVals 

            for idx, subplot in enumerate(ax):
                subplot.plot(qList, paramsList[idx], marker='o', label=fileName) 
                subplot.set_ylabel(self.paramsNames[fileIdx][idx])
                subplot.set_xlabel(r'$q \ (\AA^{-1})$')
                subplot.grid(True)

        plt.legend(framealpha=0.5, fontsize=10, bbox_to_anchor=(0.3, 2.5))

        self.canvas.draw()


    def resPlot(self):
	   
        plt.gcf().clear()     
        ax = subplotsFormat(self, False, True, res=True) 

        for idx, subplot in enumerate(ax):
            #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
            qValToShow = min(self.resData[idx].qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
            qValIdx = int(np.argwhere(self.resData[idx].qVals == qValToShow)[0])

            #_Get the normalization factor
            normF = self.resParams[idx][qValIdx][0][0]

            #_Plot the datas for selected q value normalized with integrated curves at low temperature
            subplot.errorbar(self.resData[idx].X, 
                        self.resData[idx].intensities[qValIdx] / normF,
                        self.resData[idx].errors[qValIdx] / normF, 
                        fmt='o',
                        zorder=1)

            #_Plot the model
            subplot.plot(self.resData[idx].X, self.resFunc[idx](self.resData[idx].X, 
                                                      *self.resParams[idx][qValIdx][0]) / normF, 
                                                      zorder=2)

            subplot.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=16)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=16)   
            subplot.set_title(self.resFiles[idx], fontsize=10)   
            
            subplot.grid()

        self.canvas.draw()


    def initChecks(self):
        """ This methods is used to perform some checks before finishing class initialization. """

        if self.resData == None:
            raise Exception("No data for resolution function were loaded.")

        if self.resParams == None:
            raise Exception("No parameters for resolution function were found.\n" 
                            + "Please use a fitting method before plotting.\n")

        if self.paramsNames == None:
            self.paramsNames = ["P%i" % i for i, val in enumerate(self.resParams[0][0])]

        for idx in range(len(self.resFiles)):
            if len(self.paramsNames[idx]) != self.resParams[idx][0][0].size:
                raise Exception("Length of paramsNames different to resParams")


