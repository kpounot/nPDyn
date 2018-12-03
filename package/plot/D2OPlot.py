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

from ..fit.D2OFit import modelFunc, fitFunc
 
class D2OPlot(QWidget):
    """ This class created a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

        Plot        -> plot the normalized experimental data for the selected q-value
        3D Plot     -> plot the whole normalized dataSet
        Analysis    -> plot the different model parameters as a function of q-value
        Fit  -> plot the fitted model on top of the experimental data for the selected q-value """

    def __init__(self, parent):

        super().__init__()

        self.D2OData    = parent.D2OData
        self.D2OParams  = parent.D2OParams
        self.resFunc    = parent.resFunc
        self.resParams  = parent.resParams
        self.parent     = parent

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

        self.fitButton = QPushButton('Fit')
        self.fitButton.clicked.connect(self.fitPlot)

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
        layout.addWidget(self.fitButton)
        self.setLayout(layout)


    #--------------------------------------------------
    #_Definitions of the slots for the plot window
    #--------------------------------------------------
    def plot(self):
        """ This is used to plot the experimental data, without any fit. """
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qValToShow = min(self.D2OData.qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(self.D2OData.qVals == qValToShow)[0])

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        normF = self.resParams[qValIdx][0][0]

        ax.errorbar(self.D2OData.X, 
                    self.D2OData.intensities[qValIdx] / normF,
                    self.D2OData.errors[qValIdx] / normF, 
                    fmt='o')

        ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
        ax.set_yscale('log')
        ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot3D(self):
        """ 3D plot of the whole dataset. """

        plt.gcf().clear()     

        #_Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        ax = self.figure.add_subplot(111, projection='3d')

        for i, qWiseData in enumerate(self.D2OData.intensities):
            normF = self.resParams[i][0][0]

            ax.plot(self.D2OData.X, 
                    qWiseData / normF,
                    self.D2OData.qVals[i], 
                    zdir='y', 
                    c=cmap(normColors(self.D2OData.qVals[i])))

        ax.set_xlabel(r'$\hslash \omega (\mu eV)$')
        ax.set_ylabel(r'$q$')
        ax.set_zlabel(r'$S_{300K}(q, \omega)$')
        ax.set_ylim((0, 2))
        ax.set_zlim((0, 1))
        ax.grid()

        self.canvas.draw()
    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        #_Creates as many subplots as there are parameters in the model
        ax = self.figure.subplots(self.D2OParams[0].x.size, 1, sharex=True)

        qList = self.D2OData.qVals 
        
        #_Create 2D numpy array to easily access the q dependance of each parameter
        paramsList = np.column_stack( (params.x for params in self.D2OParams) )

        #_Plot the parameters of the fits
        for idx in range(self.D2OParams[0].x.size):
            ax[idx].plot(qList, paramsList[idx], marker='o') 
            ax[idx].set_ylabel(r'P %i' % idx, fontsize=14)
            ax[idx].grid(True)

        ax[-1].set_xlabel(r'$q \ (\AA^{-1})$', fontsize=14)

        plt.tight_layout()
        self.canvas.draw()

    def fitPlot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  

        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qValToShow = min(self.D2OData.qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(self.D2OData.qVals == qValToShow)[0])

        #_Get the normalization factor
        normF = self.resParams[qValIdx][0][0]

        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        ax.errorbar(self.D2OData.X, 
                    self.D2OData.intensities[qValIdx] / normF,
                    self.D2OData.errors[qValIdx] / normF, 
                    fmt='o',
                    zorder=1)

        ax.plot(self.D2OData.X, fitFunc(self.D2OParams[qValIdx].x,
                                             self.D2OData, 
                                             self.parent,
                                             qValIdx,
                                             returnCost=False), zorder=2)

        ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
        ax.set_yscale('log')
        ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()


    def initChecks(self):
        """ This methods is used to perform some checks before finishing class initialization. """

        if self.D2OData == None:
            raise Exception("No data for D2O were loaded.")

        if self.D2OParams == None:
            raise Exception("No parameters for D2O lineshape were found.\n" 
                            + "Please use a fitting method before plotting.\n")

