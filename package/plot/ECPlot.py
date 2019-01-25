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

 
class ECPlot(QWidget):
    """ This class created a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

        Plot        -> plot the normalized experimental data for the selected q-value
        3D Plot     -> plot the whole normalized dataSet
        Analysis    -> plot the different model parameters as a function of q-value
        Resolution  -> plot the fitted model on top of the experimental data for the selected q-value """

    def __init__(self, dataset):

        super().__init__()

        self.dataset   = dataset

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
        self.fitButton.clicked.connect(self.resPlot)

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
        ax = subplotsFormat(self, False, True)  
        
        for idx, subplot in enumerate(ax):
            #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
            qValToShow = min(self.dataset[idx].data.qVals, 
                                                        key = lambda x : abs(float(self.lineEdit.text()) - x))
            qValIdx = int(np.argwhere(self.dataset[idx].data.qVals == qValToShow)[0])

            #_Plot the datas for selected q value normalized with integrated curves at low temperature
            normF = self.dataset[idx].params[qValIdx][0][0]

            subplot.errorbar(self.dataset[idx].data.X, 
                        self.dataset[idx].data.intensities[qValIdx] / normF,
                        self.dataset[idx].data.errors[qValIdx] / normF, 
                        fmt='o')

            subplot.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            subplot.set_title(self.dataset[idx].fileName, fontsize=10)   
            subplot.grid()
            
        self.canvas.draw()




    def plot3D(self):
        """ 3D plot of the whole dataset. """

        plt.gcf().clear()     

        #_Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        ax = subplotsFormat(self, projection='3d')

        for idx, subplot in enumerate(ax):
            for i, qWiseData in enumerate(self.dataset[idx].data.intensities):
                normF = self.dataset[idx].params[i][0][0]

                subplot.plot(self.dataset[idx].data.X, 
                        qWiseData / normF,
                        self.dataset[idx].data.qVals[i], 
                        zdir='y', 
                        c=cmap(normColors(self.dataset[idx].data.qVals[i])))

            subplot.set_xlabel(r'$\hslash \omega \ (\mu eV)$')
            subplot.set_ylabel(r'$q$')
            subplot.set_zlabel(r'$S \ (q, \omega)$')
            subplot.set_title(self.dataset[idx].fileName, fontsize=10)   
            subplot.set_ylim((0, 2))
            subplot.set_zlim((0, 1))
            subplot.grid()

        self.canvas.draw()
    



    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        #_Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, params=True)

        #_Plot the parameters of the fits
        for fileIdx, dataset in enumerate(self.dataset):
            #_Create 2D numpy array to easily access the q dependance of each parameter
            paramsList = np.column_stack( (params[0] for params in dataset.params) )

            qList = dataset.data.qVals 

            for idx, subplot in enumerate(ax):
                subplot.plot(qList, paramsList[idx], marker='o', label=dataset.fileName) 
                subplot.set_ylabel(dataset.paramsNames[idx])
                subplot.set_xlabel(r'$q \ (\AA^{-1})$')
                subplot.grid(True)

        plt.legend(framealpha=0.5, fontsize=10, bbox_to_anchor=(0.3, 2.5))

        self.canvas.draw()




    def resPlot(self):
	   
        plt.gcf().clear()     
        ax = subplotsFormat(self, False, True) 

        for idx, subplot in enumerate(ax):
            #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
            qValToShow = min(self.dataset[idx].data.qVals, 
                                                    key = lambda x : abs(float(self.lineEdit.text()) - x))
            qValIdx = int(np.argwhere(self.dataset[idx].data.qVals == qValToShow)[0])

            #_Get the normalization factor
            normF = self.dataset[idx].params[qValIdx][0][0]

            #_Plot the datas for selected q value normalized with integrated curves at low temperature
            subplot.errorbar(self.dataset[idx].data.X, 
                        self.dataset[idx].data.intensities[qValIdx] / normF,
                        self.dataset[idx].data.errors[qValIdx] / normF, 
                        fmt='o',
                        zorder=1)

            #_Plot the model
            subplot.plot(self.dataset[idx].data.X, self.dataset[idx].model(self.dataset[idx].data.X, 
                                                      *self.dataset[idx].params[qValIdx][0]) / normF, 
                                                      zorder=2)

            subplot.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=16)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=16)   
            subplot.set_title(self.dataset[idx].fileName, fontsize=10)   
            
            subplot.grid()

        self.canvas.draw()


    def initChecks(self):
        """ This methods is used to perform some checks before finishing class initialization. """

        if self.dataset == None:
            raise Exception("No data for resolution function were loaded.")

        try:
            for idx, data in enumerate(self.dataset):
                data.params
        except AttributeError:
            print("No parameters for resolution function at index %i were found.\n" % idx 
                            + "Please use a fitting method before plotting.\n")


