"""

Classes
^^^^^^^

"""

import sys, os
import numpy as np

from collections import namedtuple

from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame, QCheckBox)
from PyQt5 import QtGui
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib

from nPDyn.plot.subPlotsFormat import subplotsFormat, subplotsFormatWithColorBar
 


class FWSPlot(QWidget):
    """ This class creates a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

            - Plot        - plot the normalized experimental data for the selected q-value
            - 3D Plot     - plot the whole normalized dataSet
            - Analysis    - plot the different model parameters as a function of q-value
            - Fit         - plot the fitted model on top of the experimental data for the selected q-value 

    """

    def __init__(self, dataset):

        super().__init__()

        #_Dataset related attributes
        self.dataset = [dataset]

        self._initChecks()

#--------------------------------------------------
#_Construction of the GUI
#--------------------------------------------------
        #_A figure instance to plot on
        self.figure = Figure()

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

        self.errBox = QCheckBox("Plot errors", self)


        #_Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.errBox)
        layout.addWidget(self.button)
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.fitButton)
        self.setLayout(layout)




    def drawCustomColorBar(self, ax, cmap, dataMin, dataMax):
        """ Draw a custom color bar on the given axis.

            Input:  ax      -> matplotlib's Axes instance on whoch color bar will be drawn
                    cmap    -> color map to be used
                    dataMin -> minimum value for data series
                    dataMax -> maximum value for data series """

        norm = matplotlib.colors.Normalize(dataMin, dataMax)

        matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
        ax.yaxis.tick_right()




#--------------------------------------------------
#_Definitions of the slots for the plot window
#--------------------------------------------------
    def plot(self):
        """ This is used to plot the experimental data, without any fit. """
	   
        self.figure.clear()     
        ax0, ax1 = subplotsFormatWithColorBar(self)
        
        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])

        cmap=matplotlib.cm.get_cmap('winter')

        for idx, subplot in enumerate(ax0):
            for tIdx in range(self.dataset[0].data.intensities.shape[0]):
                subplot.errorbar(   self.dataset[0].data.X, 
                                    self.dataset[0].data.intensities[tIdx, qValIdx],
                                    self.dataset[0].data.errors[tIdx, qValIdx], 
                                    color=cmap(tIdx / self.dataset[0].data.intensities.shape[0]),
                                    label="Scan %i" % (tIdx + 1),
                                    fmt='-o')
                        

            subplot.set_title(self.dataset[0].fileName, fontsize=10)
            subplot.set_xlabel(r'$\hslash\omega [\mu eV]$', fontsize=18)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(q=' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            subplot.grid()


        for ax in ax1:
            
            #_Chech if timestep is given
            if self.dataset[0].timestep is not None:
                yy = self.dataset[0].timestep * np.arange(self.dataset[0].data.intensities.shape[0])
                ylabel = 'Time [h]'
            else:
                yy = np.arange(self.dataset[0].data.intensities.shape[0])
                ylabel = 'Scan number'


            #_Creates a custom color bar
            self.drawCustomColorBar(ax, cmap, yy[0], yy[-1])
            ax.set_aspect(15)
            ax.set_ylabel(ylabel)


        self.figure.tight_layout()
        self.canvas.draw()




    def plot3D(self):
        """ 3D plot of the whole dataset """

        self.figure.clear()     
        ax = subplotsFormat(self, False, False, '3d', FWS=True) 
        

        #_Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=self.dataset[0].data.intensities.shape[0])
        cmapList =  [ matplotlib.cm.get_cmap('winter'),
                      matplotlib.cm.get_cmap('spring'),
                      matplotlib.cm.get_cmap('summer'),
                      matplotlib.cm.get_cmap('autumn'),
                      matplotlib.cm.get_cmap('cool'),
                      matplotlib.cm.get_cmap('Wistia') ]



        for i, subplot in enumerate(ax):

            maxScan = self.dataset[0].data.intensities.shape[0]
            qIds = self.dataset[0].data.qIdx

            #_Chech if timestep is given
            if self.dataset[0].timestep is not None:
                yy = self.dataset[0].timestep * np.arange(self.dataset[0].data.intensities.shape[0])
                ylabel = 'Time [h]'
            else:
                yy = np.arange(self.dataset[0].data.intensities.shape[0])
                ylabel = 'Scan number'


            xx, yy = np.meshgrid(self.dataset[0].data.qVals[qIds], yy)

            subplot.plot_wireframe( xx, 
                                    yy,
                                    self.dataset[0].data.intensities[:,qIds,i],
                                    label = '$\\Delta E$ = %.2f $\mu eV$' % self.dataset[0].data.X[i], 
                                    colors=cmapList[i]( normColors(np.arange(maxScan)) ) )

            subplot.set_xlabel(r'$q\ [\AA^{-1}]$')
            subplot.set_ylabel(ylabel)
            subplot.set_zlabel(r'$S(q, \Delta E)$')
            subplot.legend(framealpha=0.5)
            subplot.grid()


        self.canvas.draw()
    




    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):
        """ This method plots the fitted parameters for each file. """ 

        self.figure.clear()     


        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])


        #_Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, sharex=True, params=True)

        #_Create 2D numpy array to easily access parameters for each file
        paramsList  = self.dataset[0].getParams(qValIdx)

        if self.errBox.isChecked(): #_Whether or not using error bars
            errList = self.dataset[0].getParamsErrors(qValIdx)
        else:
            errList = np.zeros_like(paramsList)

        #_Chech if timestep is given
        if self.dataset[0].timestep is not None:
            X = self.dataset[0].timestep * np.arange(self.dataset[0].data.intensities.shape[0])
            xlabel = 'Time [h]'
        else:
            X = np.arange(self.dataset[0].data.intensities.shape[0])
            xlabel = 'Scan number'

        #_Plot the parameters of the fits
        for idx, subplot in enumerate(ax):
            subplot.errorbar(X, 
                             paramsList[:,idx], 
                             errList[:,idx],
                             marker='o')
            subplot.set_ylabel(self.dataset[0].paramsNames[idx]) 
            subplot.set_xlabel(xlabel)
        
        self.canvas.draw()






    def fitPlot(self):
        """ Plot the fitted model. """
	   
        self.figure.clear()     

        ax = subplotsFormat(self, False, False, '3d')


        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qIdxList = self.dataset[0].data.qIdx
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])


        scanNbr = self.dataset[0].data.intensities.shape[0]

        
        for idx, subplot in enumerate(ax):

            #_Chech if timestep is given
            if self.dataset[0].timestep is not None:
                yy = self.dataset[0].timestep * np.arange(self.dataset[0].data.intensities.shape[0])
                ylabel = 'Time [h]'
            else:
                yy = np.arange(self.dataset[0].data.intensities.shape[0])
                ylabel = 'Scan number'

            #_Plot the datas for selected q value normalized with integrated curves at low temperature
            xx, yy = np.meshgrid(self.dataset[0].data.X, yy)
            subplot.scatter( xx, yy,
                        self.dataset[idx].data.intensities[:,qIdxList[qValIdx]] )


            try:
                #_Plot the model
                params  = self.dataset[idx].getParams(qValIdx)
                subplot.plot_wireframe(xx, yy,
                        np.array([self.dataset[idx].getModel(sIdx, qValIdx)
                                                        for sIdx in range(scanNbr)]),
                        label='Model',
                        color='red',
                        cstride=0)
            except AttributeError:
                continue
            except TypeError:
                continue


        subplot.set_title(self.dataset[idx].fileName, fontsize=10)
        subplot.set_xlabel(r'$\hslash\omega (\mu eV)$')
        subplot.set_ylabel(ylabel)
        subplot.set_zlabel(r'$S(q=' + str(np.round(qValToShow, 2)) + ', \omega)$')   

        self.canvas.draw()




#--------------------------------------------------
#_Initialization checks and others
#--------------------------------------------------
    def _initChecks(self):
        """ This methods is used to perform some checks before finishing class initialization. """

        try: 
            if self.dataset[0].params is None:
                print("WARNING: no fitted parameters were found for data.\n"     
                  + "Some plotting methods might not work properly.\n")
        except AttributeError:
            print("No parameters for dataset were found.\n" 
                        + "Please assign a model and use a fitting method before plotting.\n")
            pass

