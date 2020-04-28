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

matplotlib.interactive(False)

from nPDyn.plot.subPlotsFormat import subplotsFormat, subplotsFormatWithColorBar

class TempRampPlot(QWidget):
    """ This class creates a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

            - Total scattering    - plot the total scattering (sum for all q-values)
            - q-Wise scattering   - plot the scattering for each q-value
            - Fit                 - plot the fitted model on data for all temperatures
            - MSD                 - plot the fitted MSD as a function of temperature 

    """

    def __init__(self, datasetList):

        super().__init__()

        #_Dataset related attributes
        self.dataset = datasetList

        try:
            self._initChecks()
        except Exception as e:
            print(e)
            return





        #--------------------------------------------------
        #_Construction of the GUI
        #--------------------------------------------------

        #_A figure instance to plot on
        self.figure = Figure()

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

        self.errBox = QCheckBox("Plot errors", self)

        #_Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.errBox)
        layout.addWidget(self.totalScatButton)
        layout.addWidget(self.qWiseScatButton)
        layout.addWidget(self.fitButton)
        layout.addWidget(self.msdButton)
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
    def totalScat(self):
        """ This is used to plot the experimental data, without any fit. """

        self.figure.clear()       
        ax = self.figure.add_subplot(111)  

        for dataset in self.dataset:
            ax.errorbar(dataset.data.X, 
                    np.sum(dataset.data.intensities, axis=0),
                    np.sum(dataset.data.errors, axis=0) if self.errBox.isChecked() else None,
                    label=dataset.fileName)

            ax.set_xlabel(r'$Temperature (K)$')
            ax.set_ylabel(r'$Scattering$')
            ax.legend(framealpha=0.5, fontsize=12)

        self.canvas.draw()
    


    def qWiseScat(self):
        """ For each file, plots the temperature of elastic scattering intensity for each q-value. """

        self.figure.clear()     
        ax0, ax1 = subplotsFormatWithColorBar(self)


        #_Use a fancy colormap
        cmap = matplotlib.cm.get_cmap('winter')

        for i, subplot in enumerate(ax0):

            qValFirst   = self.dataset[i].data.qVals[self.dataset[i].data.qIdx][0]
            qValLast    = self.dataset[i].data.qVals[self.dataset[i].data.qIdx][-1]
            normColors = matplotlib.colors.Normalize(vmin=qValFirst, vmax=qValLast)

            for qIdx in self.dataset[i].data.qIdx:
                subplot.plot(self.dataset[i].data.X, 
                             self.dataset[i].data.intensities[qIdx],
                             label=self.dataset[i].data.qVals[qIdx],
                             c=cmap(normColors(self.dataset[i].data.qVals[qIdx])))

            subplot.set_title(self.dataset[i].fileName, fontsize=10)
            subplot.set_xlabel(r'$Temperature (K)$')
            subplot.set_ylabel(r'$Scattering$')
            subplot.grid()

        for ax in ax1:
            #_Creates a custom color bar
            qValFirst   = self.dataset[i].data.qVals[self.dataset[i].data.qIdx][0]
            qValLast    = self.dataset[i].data.qVals[self.dataset[i].data.qIdx][-1]
            self.drawCustomColorBar(ax, cmap, qValFirst, qValLast)
            ax.set_ylabel('q^2 \ [$\AA^{-2}$]')


        self.canvas.draw()
    


    #_Plot of the parameters resulting from the fit procedure
    def fit(self):
        """ Plots the fitted model. """

        self.figure.clear()     
        ax = subplotsFormat(self, True, True)

        for i, subplot in enumerate(ax):
            #_Obtaining the temperature to plot as being the closest one to the number entered by the user 
            tempToShow = min(self.dataset[i].data.X, key = lambda x : abs(float(self.lineEdit.text()) - x))
            tempIdx = int(np.argwhere(self.dataset[i].data.X == tempToShow)[0])


            #_Plotting experimental data
            subplot.errorbar(   self.dataset[i].data.qVals**2, 
                                self.dataset[i].data.intensities[:,tempIdx],
                                self.dataset[i].data.errors[:,tempIdx] if self.errBox.isChecked() else None,
                                label="Experimental"   )
    
            #_Adding a transparent blue region to easily locate the fitted data points
            #_(provided that no detectors were discarded between fitting and plotting)
            qMin = self.dataset[i].data.qVals[self.dataset[i].data.qIdx[0]]
            qMax = self.dataset[i].data.qVals[self.dataset[i].data.qIdx[-1]]
            subplot.axvspan(qMin**2, qMax**2, color='c', alpha=0.4)

            #_Plotting the model and adding the parameters values to the label
            label = "Model\n"
            if len(self.dataset[i].params) == len(self.dataset[i].paramsNames): #_Formatting the legend output
                paramVals = np.round(self.dataset.params[tempIdx][0], 2).astype(str)
                for pIdx, val in enumerate(paramVals): 
                    label += self.dataset[i].paramsNames[pIdx] + ": " + val + "\n"

            subplot.plot(   self.dataset[i].data.qVals**2,
                            self.dataset[i].model(self.dataset[i].data.qVals, 
                                                    *self.dataset[i].params[tempIdx][0]),
                            label=label)

            subplot.set_title(self.dataset[i].fileName, fontsize=10)
            subplot.set_xlabel(r'$Scattering \ vector \ q^2 \ (\AA^{-2})$')
            subplot.set_ylabel(r'EISF at %d K' % tempToShow)
            subplot.legend(framealpha=0.5, fontsize=12)

        self.canvas.draw()
 

    def MSD(self):
        """ Plots the fitted mean-squared displacement (MSD). """
	   
        self.figure.clear()     
        ax = self.figure.add_subplot(111)  


        markers = ['o', 's', 'v', '^', 'h', 'p', 'd', '*', 'P', 'D', '+', '1']

        #_Plot the mean-squared displacement as a function of temperature for each file
        for i, dataset in enumerate(self.dataset):
            #_Obtaining the temperature to plot as being the closest one to the number entered by the user 
            tempToShow = min(dataset.data.X, key = lambda x : abs(float(self.lineEdit.text()) - x))
            tempIdx = int(np.argwhere(dataset.data.X == tempToShow)[0])

            #_Extracting the MSD from parameters for each temperature
            msdList = [dataset.params[tempIdx][0][1] for tempIdx, temp in enumerate(dataset.data.X)]
            qMin = dataset.data.qVals[dataset.data.qIdx[0]]
            qMax = dataset.data.qVals[dataset.data.qIdx[-1]]

            if self.errBox.isChecked():
                #_Computing the errors for each temperature from covariant matrix
                errList = [np.sqrt(np.diag(dataset.params[tempIdx][1]))[1] for tempIdx, temp 
                                                                    in enumerate(dataset.data.X)]
            else:
                errList = np.zeros_like(msdList)

            #_Plotting the MSD
            ax.errorbar(dataset.data.X[:tempIdx+1], 
                        msdList[:tempIdx+1],
                        errList[:tempIdx+1], 
                        marker=markers[i],
                        label = dataset.fileName)

            ax.set_xlabel(r'$Temperature [K]$')
            ax.set_ylabel(r'$MSD \ [\AA^{2}]$, q=%.2f to %.2f [$\AA^{-1}]$' % ( qMin, qMax ))
            ax.legend(framealpha=0.5, fontsize=12, loc='upper left')

        self.canvas.draw()



    def _initChecks(self):
        """ This methods is used to perform some checks before finishing class initialization. """


        for idx, dataset in enumerate(self.dataset):
            try: 
                if not dataset.params:
                    print("WARNING: no fitted parameters were found for data at index %i.\n" % idx    
                      + "Some plotting methods might not work properly.\n")
            except AttributeError:
                print("No parameters for dataset at index %i were found.\n" % idx 
                            + "Please use a fitting method before plotting.\n")

