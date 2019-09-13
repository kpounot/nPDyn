import sys, os
import numpy as np

from collections import namedtuple

from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib

from .subPlotsFormat import subplotsFormat
 


class D2OPlot(QWidget):
    """ This class created a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

        Plot        -> plot the normalized experimental data for the selected q-value
        3D Plot     -> plot the whole normalized dataSet
        Analysis    -> plot the different model parameters as a function of q-value
        Resolution  -> plot the fitted model on top of the experimental data for the selected q-value """

    def __init__(self, datasetList):

        super().__init__()

        #_Dataset related attributes
        self.dataset = datasetList

        try:
            self.initChecks()
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
	   
        self.figure.clear()     
        ax = subplotsFormat(self, True, True)
        
        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])

        for idx, subplot in enumerate(ax):
            subplot.errorbar(self.dataset[idx].data.X, 
                        self.dataset[idx].data.intensities[qValIdx],
                        self.dataset[idx].data.errors[qValIdx], 
                        fmt='o')
            
            subplot.set_title(self.dataset[idx].fileName, fontsize=10)
            subplot.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            subplot.grid()

        self.figure.tight_layout()
        self.canvas.draw()




    def compare(self):
        """ This is used to plot the experimental data, without any fit. """
	   
        self.figure.clear()     
        
        ax = self.figure.add_subplot(111)

        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])

        for dataset in self.dataset:
            ax.errorbar(dataset.data.X, 
                        dataset.data.intensities[qValIdx],
                        dataset.data.errors[qValIdx], 
                        label=dataset.fileName,
                        fmt='o')
            
            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            ax.legend(framealpha=0.5)
            ax.grid()

        self.figure.tight_layout()
        self.canvas.draw()





    def plot3D(self):
        """ 3D plot of the whole dataset """

        self.figure.clear()     
        ax = subplotsFormat(self, False, False, '3d') 

        #_Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for i, subplot in enumerate(ax):
            for qIdx in self.dataset[i].data.qIdx:
                subplot.plot(self.dataset[i].data.X, 
                             self.dataset[i].data.intensities[qIdx],
                             self.dataset[i].data.qVals[qIdx], 
                             zdir='y', 
                             zorder=100-qIdx,
                             c=cmap(normColors(self.dataset[i].data.qVals[qIdx])))

            subplot.set_title(self.dataset[i].fileName, fontsize=10)
            subplot.set_xlabel(r'$\hslash \omega (\mu eV)$')
            subplot.set_ylabel(r'$q$')
            subplot.set_zlabel(r'$S(q, \omega)$')
            subplot.grid()

        self.canvas.draw()
    


    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):
        """ This method plots the fitted parameters for each file.
            There is one parameter list for each file, which consists in a q-wise list of scipy's
            OptimizeResult instance. Parameters are retrieved using OptimizeResults.x attribute. """ 

        self.figure.clear()     

        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])


        #_Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, sharex=True, params=True)

        #_Create 2D numpy array to easily access parameters for each file
        paramsList = np.column_stack( [data.params[qValIdx].x for data in self.dataset] )

        #_Plot the parameters of the fits
        for idx, subplot in enumerate(ax):
            subplot.scatter(range(paramsList.shape[1]), paramsList[idx], marker='o')
            subplot.set_ylabel(self.dataset[0].paramsNames[idx]) 
            subplot.set_xticks(range(len(self.dataset)))
            subplot.set_xticklabels([data.fileName for data in self.dataset], 
                                                                rotation=-45, ha='left', fontsize=8)
        
        self.canvas.draw()



    def fitPlot(self):
	   
        self.figure.clear()     

        #_Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, sharey=True)

        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])


        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for idx, dataset in enumerate(self.dataset):
            #_Plot the experimental data
            ax[idx].errorbar(dataset.data.X, 
                        dataset.data.intensities[dataset.data.qIdx][qValIdx],
                        dataset.data.errors[dataset.data.qIdx][qValIdx], 
                        label='Experimental',
                        zorder=1)



            #_Computes resolution function using parameters corresponding the teh right q-value
            resParams = [dataset.resData.params[i] for i in dataset.data.qIdx]

            resF = dataset.resData.model(dataset.data.X, *resParams[qValIdx][0][:-1], 0)

            if dataset.data.norm:
                resF /= resParams[qValIdx][0][0]

            #_Plot the resolution function
            ax[idx].plot( dataset.data.X, 
                          resF,
                          label='Resolution',
                          zorder=3 )



            #_Plot the model
            ax[idx].plot( dataset.data.X,
                          dataset.model(  dataset.params[qValIdx].x, 
                                                    dataset,
                                                    qIdx=qValIdx,
                                                    returnCost=False),
                          label='Model',
                          zorder=4 )

            ax[idx].set_title(dataset.fileName, fontsize=10)
            ax[idx].set_xlabel(r'$\hslash\omega (\mu eV)$')
            ax[idx].set_yscale('log')
            ax[idx].set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$')   
            ax[idx].grid()
        
        self.figure.legend(framealpha=0.5, fontsize=12)
        self.canvas.draw()


#--------------------------------------------------
#_Initialization checks and others
#--------------------------------------------------
    def initChecks(self):
        """ This methods is used to perform some checks before finishing class initialization. """

        for idx, dataset in enumerate(self.dataset):
            try: 
                if not dataset.params:
                    print("WARNING: no fitted parameters were found for data at index %i.\n" % idx    
                      + "Some plotting methods might not work properly.\n")
            except AttributeError:
                print("No parameters for dataset at index %i were found.\n" % idx 
                            + "Please assign a model and use a fitting method before plotting.\n")

