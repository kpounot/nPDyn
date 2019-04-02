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

from .subPlotsFormat import subplotsFormat, subplotsFormatWithColorBar
 


class FWSPlot(QWidget):
    """ This class created a PyQt widget containing a matplotlib canvas to draw the plots,
        a lineedit widget to allow the user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.

        Plot        -> plot the normalized experimental data for the selected q-value
        Compare     -> superimpose experimental data on one plot
        3D Plot     -> plot the whole normalized dataSet
        Analysis    -> plot the different model parameters as a function of q-value
        Resolution  -> plot the fitted model on top of the experimental data for the selected q-value """

    def __init__(self, datasetList):

        self.app = QApplication(sys.argv)

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
        self.figure = plt.figure()

        #_This is the Canvas Widget that displays the `figure`
        #_it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        #_Add some interactive elements
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        self.analysisButton = QPushButton('Analysis')
        self.analysisButton.clicked.connect(self.analysisPlot)

        self.qWiseAnalysisButton = QPushButton('q-wise analysis')
        self.qWiseAnalysisButton.clicked.connect(self.qWiseAnalysis)

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

        self.datasetLabel = QLabel('Data index to plot', self)
        self.datasetlineEdit = QLineEdit(self) 
        self.datasetlineEdit.setText('0')


        #_Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.datasetLabel)
        layout.addWidget(self.datasetlineEdit)
        layout.addWidget(self.button)
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.qWiseAnalysisButton)
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
	   
        plt.gcf().clear()     
        ax = subplotsFormatWithColorBar(self)
        
        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])

        cmap=plt.get_cmap('winter')

        for idx, subplot in enumerate(ax[::2]):
            for tIdx in range(self.dataset[idx].data.intensities.shape[0]):
                subplot.errorbar(   self.dataset[idx].data.X, 
                                    self.dataset[idx].data.intensities[tIdx, qValIdx],
                                    self.dataset[idx].data.errors[tIdx, qValIdx], 
                                    color=cmap(tIdx / self.dataset[idx].data.intensities.shape[0]),
                                    label="Scan %i" % (tIdx + 1),
                                    fmt='o')
                        
            #_Creates a custom color bar
            self.drawCustomColorBar(ax[2*idx+1], cmap, 0, self.dataset[0].data.intensities.shape[0])
            ax[2*idx+1].set_aspect(15)
            ax[2*idx+1].set_ylabel('Scan number')

            subplot.set_title(self.dataset[idx].fileName, fontsize=10)
            subplot.set_xlabel(r'$\hslash\omega [\mu eV]$', fontsize=18)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(q=' + str(np.round(qValToShow, 2)) + ', \omega)$', fontsize=18)   
            subplot.grid()


        self.figure.tight_layout()
        self.canvas.draw()




    def plot3D(self):
        """ 3D plot of the whole dataset """

        plt.gcf().clear()     
        ax = subplotsFormat(self, False, False, '3d') 
        

        dIdx = int(self.datasetlineEdit.text()) #_Data index to plot in self.dataset list

        #_Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=self.dataset[dIdx].data.intensities.shape[0])
        cmapList =  [ matplotlib.cm.get_cmap('winter'),
                      matplotlib.cm.get_cmap('spring'),
                      matplotlib.cm.get_cmap('summer'),
                      matplotlib.cm.get_cmap('autumn'),
                      matplotlib.cm.get_cmap('cool'),
                      matplotlib.cm.get_cmap('Wistia') ]



        for i, subplot in enumerate(ax):

            maxScan = self.dataset[dIdx].data.intensities.shape[0]
            qIds = self.dataset[dIdx].data.qIdx

            xx, yy = np.meshgrid(self.dataset[dIdx].data.qVals[qIds], 
                                 np.arange(self.dataset[dIdx].data.intensities.shape[0]))

            subplot.plot_wireframe( xx, 
                                    yy,
                                    self.dataset[dIdx].data.intensities[:,qIds,i],
                                    label = '$\\Delta E$ = %.2f $\mu eV$' % self.dataset[dIdx].data.X[i], 
                                    colors=cmapList[i]( normColors(np.arange(maxScan)) ) )

            subplot.set_xlabel(r'$q\ [\AA^{-1}]$')
            subplot.set_ylabel(r'$Scan \ number$')
            subplot.set_zlabel(r'$S(q, \Delta E)$')
            subplot.legend(framealpha=0.5)
            subplot.grid()


        self.canvas.draw()
    




    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):
        """ This method plots the fitted parameters for each file.
            There is one parameter list for each file, which consists in a q-wise list of scipy's
            OptimizeResult instance. Parameters are retrieved using OptimizeResults.x attribute. """ 

        plt.gcf().clear()     

        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(qVals, key = lambda x : abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])


        #_Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, sharex=True, params=True)

        #_Create 2D numpy array to easily access parameters for each file
        paramsList = np.column_stack( [data.params[qValIdx].x for data in self.dataset] )
        errList    = np.column_stack( [ np.sqrt(np.diag(
                                        data.params[qValIdx].lowest_optimization_result.hess_inv.todense())) 
                                        for data in self.dataset ] )

        #_Plot the parameters of the fits
        for idx, subplot in enumerate(ax):
            subplot.errorbar(range(paramsList.shape[1]), paramsList[idx], marker='o')
            subplot.set_ylabel(self.dataset[0].paramsNames[idx]) 
            subplot.set_xticks(range(len(self.dataset)))
            subplot.set_xticklabels([data.fileName for data in self.dataset], 
                                                                rotation=-45, ha='left', fontsize=8)
        
        self.canvas.draw()





    def qWiseAnalysis(self):
        """ This method provides a quick way to plot the q-dependence of weight factors and lorentzian
            widths for each contribution. """

        plt.gcf().clear()     

        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]**2
        qIds  = np.arange(qVals.size)

        ax = self.figure.subplots(len(self.dataset[0].getWeights_and_lorWidths(0)[0]), 2, sharex=True)

        for dIdx, dataset in enumerate(self.dataset):
            #_Get parameters for each q-value
            paramsList = np.array( [dataset.getWeights_and_lorWidths(idx) for idx in qIds] )
            errList = np.array( [dataset.getWeights_and_lorErrors(idx) for idx in qIds] ).astype(float)

            labels = paramsList[0,-1]                       #_Extracts labels
            paramsList = paramsList[:,:-1].astype(float)    #_Extracts parameters values 


            for idx, row in enumerate(ax):
                row[0].errorbar(qVals, paramsList[:,0,idx], errList[:,0,idx], marker='o')
                row[0].set_ylabel('Weight - %s' % labels[idx])
                row[0].set_xlabel(r'q [$\AA^{-2}$]')
                row[0].set_ylim(0., 1.2)

                row[1].errorbar(qVals, paramsList[:,1,idx], errList[:,1,idx], marker='o')
                row[1].set_ylabel('Width - %s' %labels[idx])
                row[1].set_xlabel(r'q [$\AA^{-2}$]')
                row[1].set_ylim(0., 1.2*np.max(paramsList[:,1,idx]))

        self.canvas.draw()



    def fitPlot(self):
	   
        plt.gcf().clear()     

        #_Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, sharey=True)

        #_Obtaining the q-value to plot as being the closest one to the number entered by the user 
        qIdxList = self.dataset[0].data.qIdx
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



            #_Plot the background
            bkgd = dataset.getBackground(qValIdx)
            if bkgd is not None:
                ax[idx].axhline(bkgd, label='Background', zorder=2)



            #_Computes resolution function using parameters corresponding the teh right q-value
            resParams = [dataset.resData.params[i] for i in dataset.data.qIdx]

            resF = dataset.resData.model(dataset.data.X, *resParams[qValIdx][0][:-1], 0)

            if dataset.data.norm:
                resF /= resParams[qValIdx][0][0]

            #_Plot the resolution function
            ax[idx].plot( dataset.data.X, 
                          resF,
                          label='Resolution',
                          ls=':',
                          zorder=3 )




            #_Plot the D2O signal, if any
            if dataset.D2OData is not None:
                temp = dataset.data.temp.mean()
                gD2O = dataset.D2OData.sD2O(temp, qValToShow)
                maxD2O = np.max( dataset.D2OData.data.intensities[dataset.data.qIdx][qValIdx])

                if not dataset.data.norm: #_Scale the normalized D2O signal if data were not normalized
                    maxD2O * normF

                D2OSignal = maxD2O * dataset.D2OData.volFraction * gD2O / (gD2O**2 + dataset.data.X**2) 

                ax[idx].plot(   dataset.data.X,
                                D2OSignal,
                                label='D2O',
                                ls=':',
                                zorder=4 )




            #_Plot the lorentzians
            for val in zip( *dataset.getWeights_and_lorWidths(qValIdx) ):
                ax[idx].plot(   dataset.data.X,
                                val[0] * val[1] / (val[1]**2 + dataset.data.X**2),
                                ls='--',
                                label=val[2],
                                zorder=5)



            #_Plot the model
            ax[idx].plot( dataset.data.X,
                          dataset.model(  dataset.params[qValIdx].x, 
                                                    dataset,
                                                    qIdx=qValIdx,
                                                    returnCost=False),
                          label='Model',
                          zorder=6 )

            ax[idx].set_title(dataset.fileName, fontsize=10)
            ax[idx].set_xlabel(r'$\hslash\omega (\mu eV)$')
            ax[idx].set_yscale('log')
            ax[idx].set_ylabel(r'$S(' + str(np.round(qValToShow, 2)) + ', \omega)$')   
            ax[idx].grid()
        
        plt.legend(framealpha=0.5, fontsize=12)
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

