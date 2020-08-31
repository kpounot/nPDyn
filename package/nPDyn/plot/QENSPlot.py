"""

Classes
^^^^^^^

"""

import numpy as np

from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout,
                             QFrame, QCheckBox)

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg
                                                as FigureCanvas)
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT
                                                as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib

from nPDyn.plot.subPlotsFormat import subplotsFormat

matplotlib.use('Qt5Agg')


class QENSPlot(QWidget):
    """ This class creates a PyQt widget containing a matplotlib canvas
        to draw the plots, a lineedit widget to allow the user to select
        the q-value to be used to show the data and several buttons
        corresponding to the different type of plots.

            - Plot              - plot the normalized experimental data for
                                  the selected q-value
            - Compare           - superimpose experimental data on one plot
            - 3D Plot           - plot the whole normalized dataSet
            - Analysis          - plot the different model parameters as a
                                  function of q-value
            - q-wise analysis   - plot the different model parameters as a
                                  function of q-value
            - Fit               - plot the fitted model on top of the
                                  experimental data

    """

    def __init__(self, datasetList):

        super().__init__()

        # Dataset related attributes
        self.dataset = datasetList

        try:
            self._initChecks()
        except Exception as e:
            print(e)
            return

# -------------------------------------------------
# Construction of the GUI
# -------------------------------------------------
        # A figure instance to plot on
        self.figure = Figure()

        # This is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # Add some interactive elements
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        self.compareButton = QPushButton('Compare')
        self.compareButton.clicked.connect(self.compare)

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

        self.errBox = QCheckBox("Plot errors", self)



        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.errBox)
        layout.addWidget(self.button)
        layout.addWidget(self.compareButton)
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.qWiseAnalysisButton)
        layout.addWidget(self.fitButton)
        self.setLayout(layout)





# -------------------------------------------------
# Definitions of the slots for the plot window
# -------------------------------------------------
    def plot(self):
        """ This is used to plot the experimental data, without any fit. """

        self.figure.clear()
        ax = subplotsFormat(self, True, True)

        # Obtaining the q-value to plot as being the closest
        # one to the number entered by the user
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(
            qVals, key = lambda x: abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])

        for idx, subplot in enumerate(ax):
            subplot.errorbar(
                self.dataset[idx].data.X,
                self.dataset[idx].data.intensities[qValIdx],
                self.dataset[idx].data.errors[qValIdx],
                fmt='o')

            subplot.set_title(self.dataset[idx].fileName, fontsize=10)
            subplot.set_xlabel(r'$\hslash\omega \ [\mu eV]$', fontsize=18)
            subplot.set_yscale('log')
            subplot.set_ylabel(r'$S(q=' + str(np.round(qValToShow, 2))
                               + ', \omega)$', fontsize=18)
            subplot.grid()

        self.canvas.draw()




    def compare(self):
        """ This is used to plot the experimental data, without any fit. """

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        # Obtaining the q-value to plot as being the closest one
        # to the number entered by the user
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(
            qVals, key = lambda x: abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])

        for dataset in self.dataset:
            ax.errorbar(dataset.data.X,
                        dataset.data.intensities[dataset.data.qIdx][qValIdx],
                        dataset.data.errors[dataset.data.qIdx][qValIdx],
                        label=dataset.fileName,
                        fmt='o')

            ax.set_xlabel(r'$\hslash\omega \ [\mu eV]$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(q=' + str(np.round(qValToShow, 2))
                          + ', \omega)$', fontsize=18)

            ax.legend(framealpha=0.5)
            ax.grid()


        # Set some limits on axes
        minData = np.min(self.dataset[0].data.intensities)
        maxData = np.max(self.dataset[0].data.intensities)
        maxX    = self.dataset[0].data.X[-1]
        ax.set_xlim(-1.5 * maxX, 1.5 * maxX)
        ax.set_ylim(0.5 * minData, 1.2 * maxData)


        self.canvas.draw()





    def plot3D(self):
        """ 3D plot of the whole dataset """

        self.figure.clear()
        ax = subplotsFormat(self, False, False, '3d')

        # Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for i, subplot in enumerate(ax):
            for qIdx in self.dataset[i].data.qIdx:
                subplot.plot(
                    self.dataset[i].data.X,
                    self.dataset[i].data.intensities[qIdx],
                    self.dataset[i].data.qVals[qIdx],
                    zdir='y',
                    zorder=100 - qIdx,
                    c=cmap(normColors(self.dataset[i].data.qVals[qIdx])))

            subplot.set_title(self.dataset[i].fileName, fontsize=10)
            subplot.set_xlabel(r'$\hslash \omega \ [\mu eV]$')
            subplot.set_ylabel(r'$q \ [\AA^{-1}]$')
            subplot.set_zlabel(r'$S(q, \omega)$')
            subplot.grid()

        self.canvas.draw()



    # Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):
        """ This method plots the fitted parameters for each file. """

        self.figure.clear()

        # Obtaining the q-value to plot as being the closest one
        # to the number entered by the user
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(
            qVals, key = lambda x: abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])


        # Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, None, True)

        # Create 2D numpy array to easily access parameters for each file
        paramsList = np.column_stack([data.getParams(qValIdx) for data
                                      in self.dataset])

        if self.errBox.isChecked():  # Whether or not using error bars
            errList = np.column_stack([data.getParamsErrors(qValIdx)
                                       for data in self.dataset])
        else:
            errList = np.zeros_like(paramsList)

        # Plot the parameters of the fits
        for idx, subplot in enumerate(ax):
            subplot.errorbar(range(paramsList.shape[1]),
                             paramsList[idx],
                             errList[idx],
                             marker='o')
            subplot.set_ylabel(self.dataset[0].paramsNames[idx])
            subplot.set_xticks(range(len(self.dataset)))
            subplot.set_xticklabels([data.fileName for data in self.dataset],
                                    rotation=-45, ha='left', fontsize=8)

        self.canvas.draw()





    def qWiseAnalysis(self):
        """ This method provides a quick way to plot the q-dependence
            of weight factors and lorentzian widths for each contribution.

        """

        self.figure.clear()

        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qIds  = np.arange(qVals.size)

        ax = self.figure.subplots(
            len(self.dataset[0].getWeights_and_lorWidths(0)[0]),
            2,
            sharex=True)
        ax = ax.reshape(
            len(self.dataset[0].getWeights_and_lorWidths(0)[0]), 2)

        for dIdx, dataset in enumerate(self.dataset):
            # Get parameters for each q-value
            weights   = []
            lorWidths = []
            labels    = []
            for idx in qIds:
                params = dataset.getWeights_and_lorWidths(idx)
                weights.append(params[0].astype(float))
                lorWidths.append(params[1].astype(float))
                labels = params[2]

            weights = np.row_stack(weights)
            lorWidths = np.row_stack(lorWidths)

            weightErr   = np.zeros((qVals.size, weights[0].size))
            lorWidthErr = np.zeros((qVals.size, lorWidths[0].size))

            if self.errBox.isChecked():  # Whether or not using error bars
                for idx in qIds:
                    params = dataset.getWeights_and_lorErrors(idx)
                    weightErr.append(params[0].astype(float))
                    lorWidthErr.append(params[1].astype(float))

                weightErr = np.row_stack(weightErr)
                lorWidthErr = np.row_stack(lorWidthErr)


            for idx, row in enumerate(ax):
                row[0].errorbar(qVals,
                                weights[:, idx],
                                weightErr[:, idx],
                                marker='o',
                                label=dataset.fileName)
                row[0].set_ylabel('Weight - %s' % labels[idx])
                row[0].set_xlabel(r'$q \ [\AA^{-1}$]')
                row[0].set_ylim(0, 1)

                row[1].errorbar(qVals,
                                lorWidths[:, idx],
                                lorWidthErr[:, idx],
                                marker='o',
                                label=dataset.fileName)
                row[1].set_ylabel('Width - %s' % labels[idx])
                row[1].set_xlabel(r'$q \ [\AA^{-1}$]')

        ax[-1][-1].legend(loc='upper left',
                          bbox_to_anchor=(1.1, 1),
                          fontsize=10)

        self.canvas.draw()



    def fitPlot(self):
        """ Plots the fitted model for each file. """

        self.figure.clear()

        # Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, sharey=True)

        # Obtaining the q-value to plot as being the closest one to
        # the number entered by the user
        qVals = self.dataset[0].data.qVals[self.dataset[0].data.qIdx]
        qValToShow = min(
            qVals, key = lambda x: abs(float(self.lineEdit.text()) - x))
        qValIdx = int(np.argwhere(qVals == qValToShow)[0])


        # Plot the datas for selected q value normalized with
        # integrated curves at low temperature
        for idx, dataset in enumerate(self.dataset):
            # Plot the experimental data
            if self.errBox.isChecked():
                errors = dataset.data.errors[dataset.data.qIdx][qValIdx]
            else:
                errors = np.zeros_like(
                    dataset.data.intensities[dataset.data.qIdx][qValIdx])

            ax[idx].errorbar(
                dataset.data.X,
                dataset.data.intensities[dataset.data.qIdx][qValIdx],
                errors,
                label='Experimental',
                fmt='o',
                zorder=1)



            # Plot the background
            bkgd = dataset.getBackground(qValIdx)
            if bkgd is not None:
                ax[idx].axhline(bkgd, label='Background', zorder=2)



            # Computes resolution function using parameters
            # corresponding the right q-value
            resF = dataset.getSubCurves(qValIdx)[0]

            # Plot the resolution function
            ax[idx].plot(dataset.data.X,
                         resF,
                         label='Resolution',
                         ls=':',
                         zorder=3)




            # Plot the D2O signal, if any
            if dataset.D2OData is not None:
                D2OSignal = dataset.getD2OSignal(qValIdx)

                ax[idx].plot(dataset.data.X,
                             D2OSignal,
                             label='$D_2O$',
                             ls=':',
                             zorder=4)




            # Plot the lorentzians
            lorArrays = dataset.getSubCurves(qValIdx)[1:]
            for lorId, val in enumerate(lorArrays[:-1]):
                ax[idx].plot(dataset.data.X,
                             val,
                             ls='--',
                             label=lorArrays[-1][lorId],
                             zorder=6)



            # Plot the model
            ax[idx].plot(dataset.data.X,
                         dataset.getModel(qValIdx),
                         label='Model',
                         color='red',
                         zorder=7)


            ax[idx].set_title(dataset.fileName, fontsize=10)
            ax[idx].set_xlabel(r'$\hslash\omega [\mu eV]$')
            ax[idx].set_yscale('log')
            ax[idx].set_ylabel(r'$S(q=' + str(np.round(qValToShow, 2))
                               + ', \omega)$')


            # Set some limits on axis
            qData   = dataset.data.intensities[dataset.data.qIdx][qValIdx]
            minData = np.min(qData[qData > 0])
            maxData = np.max(qData)
            maxX    = dataset.data.X[-1]

            ax[idx].set_xlim(-1.1 * maxX, 1.1 * maxX)
            ax[idx].set_ylim(0.5 * minData, 1.8 * maxData)


        ax[-1].legend(framealpha=0.5, fontsize=12)
        self.canvas.draw()


# -------------------------------------------------
# Initialization checks and others
# -------------------------------------------------
    def _initChecks(self):
        """ This methods is used to perform some checks before
            finishing class initialization.

        """

        for idx, dataset in enumerate(self.dataset):
            try:
                if not dataset.params:
                    print("WARNING: no fitted parameters were found "
                          "for data at index %i.\n"
                          "Some plotting methods might "
                          "not work properly.\n" % idx)
            except AttributeError:
                print("No parameters for dataset at index "
                      "%i were found.\n"
                      "Please assign a model and use a fitting "
                      "method before plotting.\n" % idx)
