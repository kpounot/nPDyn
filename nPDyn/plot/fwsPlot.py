"""Plotting window for FWS data.

"""

import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSlider,
    QGroupBox,
    QRadioButton,
)
from PyQt5 import QtCore

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import matplotlib

from nPDyn.plot.subPlotsFormat import subplotsFormat

try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass


class FWSPlot(QWidget):
    def __init__(self, dataset):
        """Class that handle the plotting window.

        This class creates a PyQt widget containing a matplotlib
        canvas to draw the plots, a lineedit widget to allow the
        user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.
            - Plot              - plot the normalized experimental data for
                                  the selected q-value
            - 3D Plot           - plot the whole normalized dataSet
            - Analysis          - plot the different model parameters as a
                                  function of q-value
            - Resolution        - plot the fitted model on top of the
                                  experimental data

        """
        super().__init__()

        self.dataset = dataset
        if not isinstance(self.dataset, (list, np.array)):
            self.dataset = [self.dataset]

        self.noFit = False
        self.initChecks()

        self.obsRange = self.get_obsRange()
        self.qRange = self.get_qRange()
        self.eRange = self.get_eRange()

        self.currPlot = self.plot

        # -------------------------------------------------
        # Construction of the GUI
        # -------------------------------------------------
        # A figure instance to plot on
        self.figure = Figure()

        # This is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # Add some interactive elements
        self.button = QPushButton("Plot")
        self.button.clicked.connect(self.plot)

        self.analysisButton = QPushButton("Analysis")
        self.analysisButton.clicked.connect(self.analysisPlot)

        self.plot3DButton = QPushButton("3D Plot")
        self.plot3DButton.clicked.connect(self.plot3D)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        oLayout = QHBoxLayout()
        self.obsLabel = QLabel("Observable index: ", self)
        self.obsSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.obsSlider.setRange(0, self.obsRange.size - 1)
        self.obsSlider.valueChanged.connect(self.updatePlot)
        self.obsSlider.valueChanged.connect(self.updateLabels)
        self.obsVal = QLabel(self.obsRange.astype(str)[0], self)
        oLayout.addWidget(self.obsLabel)
        oLayout.addWidget(self.obsSlider)
        oLayout.addWidget(self.obsVal)

        qLayout = QHBoxLayout()
        self.qLabel = QLabel("Momentum transfer (q) value: ", self)
        self.qSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.qSlider.setRange(0, self.qRange.size - 1)
        self.qSlider.valueChanged.connect(self.updatePlot)
        self.qSlider.valueChanged.connect(self.updateLabels)
        self.qVal = QLabel("%.2f" % self.qRange[0], self)
        qLayout.addWidget(self.qLabel)
        qLayout.addWidget(self.qSlider)
        qLayout.addWidget(self.qVal)

        eLayout = QHBoxLayout()
        self.eLabel = QLabel("Energy transfer value: ", self)
        self.eSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.eSlider.setRange(0, self.eRange.size - 1)
        self.eSlider.valueChanged.connect(self.updatePlot)
        self.eSlider.valueChanged.connect(self.updateLabels)
        self.eVal = QLabel("%.2f" % self.eRange[0], self)
        eLayout.addWidget(self.eLabel)
        eLayout.addWidget(self.eSlider)
        eLayout.addWidget(self.eVal)

        axGroupBox = QGroupBox("Plot data along: ", self)
        axLayout = QHBoxLayout()
        self.oRadioButton = QRadioButton("observables", self)
        self.oRadioButton.setChecked(True)
        self.oRadioButton.clicked.connect(self.updatePlot)
        self.eRadioButton = QRadioButton("energies", self)
        self.eRadioButton.clicked.connect(self.updatePlot)
        self.qRadioButton = QRadioButton("momentum transfers", self)
        self.qRadioButton.clicked.connect(self.updatePlot)
        axLayout.addWidget(self.oRadioButton)
        axLayout.addWidget(self.eRadioButton)
        axLayout.addWidget(self.qRadioButton)
        axGroupBox.setLayout(axLayout)

        self.errBox = QCheckBox("Plot errors", self)
        self.errBox.setCheckState(QtCore.Qt.Checked)
        self.errBox.stateChanged.connect(self.updatePlot)

        if not self.noFit:
            self.fitBox = QCheckBox("Plot fit", self)
            self.fitBox.stateChanged.connect(self.updatePlot)
            self.compBox = QCheckBox("Plot components", self)
            self.compBox.stateChanged.connect(self.updatePlot)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addItem(oLayout)
        layout.addItem(qLayout)
        layout.addItem(eLayout)
        layout.addWidget(axGroupBox)
        layout.addWidget(self.errBox)
        if not self.noFit:
            layout.addWidget(self.fitBox)
            layout.addWidget(self.compBox)
        layout.addWidget(self.button)
        layout.addWidget(self.plot3DButton)
        if not self.noFit:
            layout.addWidget(self.analysisButton)
        self.setLayout(layout)

    # -------------------------------------------------
    # Definitions of the slots for the plot window
    # -------------------------------------------------
    def plot(self):
        """Plot the experimental data, with or without fit"""
        self.currPlot = self.plot

        self.figure.clear()
        ax = subplotsFormat(self, False, True)

        qIdx = self.qSlider.value()
        eIdx = self.eSlider.value()

        for idx, subplot in enumerate(ax):
            obsIdx = self.obsIdx[idx]
            errors = self.dataset[idx].data.errors
            if not self.errBox.isChecked():
                errors = np.zeros_like(errors)

            energies = self.dataset[idx].data.energies
            intensities = self.dataset[idx].data.intensities

            if self.oRadioButton.isChecked():
                X = self.dataset[idx].data.observable
                Y = intensities[:, qIdx, eIdx]
                Err = errors[:, qIdx, eIdx]
                xLabel = self.dataset[idx].data.observable_name
            elif self.qRadioButton.isChecked():
                X = self.dataset[idx].data.qVals
                Y = intensities[obsIdx, :, eIdx]
                Err = errors[obsIdx, :, eIdx]
                xLabel = r"Momentum transfer q $[\AA^{-1}$]"
            elif self.eRadioButton.isChecked():
                X = energies
                Y = intensities[obsIdx, qIdx]
                Err = errors[obsIdx, qIdx]
                xLabel = r"Energies [$\mu eV$]"

            # Plot the data for selected q value
            subplot.errorbar(
                X, Y, Err, fmt="o", label="experimental", zorder=1
            )

            if not self.noFit:
                if self.fitBox.isChecked():
                    if self.oRadioButton.isChecked():
                        Y = self.dataset[idx].fit_best()[:, qIdx, eIdx]
                    elif self.qRadioButton.isChecked():
                        Y = self.dataset[idx].fit_best()[obsIdx, :, eIdx]
                    elif self.eRadioButton.isChecked():
                        Y = self.dataset[idx].fit_best()[obsIdx, qIdx]

                    # Plot the model
                    subplot.plot(
                        X, Y, label=self.dataset[idx].model.name, zorder=3
                    )

                if self.compBox.isChecked():
                    components = self.dataset[idx].fit_components()
                    # Plot the model components
                    for key, val in components.items():
                        if self.oRadioButton.isChecked():
                            Y = val[:, qIdx, eIdx]
                        elif self.qRadioButton.isChecked():
                            Y = val[obsIdx, :, eIdx]
                        elif self.eRadioButton.isChecked():
                            Y = val[obsIdx, qIdx]

                        subplot.plot(X, Y, label=key, ls="--", zorder=2)

            subplot.set_xlabel(xLabel)
            subplot.set_ylabel(r"$S(q, \omega)$")
            subplot.set_title(self.dataset[idx].fileName)
            subplot.legend()

        self.canvas.draw()

    def plot3D(self):
        """3D plot of the whole dataset."""
        self.currPlot = self.plot3D
        self.figure.clear()

        ax = self.figure.subplots(subplot_kw={"projection": "3d"})

        eIdx = self.eSlider.value()

        # Use a fancy colormap
        normColors = matplotlib.colors.Normalize(
            vmin=0, vmax=self.dataset[0].data.intensities.shape[0]
        )
        cmapList = [
            matplotlib.cm.get_cmap("winter"),
            matplotlib.cm.get_cmap("spring"),
            matplotlib.cm.get_cmap("summer"),
            matplotlib.cm.get_cmap("autumn"),
            matplotlib.cm.get_cmap("cool"),
            matplotlib.cm.get_cmap("Wistia"),
        ]

        maxScan = self.dataset[0].data.intensities.shape[0]

        yy = self.dataset[0].data.observable
        ylabel = self.dataset[0].data.observable_name

        xx, yy = np.meshgrid(self.dataset[0].data.qVals, yy)

        ax.plot_wireframe(
            xx,
            yy,
            self.dataset[0].data.intensities[:, :, eIdx],
            label=(
                "$\\Delta E$ = %.2f $\\mu eV$"
                % self.dataset[0].data.energies[eIdx]
            ),
            colors=cmapList[eIdx](normColors(np.arange(maxScan))),
        )

        ax.set_xlabel(r"$q\ [\AA^{-1}]$")
        ax.set_ylabel(ylabel)
        ax.set_zlabel(r"$S(q, \Delta E)$")
        ax.legend(framealpha=0.5)
        ax.grid()

        self.canvas.draw()

    # Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):
        """Plot the fitted parameters."""
        self.currPlot = self.analysisPlot

        self.figure.clear()

        qIdx = self.qSlider.value()

        # Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, None, True)

        # Plot the parameters of the fits
        for fileIdx, dataset in enumerate(self.dataset):
            obsIdx = self.obsIdx[fileIdx]
            params = dataset.params
            qList = dataset.data.qVals

            for idx, key in enumerate(params[0].keys()):
                if self.oRadioButton.isChecked():
                    X = dataset.data.observable
                    Y = np.array([p[key].value for p in params])
                    Err = np.array([p[key].error for p in params])
                    xLabel = dataset.data.observable_name
                elif self.qRadioButton.isChecked():
                    X = qList
                    Y = np.array(params[obsIdx][key].value)
                    Err = np.array(params[obsIdx][key].error)
                    xLabel = r"Momentum transfer q $[\AA^{-1}$]"
                elif self.eRadioButton.isChecked():
                    X = self.eRange
                    Y = np.array(params[obsIdx][key].value).flatten()
                    Err = np.array(params[obsIdx][key].error).flatten()
                    if Y.size == qList.size:
                        Y = Y[qIdx]
                        Err = Err[qIdx]
                    xLabel = r"Energies [$\mu eV$]"

                Y = Y.flatten()
                Err = Err.flatten()

                if not self.errBox.isChecked():
                    Err = np.zeros_like(Err)

                marker = "o"
                if Y.size == 1:
                    Y = np.zeros_like(X) + Y
                    Err = np.zeros_like(X) + Err
                    marker = None

                ax[idx].plot(X, Y, marker=marker, label=dataset.fileName)

                ax[idx].fill_between(X, Y - Err, Y + Err, alpha=0.4)
                ax[idx].set_ylabel(key)
                ax[idx].set_xlabel(xLabel)

        ax[-1].legend(framealpha=0.5)

        self.canvas.draw()

    # -------------------------------------------------
    # Helper functions
    # -------------------------------------------------
    def get_qRange(self):
        """Return the q-values used in the dataset(s).

        This assumes the q-values are the same for all datasets.

        """
        return self.dataset[0].data.qVals

    def get_eRange(self):
        """Return the energies used in the dataset(s).

        This assumes the energies are the same for all datasets.

        """
        return self.dataset[0].data.energies

    def get_obsRange(self):
        """Return the observables used in the dataset(s)."""
        obsRange = []
        for idx, dataset in enumerate(self.dataset):
            for obs in dataset.data.observable:
                if obs not in obsRange:
                    obsRange.append(obs)

        return np.sort(obsRange)

    @property
    def obsIdx(self):
        """Return a list of index of the closest observable value to the
        slider value for each dataset.

        """
        ids = []
        for idx, dataset in enumerate(self.dataset):
            idx = np.argmin(
                (self.obsSlider.value() - dataset.data.observable) ** 2
            )
            ids.append(idx)
        return ids

    def updateLabels(self):
        """Update the labels on the right of the sliders."""
        obsIdx = self.obsSlider.value()
        qIdx = self.qSlider.value()
        eIdx = self.eSlider.value()

        self.obsVal.setText("%.2f" % self.obsRange[obsIdx])
        self.qVal.setText("%.2f" % self.qRange[qIdx])
        self.eVal.setText("%.1f" % self.eRange[eIdx])

    def updatePlot(self):
        """Redraw the current plot based on the selected parameters."""
        return self.currPlot()

    def initChecks(self):
        """This methods is used to perform some checks before
        finishing class initialization.

        """

        if np.any(np.array(self.dataset) is None):
            raise ValueError(
                "No data were loaded.\n"
                "Please import data before using this method."
            )

        for idx, data in enumerate(self.dataset):
            if len(data._fit) == 0:
                print(
                    "No fitted model for resolution function at "
                    "index %i was found.\n"
                    "Some plotting methods are not available.\n" % idx
                )
                self.noFit = True
