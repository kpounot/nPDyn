"""Plotting window for QENS data.

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


class QENSPlot(QWidget):
    def __init__(self, dataset):
        """Class that handle the plotting window.

        This class creates a PyQt widget containing a matplotlib
        canvas to draw the plots, a lineedit widget to allow the
        user to select the q-value to be used to show the data
        and several buttons corresponding to the different type of plots.
            - Plot      - plot the experimental data for
                          the selected observable and q-value.
            - Compare   - plot the datasets on top of each other for
                          direct comparison.
            - 3D Plot   - plot the whole datasets in 3D
                          (energies E, q, S(q, E)).
            - Analysis  - plot the different model parameters as a
                          function of q-value.

        """
        super().__init__()

        self.dataset = dataset
        if not isinstance(self.dataset, (list, np.array)):
            self.dataset = [self.dataset]

        self.noFit = False
        self.initChecks()

        self.obsRange = self.get_obsRange()
        self.qRange = self.get_qRange()

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

        self.compareButton = QPushButton("Compare")
        self.compareButton.clicked.connect(self.compare)

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
        layout.addWidget(self.errBox)
        if not self.noFit:
            layout.addWidget(self.fitBox)
            layout.addWidget(self.compBox)
        layout.addWidget(self.button)
        layout.addWidget(self.compareButton)
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

        obsIdx = self.obsSlider.value()
        qIdx = self.qSlider.value()

        for idx, subplot in enumerate(ax):
            errors = self.dataset[idx].data.errors[obsIdx][qIdx]
            if not self.errBox.isChecked():
                errors = np.zeros_like(errors)

            energies = self.dataset[idx].data.energies
            intensities = self.dataset[idx].data.intensities[obsIdx][qIdx]

            # Plot the data for selected q value
            subplot.errorbar(
                energies,
                intensities,
                errors,
                fmt="o",
                label="experimental",
                zorder=1,
            )

            subplot.set_ylim(
                0.1 * intensities[intensities > 0].min(),
                1.5 * intensities.max(),
            )

            if not self.noFit:
                if self.fitBox.isChecked():
                    # Plot the model
                    subplot.plot(
                        energies,
                        self.dataset[idx].fit_best(x=energies)[obsIdx][qIdx],
                        label=self.dataset[idx].model.name,
                        zorder=3,
                    )

                if self.compBox.isChecked():
                    components = self.dataset[idx].fit_components(x=energies)
                    # Plot the model components
                    for key, val in components.items():
                        subplot.plot(
                            energies,
                            val[obsIdx, qIdx],
                            label=key,
                            ls="--",
                            zorder=2,
                        )

            subplot.set_xlabel(r"$\hslash\omega [\mu eV]$")
            subplot.set_yscale("log")
            subplot.set_ylabel(r"$S(%.2f, \omega)$" % self.qRange[qIdx])
            subplot.set_title(self.dataset[idx].fileName)
            subplot.legend()

        self.canvas.draw()

    def compare(self):
        """Plot the experimental data on one subplot, with or without fit"""
        self.currPlot = self.compare

        self.figure.clear()
        ax = self.figure.add_subplot()

        obsIdx = self.obsSlider.value()
        qIdx = self.qSlider.value()

        for idx, dataset in enumerate(self.dataset):
            errors = dataset.data.errors[obsIdx][qIdx]
            if not self.errBox.isChecked():
                errors = np.zeros_like(errors)

            energies = dataset.data.energies
            intensities = dataset.data.intensities[obsIdx][qIdx]

            # Plot the data for selected q value
            ax.errorbar(
                energies,
                intensities,
                errors,
                fmt="o",
                label="%s" % dataset.fileName,
                zorder=1,
            )

            ax.set_ylim(
                0.1 * intensities[intensities > 0].min(),
                1.5 * intensities.max(),
            )

            ax.set_xlabel(r"$\hslash\omega [\mu eV]$")
            ax.set_yscale("log")
            ax.set_ylabel(r"$S(%.2f, \omega)$" % self.qRange[qIdx])
            ax.set_title(dataset.fileName)
            ax.legend()

        self.canvas.draw()

    def plot3D(self):
        """3D plot of the whole dataset."""
        self.currPlot = self.plot3D

        self.figure.clear()

        obsIdx = self.obsSlider.value()

        # Use a fancy colormap
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap("winter")

        ax = subplotsFormat(self, projection="3d")

        for idx, subplot in enumerate(ax):
            for i, qWiseData in enumerate(
                self.dataset[idx].data.intensities[obsIdx]
            ):
                subplot.plot(
                    self.dataset[idx].data.energies,
                    qWiseData,
                    self.dataset[idx].data.qVals[i],
                    zdir="y",
                    zorder=len(ax) - idx,
                    c=cmap(normColors(self.dataset[idx].data.qVals[i])),
                )

            subplot.set_xlabel(r"$\hslash \omega \ [\mu eV]$")
            subplot.set_ylabel("$q$")
            subplot.set_zlabel(r"$S \ (q, \omega)$")
            subplot.set_title(self.dataset[idx].fileName, fontsize=10)

        self.canvas.draw()

    # Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):
        """Plot the fitted parameters."""
        self.currPlot = self.analysisPlot

        self.figure.clear()

        obsIdx = self.obsSlider.value()

        # Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, None, True)

        # Plot the parameters of the fits
        for fileIdx, dataset in enumerate(self.dataset):
            params = dataset.params[obsIdx]
            qList = dataset.data.qVals

            for idx, key in enumerate(params.keys()):
                values = params[key].value
                errors = params[key].error
                values = np.array(values).flatten()
                errors = np.array(errors).flatten()

                if not self.errBox.isChecked():
                    errors = np.zeros_like(errors)

                marker = "o"
                if values.size == 1:
                    values = np.zeros_like(qList) + values
                    errors = np.zeros_like(qList) + errors
                    marker = None

                ax[idx].plot(
                    qList, values, marker=marker, label=dataset.fileName
                )

                ax[idx].fill_between(
                    qList, values - errors, values + errors, alpha=0.4
                )
                ax[idx].set_ylabel(key)
                ax[idx].set_xlabel(r"$q \ [\AA^{-1}]$")

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

    def get_obsRange(self):
        """Return the observables used in the dataset(s).

        This assumes the observables are the same for all datasets.

        """
        return self.dataset[0].data.observable

    def updateLabels(self):
        """Update the labels on the right of the sliders."""
        obsIdx = self.obsSlider.value()
        qIdx = self.qSlider.value()

        self.obsVal.setText("%.2f" % self.obsRange[obsIdx])
        self.qVal.setText("%.2f" % self.qRange[qIdx])

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
