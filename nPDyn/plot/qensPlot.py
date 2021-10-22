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
from nPDyn.plot.create_window import makeWindow

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
        self.noFit = False

        self.initChecks()

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

        self.analysisQButton = QPushButton("Analysis - q-wise")
        self.analysisQButton.clicked.connect(self.analysisQPlot)

        self.analysisObsButton = QPushButton("Analysis - observable-wise")
        self.analysisObsButton.clicked.connect(self.analysisObsPlot)

        self.plot3DButton = QPushButton("3D Plot")
        self.plot3DButton.clicked.connect(self.plot3D)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        oLayout = QHBoxLayout()
        self.obsLabel = QLabel("Observable index: ", self)
        self.obsSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.obsSlider.setRange(0, self.get_obsRange().size - 1)
        self.obsSlider.valueChanged.connect(self.updatePlot)
        self.obsSlider.valueChanged.connect(self.updateLabels)
        self.obsVal = QLabel(self.get_obsRange().astype(str)[0], self)
        oLayout.addWidget(self.obsLabel)
        oLayout.addWidget(self.obsSlider)
        oLayout.addWidget(self.obsVal)

        qLayout = QHBoxLayout()
        self.qLabel = QLabel("Momentum transfer (q) value: ", self)
        self.qSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.qSlider.setRange(0, self.get_qRange().size - 1)
        self.qSlider.valueChanged.connect(self.updatePlot)
        self.qSlider.valueChanged.connect(self.updateLabels)
        self.qVal = QLabel("%.2f" % self.get_qRange()[0], self)
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

        self.legendBox = QCheckBox("Show legend", self)
        self.legendBox.setCheckState(QtCore.Qt.Checked)
        self.legendBox.stateChanged.connect(self.updatePlot)

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
        layout.addWidget(self.legendBox)
        layout.addWidget(self.button)
        layout.addWidget(self.compareButton)
        layout.addWidget(self.plot3DButton)
        if not self.noFit:
            layout.addWidget(self.analysisQButton)
            layout.addWidget(self.analysisObsButton)
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

        plot_errors = False
        if self.errBox.isChecked():
            plot_errors = True

        plot_legend = False
        if self.legendBox.isChecked():
            plot_legend = True

        ymin = (
            0.1
            * np.asarray(self.dataset[0])[
                np.asarray(self.dataset[0]) > 0
            ].min()
        )
        ymax = 5 * np.asarray(self.dataset[0]).max()
        for idx, subplot in enumerate(ax):
            data = self.dataset[idx]
            axes = data.axes.copy()
            if data.observable in axes:
                obs_ax = data.axes.index(data.observable)
                data = data.take(obsIdx, obs_ax)
            if "q" in axes:
                q_ax = data.axes.index("q")
                data = data.take(qIdx, q_ax)

            data.plot(
                subplot,
                plot_errors=plot_errors,
                plot_legend=plot_legend,
                label="experimental",
            )

            tmpmin = 0.1 * np.asarray(data)[np.asarray(data) > 0].min()
            tmpmax = 5 * np.asarray(data).max()
            ymin = min((tmpmin, ymin))
            ymax = max((tmpmax, ymax))

            if not self.noFit:
                fit = data.fit_best()
                components = data.fit_components()
                if data.observable in axes:
                    fit = fit.take(obsIdx, obs_ax)
                if "q" in axes:
                    fit = fit.take(qIdx, q_ax)

                if self.fitBox.isChecked():
                    # Plot the model
                    subplot.plot(
                        getattr(data, axes[-1]),
                        fit,
                        label=self.dataset[idx].model.name,
                        zorder=3,
                    )

                if self.compBox.isChecked():
                    # Plot the model components
                    for key, val in components.items():
                        if data.observable in axes:
                            val = val.take(obsIdx, obs_ax)
                        if "q" in axes:
                            val = val.take(qIdx, q_ax)
                        subplot.plot(
                            getattr(data, axes[-1]),
                            val,
                            label=key,
                            ls="--",
                            zorder=2,
                        )

                if plot_legend:
                    subplot.legend()

            subplot.set_ylim(ymin, ymax)
            subplot.set_title(self.dataset[idx].name)

        self.canvas.draw()

    def compare(self):
        """Plot the experimental data on one subplot, with or without fit"""
        self.currPlot = self.compare

        self.figure.clear()
        ax = self.figure.add_subplot()

        obsIdx = self.obsSlider.value()
        qIdx = self.qSlider.value()

        plot_errors = False
        if self.errBox.isChecked():
            plot_errors = True

        plot_legend = False
        if self.legendBox.isChecked():
            plot_legend = True

        ymin = (
            0.1
            * np.asarray(self.dataset[0])[
                np.asarray(self.dataset[0]) > 0
            ].min()
        )
        ymax = 5 * np.asarray(self.dataset[0]).max()
        for idx, data in enumerate(self.dataset):
            axes = data.axes.copy()
            if data.observable in axes:
                obs_ax = data.axes.index(data.observable)
                data = data.take(obsIdx, obs_ax)
            if "q" in axes:
                q_ax = data.axes.index("q")
                data = data.take(qIdx, q_ax)

            data.plot(
                ax,
                plot_errors=plot_errors,
                plot_legend=plot_legend,
                label=data.name,
            )

            tmpmin = 0.1 * np.asarray(data)[np.asarray(data) > 0].min()
            tmpmax = 5 * np.asarray(data).max()
            ymin = min((tmpmin, ymin))
            ymax = max((tmpmax, ymax))

        ax.set_ylim(ymin, ymax)

        self.canvas.draw()

    def plot3D(self):
        """3D plot of the whole dataset."""
        self.currPlot = self.plot3D

        self.figure.clear()

        obsIdx = self.obsSlider.value()
        ax = subplotsFormat(self, projection="3d")
        for idx, subplot in enumerate(ax):
            self.dataset[idx].plot_3D(
                subplot,
                index=obsIdx,
            )

            if self.legendBox.isChecked():
                subplot.set_title(self.dataset[idx].name, fontsize=10)

        self.canvas.draw()

    # Plot of the parameters resulting from the fit procedure
    def analysisQPlot(self):
        """Plot the fitted parameters."""
        self.currPlot = self.analysisQPlot

        self.figure.clear()

        obsIdx = self.obsSlider.value()

        # Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, None, True)

        # Plot the parameters of the fits
        for fileIdx, dataset in enumerate(self.dataset):
            params = dataset.params[obsIdx]
            qList = dataset.q

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

                ax[idx].plot(qList, values, marker=marker, label=dataset.name)

                ax[idx].fill_between(
                    qList, values - errors, values + errors, alpha=0.4
                )
                ax[idx].set_ylabel(key)
                ax[idx].set_xlabel(r"$q \ [\AA^{-1}]$")

        if self.legendBox.isChecked():
            ax[-1].legend(framealpha=0.5)

        self.canvas.draw()

    # Plot of the parameters resulting from the fit procedure
    def analysisObsPlot(self):
        """Plot the fitted parameters."""
        self.currPlot = self.analysisObsPlot

        self.figure.clear()

        qIdx = self.qSlider.value()
        obsList = self.get_obsRange()

        # Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, None, True)

        # Plot the parameters of the fits
        for fileIdx, dataset in enumerate(self.dataset):
            params = {key: [] for key in dataset.params[0].keys()}
            pErrors = {key: [] for key in dataset.params[0].keys()}
            for obsIdx, obs in enumerate(obsList):
                for key, item in dataset.params[obsIdx].items():
                    if isinstance(item.value, (list, np.ndarray)):
                        params[key].append(
                            np.array(item.value).flatten()[qIdx]
                        )
                        pErrors[key].append(
                            np.array(item.error).flatten()[qIdx]
                        )
                    else:
                        params[key].append(item.value)
                        pErrors[key].append(item.error)

            for idx, key in enumerate(params.keys()):
                values = params[key]
                errors = pErrors[key]
                values = np.array(values).flatten()
                errors = np.array(errors).flatten()

                if not self.errBox.isChecked():
                    errors = np.zeros_like(errors)

                marker = "o"
                if values.size == 1:
                    values = np.zeros_like(obsList) + values
                    errors = np.zeros_like(obsList) + errors
                    marker = None

                ax[idx].plot(
                    obsList, values, marker=marker, label=dataset.name
                )

                ax[idx].fill_between(
                    obsList, values - errors, values + errors, alpha=0.4
                )
                ax[idx].set_ylabel(key)
                ax[idx].set_xlabel(dataset.observable_name)

        if self.legendBox.isChecked():
            ax[-1].legend(framealpha=0.5)

        self.canvas.draw()

    # -------------------------------------------------
    # Helper functions
    # -------------------------------------------------
    def get_qRange(self, idx=0):
        """Return the q-values used in the dataset(s).

        This assumes the q-values are the same for all datasets.

        """
        out = self.dataset[idx].q
        if isinstance(out, int):
            out = [out]
        return out

    def get_obsRange(self, idx=0):
        """Return the observables used in the dataset(s).

        This assumes the observables are the same for all datasets.

        """
        out = getattr(self.dataset[idx], self.dataset[idx].observable)
        if isinstance(out, int):
            out = [out]
        return out

    def updateLabels(self):
        """Update the labels on the right of the sliders."""
        obsIdx = self.obsSlider.value()
        qIdx = self.qSlider.value()

        self.obsVal.setText("%.2f" % self.get_obsRange()[obsIdx])
        self.qVal.setText("%.2f" % self.get_qRange()[qIdx])

    def updatePlot(self):
        """Redraw the current plot based on the selected parameters."""
        return self.currPlot()

    def initChecks(self):
        """This methods is used to perform some checks before
        finishing class initialization.

        """
        for idx, data in enumerate(self.dataset):
            if len(data._fit) == 0:
                print(
                    "No fitted model for resolution function at "
                    "index %i was found.\n"
                    "Some plotting methods are not available.\n" % idx
                )
                self.noFit = True


def plotQENS(*samples):
    """This methods plot the sample data in a PyQt5 widget allowing
    the user to show different types of plots.

    The resolution function and other parameters are automatically
    obtained from the current dataset class instance.

    Parameters
    ----------
    samples : :py:class:`nPDyn.Sample`
        Samples to be plotted.

    """
    makeWindow(QENSPlot, samples)
