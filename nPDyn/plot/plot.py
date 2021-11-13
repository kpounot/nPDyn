"""Plotting window for Sample class instances.

"""

import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
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

from qtwidgets import Toggle

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.colors import LightSource
import matplotlib
from matplotlib.cm import get_cmap

from nPDyn.plot.subPlotsFormat import subplotsFormat
from nPDyn.plot.create_window import makeWindow
from nPDyn.plot.interactive_legend import InteractiveLegend

try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass


class Plot(QWidget):
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
        self.scale_toggle = Toggle()
        toggleLayout = QHBoxLayout()
        toggleLayout.addWidget(QLabel("linear"))
        toggleLayout.addWidget(self.scale_toggle)
        toggleLayout.addWidget(QLabel("log"))
        toggleLayout.addStretch(1)
        self.scale_toggle.stateChanged.connect(self.updatePlot)

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

        slidersLayout = QVBoxLayout()

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

        slidersLayout.addItem(oLayout)
        slidersLayout.addItem(qLayout)
        slidersLayout.addItem(eLayout)

        axGroupBox = QGroupBox("Plot data along: ", self)
        axLayout = QVBoxLayout()
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

        midLayout = QHBoxLayout()
        midLayout.addWidget(axGroupBox)
        midLayout.addItem(slidersLayout)

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

        checkLayout = QHBoxLayout()
        checkLayout.addWidget(self.errBox)
        if not self.noFit:
            checkLayout.addWidget(self.fitBox)
            checkLayout.addWidget(self.compBox)
        checkLayout.addWidget(self.legendBox)
        checkLayout.addStretch()
        checkLayout.addItem(toggleLayout)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addItem(midLayout)
        layout.addItem(checkLayout)
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
        eIdx = self.eSlider.value()

        plot_errors = False
        if self.errBox.isChecked():
            plot_errors = True

        plot_legend = False
        if self.legendBox.isChecked():
            plot_legend = True

        yscale = "log" if self.scale_toggle.isChecked() else "linear"

        ymin = (
            0.1
            * np.asarray(self.dataset[0])[
                np.asarray(self.dataset[0]) > 0
            ].min()
        )
        ymax = 5 * np.asarray(self.dataset[0]).max()

        for idx, subplot in enumerate(ax):
            data = self.dataset[idx]
            obs_ax = data.axes.index(data.observable)
            q_ax = data.axes.index("q")
            e_ax = data.axes.index("energies")

            if self.oRadioButton.isChecked():
                x = getattr(data, data.observable)
                data = data.take([min((qIdx, data.shape[q_ax] - 1))], q_ax)
                data = data.take([min((eIdx, data.shape[e_ax] - 1))], e_ax)
            elif self.qRadioButton.isChecked():
                x = data.q
                data = data.take(
                    [min((obsIdx, data.shape[obs_ax] - 1))], obs_ax
                )
                data = data.take([min((eIdx, data.shape[e_ax] - 1))], e_ax)
            elif self.eRadioButton.isChecked():
                x = data.energies
                data = data.take(
                    [min((obsIdx, data.shape[obs_ax] - 1))], obs_ax
                )
                data = data.take([min((qIdx, data.shape[q_ax] - 1))], q_ax)

            data.squeeze().plot(
                subplot,
                plot_errors=plot_errors,
                plot_legend=plot_legend,
                label="experimental",
                yscale=yscale,
            )

            if not self.noFit:
                if self.fitBox.isChecked():
                    Y = data.fit_best()
                    if self.oRadioButton.isChecked():
                        Y = Y.take([min((qIdx, Y.shape[q_ax] - 1))], q_ax)
                        Y = Y.take([min((eIdx, Y.shape[e_ax] - 1))], e_ax)
                        Y = Y.squeeze()
                    elif self.qRadioButton.isChecked():
                        Y = Y.take(
                            [min((obsIdx, Y.shape[obs_ax] - 1))], obs_ax
                        )
                        Y = Y.take([min((eIdx, Y.shape[e_ax] - 1))], e_ax)
                        Y = Y.squeeze()
                    elif self.eRadioButton.isChecked():
                        Y = Y.take(
                            [min((obsIdx, Y.shape[obs_ax] - 1))], obs_ax
                        )
                        Y = Y.take([min((qIdx, Y.shape[q_ax] - 1))], q_ax)
                        Y = Y.squeeze()

                    # Plot the model
                    subplot.plot(
                        x,
                        Y,
                        label=data.model.name,
                        zorder=3,
                    )

                if self.compBox.isChecked():
                    components = data.fit_components()
                    # Plot the model components
                    for key, val in components.items():
                        Y = val
                        if self.oRadioButton.isChecked():
                            Y = Y.take([min((qIdx, Y.shape[q_ax] - 1))], q_ax)
                            Y = Y.take([min((eIdx, Y.shape[e_ax] - 1))], e_ax)
                            Y = Y.squeeze()
                        elif self.qRadioButton.isChecked():
                            Y = Y.take(
                                [min((obsIdx, Y.shape[obs_ax] - 1))], obs_ax
                            )
                            Y = Y.take([min((eIdx, Y.shape[e_ax] - 1))], e_ax)
                            Y = Y.squeeze()
                        elif self.eRadioButton.isChecked():
                            Y = Y.take([min((qIdx, Y.shape[q_ax] - 1))], q_ax)
                            Y = Y.take(
                                [min((obsIdx, Y.shape[obs_ax] - 1))], obs_ax
                            )
                            Y = Y.squeeze()

                        subplot.plot(x, Y, label=key, ls="--", zorder=2)

                if plot_legend:
                    leg = subplot.legend()
                    leg.set_draggable(True)

                tmpmin = 0.1 * np.asarray(data)[np.asarray(data) > 0].min()
                tmpmax = 5 * np.asarray(data).max()
                ymin = min((tmpmin, ymin))
                ymax = max((tmpmax, ymax))

            subplot.set_ylim(ymin, ymax)
            subplot.set_title(data.name)

        self.canvas.draw()

    def compare(self):
        """Plot the experimental data on one subplot, with or without fit"""
        self.currPlot = self.compare

        self.figure.clear()
        ax = self.figure.add_subplot()

        obsIdx = self.obsSlider.value()
        qIdx = self.qSlider.value()
        eIdx = self.eSlider.value()

        plot_errors = False
        if self.errBox.isChecked():
            plot_errors = True

        plot_legend = False
        if self.legendBox.isChecked():
            plot_legend = True

        yscale = "log" if self.scale_toggle.isChecked() else "linear"

        ymin = (
            0.1
            * np.asarray(self.dataset[0])[
                np.asarray(self.dataset[0]) > 0
            ].min()
        )
        ymax = 5 * np.asarray(self.dataset[0]).max()
        for idx, data in enumerate(self.dataset):
            obs_ax = data.axes.index(data.observable)
            q_ax = data.axes.index("q")
            e_ax = data.axes.index("energies")

            if self.oRadioButton.isChecked():
                data = data.take([min((qIdx, data.shape[q_ax] - 1))], q_ax)
                data = data.take([min((eIdx, data.shape[e_ax] - 1))], e_ax)
            elif self.qRadioButton.isChecked():
                data = data.take(
                    [min((obsIdx, data.shape[obs_ax] - 1))], obs_ax
                )
                data = data.take([min((eIdx, data.shape[e_ax] - 1))], e_ax)
            elif self.eRadioButton.isChecked():
                data = data.take(
                    [min((obsIdx, data.shape[obs_ax] - 1))], obs_ax
                )
                data = data.take([min((qIdx, data.shape[q_ax] - 1))], q_ax)

            data.squeeze().plot(
                ax,
                plot_errors=plot_errors,
                plot_legend=plot_legend,
                label=data.name,
                yscale=yscale,
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

        if self.oRadioButton.isChecked():
            plot_ax = "observable"
            plot_idx = self.obsSlider.value()
        elif self.qRadioButton.isChecked():
            plot_ax = "q"
            plot_idx = self.qSlider.value()
        elif self.eRadioButton.isChecked():
            plot_ax = "energies"
            plot_idx = self.eSlider.value()

        zscale = "log" if self.scale_toggle.isChecked() else "linear"

        ax = subplotsFormat(self, projection="3d")
        for idx, data in enumerate(self.dataset):
            data.plot_3D(
                ax[idx],
                axis=plot_ax,
                index=plot_idx,
                zscale=zscale,
            )

            if plot_ax == "observable":
                plot_ax = data.observable
            ax_idx = data.axes.index(plot_ax)
            if not self.noFit:
                data = data.swapaxes(0, ax_idx)
                x = getattr(data, data.axes[1])
                y = getattr(data, data.axes[2])
                xx, yy = np.meshgrid(y, x)
                if self.fitBox.isChecked():
                    z = data.fit_best().swapaxes(0, ax_idx)[plot_idx]
                    xx, yy = np.meshgrid(y, x)
                    z = np.log10(z) if zscale == "log" else z

                    ax[idx].plot_wireframe(
                        xx,
                        yy,
                        z,
                        rcount=min((20, x.size)),
                        ccount=min((20, y.size)),
                        colors="red",
                        label="model",
                        alpha=0.5,
                    )

                if self.compBox.isChecked():
                    components = data.fit_components()
                    # Plot the model components
                    for key, comp in components.items():
                        z = comp.swapaxes(0, ax_idx)[plot_idx]
                        z = np.log10(z) if zscale == "log" else z

                        comp_colors = get_cmap("tab20")

                        ax[idx].plot_wireframe(
                            xx,
                            yy,
                            z,
                            rcount=min((20, x.size)),
                            ccount=min((20, y.size)),
                            color=comp_colors(
                                list(components.keys()).index(key)
                            ),
                            label=key,
                            alpha=0.5,
                        )

            if self.legendBox.isChecked():
                ax[idx].set_title(data.name, fontsize=10)
                leg = ax[idx].legend()
                leg.set_draggable(True)

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
            leg = ax[-1].legend(framealpha=0.5)
            leg.set_draggable(True)

        self.canvas.draw()

    # Plot of the parameters resulting from the fit procedure
    def analysisObsPlot(self):
        """Plot the fitted parameters."""
        self.currPlot = self.analysisObsPlot

        self.figure.clear()

        qIdx = self.qSlider.value()

        # Creates as many subplots as there are parameters in the model
        ax = subplotsFormat(self, True, False, None, True)

        # Plot the parameters of the fits
        for fileIdx, dataset in enumerate(self.dataset):
            obsList = getattr(dataset, dataset.observable)
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
                ax[idx].set_xlabel(dataset.observable)

        if self.legendBox.isChecked():
            leg = ax[-1].legend(framealpha=0.5)
            leg.set_draggable(True)

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
        return np.array(out)

    def get_obsRange(self, idx=0):
        """Return the observables used in the dataset(s).

        This assumes the observables are the same for all datasets.

        """
        out = getattr(self.dataset[idx], self.dataset[idx].observable)
        if isinstance(out, int):
            out = [out]
        return np.array(out)

    def get_eRange(self, idx=0):
        """Return the energy values used in the dataset(s).

        This assumes the q-values are the same for all datasets.

        """
        out = self.dataset[idx].energies
        if isinstance(out, int):
            out = [out]
        return np.array(out)

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


def plot(*samples):
    """This methods plot the sample data in a PyQt5 widget allowing
    the user to show different types of plots.

    The resolution function and other parameters are automatically
    obtained from the current dataset class instance.

    Parameters
    ----------
    samples : :py:class:`nPDyn.Sample`
        Samples to be plotted.

    """
    makeWindow(Plot, samples)