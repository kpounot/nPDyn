"""

Classes
^^^^^^^

"""

import numpy as np

from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QFrame)

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg
                                                as FigureCanvas)
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT
                                                as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib


matplotlib.use('Qt5Agg')


class plotMSDSeries(QWidget):
    """ This class created a PyQt widget containing a matplotlib
        canvas to draw the plots.

        :arg msdSeries:   list of mean-squared displacements (MSD)
                          series obtained from MD simulations
        :arg tempList:    temperature list for MSD from simulation
        :arg datasetList: indices of experimental MSD to be compared
                          with msdSeries

    """


    def __init__(self, msdSeriesList, tempList, datasetList=[]):

        super().__init__()

        self.tempList   = np.array(tempList)
        self.dataset    = datasetList
        self.msdSeries  = msdSeriesList

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
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.label = QLabel('Max temperature for plotting', self)
        self.lineEdit = QLineEdit(self)
        self.lineEdit.setText('300')

        self.msdButton = QPushButton('Replot')
        self.msdButton.clicked.connect(self.MSD)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.msdButton)
        self.setLayout(layout)


        # Calling method to plot MSD
        self.MSD()


    # -------------------------------------------------
    # Definitions of the slots for the plot window
    # -------------------------------------------------

    def MSD(self):

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Plot the mean-squared displacement as a function of temperature
        # for each file
        for dataset in self.dataset:
            # Obtaining the temperature to plot as being the closest
            # one to the number entered by the user
            tempIdx = int(np.argwhere(
                dataset.data.X <= float(self.lineEdit.text()))[-1])

            # Extracting the MSD from parameters for each temperature
            msdList = [dataset.params[idx][0][1]
                       for idx, temp in enumerate(dataset.data.X)
                       if idx <= tempIdx]

            # Computing the errors for each temperature from covariant matrix
            errList = [np.sqrt(np.diag(dataset.params[tempIdx][1]))[1]
                       for tempIdx, temp in enumerate(dataset.data.X)]

            # Plotting the experimental MSD
            ax.errorbar(dataset.data.X[:tempIdx],
                        msdList[:tempIdx],
                        errList[:tempIdx],
                        label = dataset.fileName)

        for i in range(len(self.msdSeries)):

            tempIdx = int(np.argwhere(
                self.tempList <= float(self.lineEdit.text()))[-1])

            # Plotting the MSD from simulation
            ax.errorbar(self.tempList, self.msdSeries[i][:tempIdx, 0],
                        self.msdSeries[i][:tempIdx, 1], label="Simulated MSD")

            ax.set_xlabel(r'$Temperature (K)$')
            ax.set_ylabel(r'$MSD \ (\AA^{2})$')
            ax.legend(framealpha=0.5, fontsize=12, loc='upper left')
            ax.grid()

        self.canvas.draw()


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
                      "Please use a fitting method before plotting.\n" % idx)
