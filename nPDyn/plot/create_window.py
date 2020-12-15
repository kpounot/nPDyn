"""Helper function to create a Qt widget from command line.

Adapted from:
https://cyrille.rossant.net/making-pyqt4-pyside-and-ipython-work-together/

"""

import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication


def makeWindow(window_class, *args):
    """Create a Qt window in Python, or interactively in IPython with Qt GUI
    event loop integration.

    """
    app = QtCore.QCoreApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    window = window_class(*args)
    window.show()

    app.exec_()

    return window
