"""Helper function to create a Qt widget from command line.

Adapted from: 
https://cyrille.rossant.net/making-pyqt4-pyside-and-ipython-work-together/

"""

import sys

from PySide2 import QtCore
from PySide2.QtWidgets import QApplication


def create_window(window_class):
    """Create a Qt window in Python, or interactively in IPython with Qt GUI
    event loop integration.
    """
    app_created = False
    app = QtCore.QCoreApplication.instance()

    if app is None:
        app = QApplication(sys.argv)
        app_created = True

    window = window_class()
    window.show()

    app.exec_()
    
    return window
