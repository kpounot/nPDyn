import os
os.environ['QT_API'] = 'pyqt5'

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtSvg 

# Import the console machinery from ipython
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.manager import QtKernelManager
import qtconsole


def get_app_qt5(*args, **kwargs):
    """Create a new qt5 app or return an existing one."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        if not args:
            args = ([''],)
        app = QtWidgets.QApplication(*args, **kwargs)
    return app


class QIPythonWidget(RichJupyterWidget):
    """ Convenience class for a live IPython console widget. 
        We can replace the standard banner using the customBanner argument"""
    def __init__(self,customBanner=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if customBanner!=None: self.banner=customBanner
        self.kernel_manager = kernel_manager = QtKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = 'qt'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            get_app_qt5().exit()            
        self.exit_requested.connect(stop)

    def pushVariables(self,variableDict):
        """ Given a dictionary containing name / value pairs, 
            push those variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)
    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()    
    def printText(self,text):
        """ Prints some plain text to the console """
        self._append_plain_text(text)        
    def executeCommand(self,command):
        """ Execute a command in the frame of the console widget """
        self._execute(command,False)

    
