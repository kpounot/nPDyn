import sys, os
import re
import scripts.argParser as argParser
import pickle as pk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QToolTip, QTextEdit, QMessageBox, QAction, qApp,
                             QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QFileSystemModel, QTreeView, QLayout, QComboBox,
                             QFileDialog, QMessageBox, QGroupBox, QLineEdit,
                             QAbstractItemView, QFrame, QSplitter, QFrame, QCheckBox)
from PyQt5.QtGui import QFont, QIcon, QMouseEvent, QCursor, QPixmap, QWindow
from PyQt5.QtCore import (QCoreApplication, QDir, QVariant, QFile, QTextStream, 
                          QFileDevice, Qt, QFileInfo, QProcess, QByteArray)
from PyQt5 import QtSvg
import JupyterWidget
from IPython.utils import io
import subprocess

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        with open(os.path.dirname(os.path.abspath(__file__)) + '/config', 'rb') as configFile:
            self.config = pk.Unpickler(configFile).load()
            self.config['softPath'] += '/'
            self.config['treePath'] += '/'
            self.config['scriptsPath'] += '/'

    #_Fonts
        QToolTip.setFont(QFont('SansSerif', 10))
        self.boldFont = QFont('SansSerif', 10, QFont.Bold)

    #_Gui elements
        self.textEdit = QTextEdit()
        self.textEdit.setCurrentFont(QFont('Lucida Console', 10))
        self.textEdit.setStatusTip('Print area for files')

        self.btnShowContent = QPushButton('Show Content')
        self.btnShowContent.setStatusTip('Show the content of the selected file')

        self.btnLauchScript = QPushButton('Run script')         
        self.btnLauchScript.setToolTip('Run the selected script')

        self.btnOutput = QPushButton('Output') 
        self.btnOutput.setStatusTip('Show the content of the output.txt file')

        self.scriptList = QComboBox()
        
        self.btnSave = QPushButton('Save')
        self.btnSave.setStatusTip('Save the content of the text area into a file')

        self.btnSelToConsole = QPushButton('Sel To Console')
        self.btnSelToConsole.setStatusTip('''Put the selected files and parameters 
                                             to the IPython console''')

        self.pyTerm = JupyterWidget.QIPythonWidget()

        #_groupBox for parameters
        self.paramBox = QGroupBox('Analysis parameters')

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.boxLine2 = QFrame()
        self.boxLine2.setFrameShape(QFrame.HLine)
        self.boxLine2.setFrameShadow(QFrame.Sunken)

        self.normFactLabel = QLabel('Normalization Factor : ', parent = self.paramBox)
        self.normFactLabel.setStatusTip('Number of bins used for normalization'+
                                        ' starting from the lowest temperature')
        self.normFactText = QLineEdit(parent=self.paramBox)
        self.normFactText.setText('2')
        self.normFactText.setStatusTip('Number of bins used for normalization'+
                                       ' starting from the lowest temperature')
        self.binLabel = QLabel('Bin Step : ', parent = self.paramBox)
        self.binText = QLineEdit(parent=self.paramBox)
        self.binText.setText('300')

        self.qMinLabel = QLabel('q-range min: ', parent = self.paramBox)
        self.qMinLabel.setStatusTip('Minimum of the q-range used for the fit')
        self.qMinText = QLineEdit(parent = self.paramBox)
        self.qMinText.setText('0.3')
        self.qMinText.setStatusTip('Minimum of the q-range used for the fit')

        self.qMaxLabel = QLabel('q-range max: ', parent = self.paramBox)
        self.qMaxLabel.setStatusTip('Maximum of the q-range used for the fit')
        self.qMaxText = QLineEdit(parent = self.paramBox)
        self.qMaxText.setText('1.2')
        self.qMaxText.setStatusTip('Maximum of the q-range used for the fit')

        self.qDiscardLabel = QLabel('q val to discard : ', parent=self.paramBox)
        self.qDiscardText = QLineEdit(parent=self.paramBox)
        
        self.newFitLabel = QLabel('New fit  ', parent=self.paramBox)
        self.new_fit_check = QCheckBox(parent=self.paramBox)
        self.new_fit_check.setCheckState(Qt.Checked)
        self.new_fit_check.setStatusTip('Use the fitting procedure on the data or load existing parameters')

        self.elasticNormLabel = QLabel('Elastic norm factor ', parent=self.paramBox)
        self.elasticNorm = QCheckBox(parent=self.paramBox)
        self.elasticNorm.setStatusTip('Use the elastic data at low temperature for normalization')

        self.vanaNormLabel = QLabel('Vanadium norm (elastic) ', parent=self.paramBox)
        self.vanaNorm = QCheckBox(parent=self.paramBox)
        self.vanaNorm.setStatusTip('Use the vanadium data for normalizing the elastic ones')

        #_Additional parameters widgets
        self.addParamLabel = QLabel('Additional parameters :')
        self.addParamText = QTextEdit()
        self.addParamText.setStatusTip('Put the list members as : parameter_name=value')

    #_Actions
        exitAction = QAction(QIcon(), '&Exit', self)
        exitAction.setShortcut('ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        saveAction = QAction(QIcon(), '&Save', self)
        saveAction.setShortcut('ctrl+S')
        saveAction.setStatusTip('Save the content of the text area')
        saveAction.triggered.connect(self.saveFile)

        aboutAction = QAction(QIcon(), '&About', self)
        aboutAction.setStatusTip('About the application')
        aboutAction.triggered.connect(self.aboutDialog)

        launchScriptAction = QAction(QIcon(), '&Run script', self)
        launchScriptAction.setShortcut('ctrl+L')
        launchScriptAction.triggered.connect(self.scriptLaunch)  

        selToConsoleAction = QAction(QIcon(), '&Sel To Console', self)
        selToConsoleAction.setShortcut('alt+S')
        selToConsoleAction.triggered.connect(self.selToConsole)  

        outputAction = QAction(QIcon(), 'Output', self)
        outputAction.setShortcut('ctrl+alt+A')
        outputAction.setStatusTip('Show the content of the output.txt file')
        outputAction.triggered.connect(self.showOutput)

        showAction = QAction(QIcon(), 'Show content', self)
        showAction.setShortcut('alt+P')
        showAction.setStatusTip('Show the content of the selected file')
        showAction.triggered.connect(self.showContent)

        configAction = QAction(QIcon(), 'Configuration', self)
        configAction.setShortcut('alt+C')
        configAction.setStatusTip('Set the folder paths up')
        configAction.triggered.connect(self.configDialog)

    #_Menus
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        editMenu = menuBar.addMenu('&Edit')
        aboutMenu = menuBar.addMenu('&About')

        fileMenu.addAction(exitAction)
        fileMenu.addAction(saveAction)

        editMenu.addAction(launchScriptAction)
        editMenu.addAction(outputAction)
        editMenu.addAction(showAction)
        editMenu.addAction(selToConsoleAction)
        editMenu.addAction(configAction)

        aboutMenu.addAction(aboutAction)

        
    #_Status
        self.statusBar().showMessage('ready')
       
    #_TreeView Widget explore system files
        self.sysFiles = QFileSystemModel()
        self.sysFiles.setRootPath(os.getcwd())
        self.fileTree = QTreeView()
        self.fileTree.setModel(self.sysFiles)
        self.fileTree.setColumnHidden(1, True)
        self.fileTree.setColumnHidden(2, True)
        self.fileTree.setColumnHidden(3, True)
        self.fileTree.setTextElideMode(Qt.ElideNone)
        self.fileTree.header().setStretchLastSection(True)
        self.fileTree.setAutoScroll(False)
        self.fileTree.setRootIndex(self.sysFiles.index(self.config['treePath']))
        self.fileTree.expand(self.sysFiles.index(self.config['treePath']))
        self.fileTree.setSelectionMode(QAbstractItemView.MultiSelection)
        self.fileTree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

    #_ScriptList QComboBox
        path = QDir(self.config['scriptsPath'])
        fileList = path.entryList(QDir.Files)
        self.scriptList.addItems(fileList)

    #_Signals to slots
        self.btnShowContent.clicked.connect(self.showContent)
        self.fileTree.expanded.connect(self.resizeColumnToContents)
        self.fileTree.collapsed.connect(self.resizeColumnToContents)
        self.fileTree.selectionModel().selectionChanged.connect(self.treeStatus)
        self.fileTree.doubleClicked.connect(self.showContent)
        self.btnLauchScript.clicked.connect(self.scriptLaunch)
        self.btnSave.clicked.connect(self.saveFile)
        self.btnOutput.clicked.connect(self.showOutput)
        self.btnSelToConsole.clicked.connect(self.selToConsole)

    #_Layout
        #_Layout for the main window
        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
       
        centralLayout = QHBoxLayout()
        centralPanel = QWidget()
        rightPanel = QWidget()
        rightPanelSpace = QWidget()

        centralVBox = QVBoxLayout()
        centralVBox.addWidget(self.scriptList)
        centralVBox.addWidget(self.btnLauchScript)
        centralVBox.addWidget(self.textEdit)
        centralPanel.setLayout(centralVBox)
    
        #_Layout for the right panel
        rightVBox = QVBoxLayout() 
        rightVBox.addWidget(self.btnShowContent)
        rightVBox.addWidget(self.btnOutput)
        rightVBox.addWidget(self.btnSave)
        rightVBox.addWidget(self.btnSelToConsole)
        rightVBox.addWidget(self.boxLine)
        rightVBox.addWidget(self.paramBox)
        rightVBox.addWidget(self.boxLine2)
        rightVBox.addWidget(self.addParamLabel)
        rightVBox.addWidget(self.addParamText)
        rightVBox.addStretch(3)
        rightPanel.setLayout(rightVBox)

        hSplitter1 = QSplitter(Qt.Horizontal)
        hSplitter1.addWidget(self.fileTree)
        hSplitter1.addWidget(centralPanel)
        hSplitter1.setStretchFactor(1, 4)

        hSplitter2 = QSplitter(Qt.Horizontal)
        hSplitter2.addWidget(hSplitter1)
        hSplitter2.addWidget(rightPanel)
        hSplitter2.setStretchFactor(0, 4)
        hSplitter2.setStretchFactor(1, 1)

        vSplitter = QSplitter(Qt.Vertical)
        vSplitter.addWidget(hSplitter2)
        vSplitter.addWidget(self.pyTerm)
        vSplitter.setStretchFactor(0, 3)
        vSplitter.setStretchFactor(1, 2)

        centralLayout.addWidget(vSplitter)
        mainWidget.setLayout(centralLayout)

        
        #_Layout for the groupbox containing the parameters
        paramGrid = QGridLayout()

        self.paramBox.setLayout(paramGrid)
        paramGrid.addWidget(self.normFactLabel, 0, 0)
        paramGrid.addWidget(self.normFactText, 0, 1)
        paramGrid.addWidget(self.binLabel, 1, 0)
        paramGrid.addWidget(self.binText, 1, 1)

        paramGrid.addWidget(self.qMinLabel, 2, 0)
        paramGrid.addWidget(self.qMinText, 2, 1)
        paramGrid.addWidget(self.qMaxLabel, 3, 0)
        paramGrid.addWidget(self.qMaxText, 3, 1)

        paramGrid.addWidget(self.qDiscardLabel, 4, 0)
        paramGrid.addWidget(self.qDiscardText, 4, 1)

        paramGrid.addWidget(self.newFitLabel, 5, 0)
        paramGrid.addWidget(self.new_fit_check, 5, 1)

        paramGrid.addWidget(self.elasticNormLabel, 6, 0)
        paramGrid.addWidget(self.elasticNorm, 6, 1)

        paramGrid.addWidget(self.vanaNormLabel, 7, 0)
        paramGrid.addWidget(self.vanaNorm, 7, 1)

    #_Miscellanous
        self.setWindowTitle('nPDyn')
        self.setGeometry(20, 20, 1200, 800)
        self.setMouseTracking(True)
        self.setWindowIcon(QIcon(self.config['softPath']+'Icons/Neutron_quarks_structure.jpg'))

        self.show()

#_Events handlers
    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Exit', 'Really exit ?', QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


#_Events sender

#_Slots
    def showContent(self):
        index = self.fileTree.selectionModel().currentIndex()
        filePath = self.sysFiles.filePath(index)
        with open(filePath, 'r') as fileToShow:
            text = fileToShow.read()
            self.textEdit.setText(text)

    def resizeColumnToContents(self):
        self.fileTree.resizeColumnToContents(0)

    def treeStatus(self):
        index = self.fileTree.selectionModel().currentIndex()
        filePath = self.sysFiles.filePath(index)
        self.fileTree.setStatusTip(filePath)
        self.statusBar().showMessage(filePath)

    def scriptLaunch(self):
        filesPath = [self.sysFiles.filePath(index) for index in self.fileTree.selectedIndexes()]
        scriptToLaunch = self.config['scriptsPath'] + self.scriptList.currentText()

        addParamPattern = re.compile(r'[\n ,;-]+')
        addParamList = addParamPattern.split(self.addParamText.toPlainText())
        arg, karg = argParser.argParser(addParamList)

        r = 'ipython3 ' + scriptToLaunch + ' ' + ' '.join(filesPath) + ' ' 
        if karg is not None:
            for item in karg.items():
                r += item[0].strip() + '=' + item[1].strip() + ' '
        r += 'new_fit=' + str(self.new_fit_check.isChecked()) + ' '
        r += 'elasticNormF=' + str(self.elasticNorm.isChecked()) + ' '
        r += 'vanaNorm=' + str(self.vanaNorm.isChecked()) + ' '
        r += 'normFactor=' + self.normFactText.text() + ' '
        r += 'qDiscard=' + self.qDiscardText.text() + ' '
        r += 'binS=' + self.binText.text() + ' '
        r += 'qMin=' + self.qMinText.text() + ' ' 
        r += 'qMax=' + self.qMaxText.text() + ' > ' + self.config['softPath'] + 'output.txt &' 

        #_Save the content of the last output into the file output_old.txt
        with open(self.config['softPath'] + 'output.txt', 'r') as fileToShow:
            oldFile = open(self.config['softPath'] + 'output_old.txt', 'w')
            oldFile.write(fileToShow.read())
            oldFile.close()

        proc = subprocess.Popen(r, shell=True)
        print('Using command : ' + r + '\n')


    def saveFile(self):
        fileName = QFileDialog.getSaveFileName()
        with open(fileName[0], 'w') as fileName:
            fileName.write(self.textEdit.toPlainText())    
            
    def aboutDialog(self):
        text = 'nPDyn \n\n'
        text += 'nPDyn is a simple GUI designed to make the analysis of \n' 
        text += 'neutron scattering data easier. \n'
        text += 'Just select your files in the left filetree, select the script you \n'
        text += 'want to use on this file and launch it. \n\n'
        text += 'The application uses the sys.argv as argument to pass to the \n'
        text += 'script and argParser.py. \n'
        text += 'So in your script, just get the files paths list by typing : \n'
        text += 'arg, karg = argParser.argParser(sys.argv)\n ' 
        text += 'filesPaths = arg[1:] \n\n'
        text += 'Other options can be given to the script : \n'
        text += 'Bin step, q-range, ...\n'
        text += 'These parameters can be get in your script using :\n'
        text += 'myParameter = karg[optionName], which is a dictionnary :' + ' \n'
        text += 'normFactor >>> number of low-temp bins used for normalization (MSD scripts). \n' 
        text += 'binS >>> bin step, number of lines used for averaging \n' 
        text += 'qMin >>> q-range min \n' 
        text += 'qMax >>> q-range max \n'
        text += 'qDiscard >>> q value to discard in the analysis \n\n' 
        text += 'Additional parameters can also be provided by the user.\n'
        text += 'For this, just write the parameters names each followed by "=" and the value.\n'
        text += 'The program automatically detect separators and line return and finally add'
        text += 'the provided parameters to the console command just as the standard ones.\n\n'
        text += 'All the print() calls in your scripts are saved in output.txt. \n'
        text += 'This file is shown in the central text edit box after the run \n'
        text += 'and you can save it using the save buttons/shortcut.' 
        aboutDialog = QMessageBox.about(self, 'About', text)

    def showOutput(self):

        with open(self.config['softPath'] + 'output.txt', 'r') as outputTxt:
            self.textEdit.setText(outputTxt.read())
        self.textEdit.verticalScrollBar().setValue(self.textEdit.verticalScrollBar().maximum())

    def selToConsole(self):
        
        filesPath = [self.sysFiles.filePath(index) for index in self.fileTree.selectedIndexes()]
        scriptToLaunch = self.config['scriptsPath'] + self.scriptList.currentText()

        addParamPattern = re.compile(r'[\n ,.:;-]+')
        addParamList = addParamPattern.split(self.addParamText.toPlainText())
        arg, karg = argParser.argParser(addParamList)

        r = ' '.join(filesPath) + ' ' 
        if karg is not None:
            for item in karg.items():
                r += item[0].strip() + '=' + item[1].strip() + ' '
        r += 'new_fit=' + str(self.new_fit_check.isChecked()) + ' '
        r += 'elasticNormF=' + str(self.elasticNorm.isChecked()) + ' '
        r += 'vanaNorm=' + str(self.vanaNorm.isChecked()) + ' '
        r += 'normFactor=' + self.normFactText.text() + ' '
        r += 'qDiscard=' + self.qDiscardText.text() + ' '
        r += 'binS=' + self.binText.text() + ' '
        r += 'qMin=' + self.qMinText.text() + ' ' 
        r += 'qMax=' + self.qMaxText.text()  
       
        self.pyTerm.printText(r) 

    def configDialog(self):

        self.configW = configWindow()
        self.configW.show()
        self.configW.destroyed.connect(self.updateGUI)


    def updateGUI(self):
        with open(os.path.dirname(os.path.abspath(__file__)) + '/config', 'rb') as configFile:
            self.config = pk.Unpickler(configFile).load()
            self.config['softPath'] += '/'
            self.config['treePath'] += '/'
            self.config['scriptsPath'] += '/'

        self.scriptList.clear()
        path = QDir(self.config['scriptsPath'])
        fileList = path.entryList(QDir.Files)
        self.scriptList.addItems(fileList)

        self.fileTree.setRootIndex(self.sysFiles.index(self.config['treePath']))
        self.fileTree.expand(self.sysFiles.index(self.config['treePath']))



#_Configuration window used by the user to define the paths needed by the program
class configWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.initUI()

    def initUI(self):

        with open(os.path.dirname(os.path.abspath(__file__)) + '/config', 'rb') as configFile:
            self.configDict = pk.Unpickler(configFile).load()

    #_GUI Objects
        self.softPathLabel = QLabel('Installation Dir: ')
        self.softPathText = QLineEdit()
        self.softPathText.setText(self.configDict['softPath'])
        self.browseButton01 = QPushButton('Browse...')

        self.scriptsPathLabel = QLabel('Scripts Dir: ')
        self.scriptsPathText = QLineEdit()
        self.scriptsPathText.setText(self.configDict['scriptsPath'])
        self.browseButton02 = QPushButton('Browse...')

        self.treePathLabel = QLabel('Data Dir: ')
        self.treePathText = QLineEdit()
        self.treePathText.setText(self.configDict['treePath'])
        self.browseButton03 = QPushButton('Browse...')

        self.confirmButton = QPushButton('Apply')
        self.confirmButton.setDefault(True)
        self.confirmButton.setAutoDefault(True)
        self.cancelButton = QPushButton('Cancel')
        self.cancelButton.setAutoDefault(True)

    #_Layout
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        myLayout = QGridLayout()
        myLayout.addWidget(self.softPathLabel, 0, 0, 1, 1)
        myLayout.addWidget(self.softPathText, 0, 1, 1, 2)
        myLayout.addWidget(self.browseButton01, 0, 3, 1, 1)

        myLayout.addWidget(self.scriptsPathLabel, 1, 0, 1, 1)
        myLayout.addWidget(self.scriptsPathText, 1, 1, 1, 2)
        myLayout.addWidget(self.browseButton02, 1, 3, 1, 1)

        myLayout.addWidget(self.treePathLabel, 2, 0, 1, 1)
        myLayout.addWidget(self.treePathText, 2, 1, 1, 2)
        myLayout.addWidget(self.browseButton03, 2, 3, 1, 1)

        myLayout.addWidget(self.confirmButton, 3, 0, 1, 2)
        myLayout.addWidget(self.cancelButton, 3, 2, 1, 2)

        myLayout.setColumnStretch(1, 2)

        self.mainWidget.setLayout(myLayout)

    #_Misc 
        self.setWindowTitle('nPDyn configuration')
        self.setGeometry(40, 40, 800, 60)
        self.show()

    #_Signals to slots
        self.browseButton01.clicked.connect(self.dirBrowser)
        self.browseButton02.clicked.connect(self.dirBrowser)
        self.browseButton03.clicked.connect(self.dirBrowser)
        
        self.confirmButton.pressed.connect(self.confirmConf)
        self.cancelButton.pressed.connect(self.cancelConf)

    #_Slots
    def dirBrowser(self):
        dirPath = QFileDialog().getExistingDirectory()
    
        button = self.sender()
        buttonIndex = self.mainWidget.layout().indexOf(button)
        lineIndex = self.mainWidget.layout().getItemPosition(buttonIndex)[0]

        self.mainWidget.layout().itemAtPosition(lineIndex, 1).widget().setText(dirPath)

    def confirmConf(self):
        myDict = {}
        myDict['softPath'] = self.softPathText.text().strip()
        myDict['scriptsPath'] = self.scriptsPathText.text().strip()
        myDict['treePath'] = self.treePathText.text().strip()

        for entry, confPath in myDict.items():
            if confPath[-1] == '/':
                myDict[entry] = confPath[:-1]

        with open(os.path.dirname(os.path.abspath(__file__)) + '/config', 'wb') as configFile:
            myFile = pk.Pickler(configFile)
            myFile.dump(myDict)
    
        self.close()

    def cancelConf(self):
        self.close()


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
