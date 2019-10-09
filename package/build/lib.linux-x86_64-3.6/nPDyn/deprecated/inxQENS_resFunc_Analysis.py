import sys, os, pickle as pk
import numpy as np
import inxBinQENS
import argParser
import re
import matplotlib.pyplot as plt
from collections import namedtuple
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sympy.functions.special.delta_functions import DiracDelta
from scipy import optimize
from scipy.signal import fftconvolve
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib

class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []
        self.meanResGauWidth = []

        #_Parsing the additional parameters

        try:
            self.lowMeanBoundary = float(karg['lowMeanBoundary'])
        except ValueError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'Error: the "lowMeanBoundary" parameter can\'t be converted to an' +
                            ' float.')
            sys.exit()
        except KeyError:
            self.lowMeanBoundary = 0.6
            print(('No value for the lower boundary to compute the mean gaussian width to be' +
                            ' used was given. \n' +
                            'Using lowMeanBoundary=0.6 as default value.\n'), flush=True)

        try:
            self.lowMeanBoundary = float(karg['highMeanBoundary'])
        except ValueError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'Error: the "highMeanBoundary" parameter can\'t be converted to an' +
                            ' float.')
            sys.exit()
        except KeyError:
            self.highMeanBoundary = 1.7
            print(('No value for the higher boundary to compute the mean gaussian width to be' +
                            ' used was given. \n' +
                            'Using highMeanBoundary=1.7 as default value.\n'), flush=True)



        for dataFile in self.dataFiles:
            #_Get datas from the file and store them into self.dataList 
            inxDatas = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxDatas)    


        self.resFit()

#_Construction of the GUI

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        

        #_Add some interactive elements
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        self.analysisButton = QPushButton('Analysis')
        self.analysisButton.clicked.connect(self.analysisPlot)

        self.plot3DButton = QPushButton('3D Plot')
        self.plot3DButton.clicked.connect(self.plot3D)

        self.resButton = QPushButton('Resolution')
        self.resButton.clicked.connect(self.resPlot)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.boxLine = QFrame()
        self.boxLine.setFrameShape(QFrame.HLine)
        self.boxLine.setFrameShadow(QFrame.Sunken)

        self.label = QLabel('Q value to plot', self)
        self.lineEdit = QLineEdit(self) 
        self.lineEdit.setText('0.8')

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.boxLine)
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button)
        layout.addWidget(self.plot3DButton)
        layout.addWidget(self.analysisButton)
        layout.addWidget(self.resButton)
        self.setLayout(layout)




#_Everything needed for the fit
    def resFunc(self, x, normF, S, lorW, gauW, shift, bkgd):

        return  (normF * (S * lorW/(lorW**2 + (x-shift)**2) /np.pi 
                + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi))
                + bkgd))  
                
    def resFit(self):
    
        for i, resFile in enumerate(self.dataList):
            resList = []
            for j, datas in enumerate(resFile):
                resList.append(optimize.curve_fit(self.resFunc, 
                                datas.energies,
                                datas.intensities,
                                sigma=[val + 0.0001 for val in datas.errors],
                                #p0 = [0.5, 1, 0.8, 50, 0], 
                                bounds=([0., 0., 0., 0., -10, 0.],  
                                        [2000, 1, 5, 5, 10, 0.5]),
                                max_nfev=10000000,
                                method='trf'))
            self.resFitList.append(resList)
            self.meanResGauWidth.append(np.mean([val[0][3] for k, val in enumerate(resList) 
                                if self.lowMeanBoundary < resFile[k].qVal < self.highMeanBoundary]))

        with open(os.path.dirname(os.path.abspath(__file__)).rstrip('scripts') + 'params/resFunc_params', 
                  'wb') as resFitFile:
                    myFile = pk.Pickler(resFitFile)
                    resFitList = np.array([self.resFitList, self.meanResGauWidth])
                    myFile.dump(resFitList)
    

        
#_Definitions of the slots for the plot window
    def plot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - x))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
                    ax.errorbar(qDatas.energies, 
                           [x for x in qDatas.intensities],
                           [x for x in qDatas.errors], fmt='o')
            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.legend(['...' + dataFile[dataFile.rfind('/'):] for dataFile in self.dataFiles], 
                   loc='upper left', framealpha=0.5)
            
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot3D(self):

        plt.gcf().clear()     
        #_3D plots of the datas for each selected datafiles
        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))

        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k], projection='3d')

            for i, datas in enumerate(fileDatas):
                normFact = self.resFitList[k][i][0][0]
                GauR = self.resFitList[k][i][0][3]

                ax.plot(datas.energies, 
                            [(GauR / self.meanResGauWidth[k])**0.5 *
                             val/normFact for val in datas.intensities], 
                            datas.qVal, zdir='y', c=cmap(normColors(datas.qVal)))

            ax.set_xlabel(r'$\hslash \omega (\mu eV)$')
            ax.set_ylabel(r'$q$')
            ax.set_zlabel(r'$S_{300K}(q, \omega)$')
            ax.set_ylim((0, 2))
            ax.set_zlim((0, 1))
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):], y=1.1)
            ax.grid()

        plt.tight_layout()
        self.canvas.draw()
    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(3, 1)
        qList = [val.qVal for val in self.dataList[0]]

        for k, fileDatas in enumerate(self.dataList):
            
            lauWList = []
            gauWList = []
            bkgdList = []            

            for j, data in enumerate(fileDatas):    
                lauWList.append(self.resFitList[k][j][0][2])
                gauWList.append(self.resFitList[k][j][0][3])
                bkgdList.append(self.resFitList[k][j][0][5])

            
            #_Plot of the lorentzian width parameter of the fits
            ax1 = self.figure.add_subplot(mplGrid[0])
            ax1.plot(qList, lauWList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax1.set_ylabel(r'$Lorentzian \ Width $', fontsize=14)
            ax1.set_xlabel(r'$q$', fontsize=18)
            ax1.grid(True)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2)).get_frame().set_alpha(0.5)

            #_Plot of the gaussian width parameter of the fits
            ax2 = self.figure.add_subplot(mplGrid[1])
            ax2.errorbar(qList, gauWList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax2.set_ylabel(r'$Gaussian \ Width $', fontsize=14)
            ax2.set_xlabel(r'$q$', fontsize=18)
            ax2.grid(True)

            #_Plot of the background parameter of the fits
            ax3 = self.figure.add_subplot(mplGrid[2])
            ax3.errorbar(qList, bkgdList, marker='o', 
                    label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax3.set_ylabel(r'$background $', fontsize=14)
            ax3.set_xlabel(r'$q$', fontsize=18)
            ax3.grid(True)


        plt.tight_layout()
        self.canvas.draw()

    def resPlot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - x))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
    
                    shift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][0]
                    S = self.resFitList[k][j][0][1]
                    gauW = self.resFitList[k][j][0][3]
                    lorW = self.resFitList[k][j][0][2]
                    bkgd = self.resFitList[k][j][0][5]

                    ax.plot(qDatas.energies, [val/normFact for val in qDatas.intensities])
    
                    ax.plot(qDatas.energies, 
                                [self.resFunc(val, 1, S, lorW, gauW, shift, bkgd)
                                             for val in qDatas.energies])
                    ax.plot(qDatas.energies, 
                                [self.resFunc(val, 1, S, lorW, gauW, shift, 0)
                                             for val in qDatas.energies])


            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylim(1e-3, 1.5)
            ax.set_ylabel(r'$S_{10K}(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.legend(['...' + dataFile[dataFile.rfind('/'):] for dataFile in self.dataFiles], 
                   loc='upper left', framealpha=0.5)
            
        
        ax.grid()
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)
    subW = Window() 
    subW.show()

    sys.exit(app.exec_())
