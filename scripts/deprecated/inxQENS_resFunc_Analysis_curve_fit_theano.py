import sys, os
import numpy as np
import inxBinQENS
import argParser
import matplotlib.pyplot as plt
from collections import namedtuple
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import optimize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib
import pickle as pk

#_Import the resolution function and the associated shared variables
import theano_functions as tf
resFunc, resGrad = tf.resFunc_pseudo_voigt_curve_fit()

''' PyQt window which is shown at the end of the fitting procedure.
    It can plot the raw data, the fits and the fitted parameters for each q-values. '''
class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []


        for dataFile in self.dataFiles:
            #_Get datas from the file and store them into self.dataList 
            inxDatas = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxDatas)    

        #_Keep the datas from wanted q-range
        qMin = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMin']) - x))
        qMax = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMax']) - x))
        for i, fileDatas in enumerate(self.dataFiles):
            self.dataList[i] = [val for val in self.dataList[i] if qMin <= val.qVal <= qMax]


        #_Discard the selected index
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])
            qDiscardList = [min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x: abs(float(val) - x)) for val in qDiscardList]
            for i, fileDatas in enumerate(self.dataFiles):
                self.dataList[i] = [val for val in self.dataList[i] 
                                    if val.qVal not in qDiscardList] 

        

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
    def resFitFunc(self, x, normF, S, lorW, gauW, shift, bkgd):

        model = resFunc([normF, S, lorW, gauW, shift, bkgd], x).astype('float64') 
        return model

    def resFitGrad(self, x, normF, S, lorW, gauW, shift, bkgd):

        model = resGrad([normF, S, lorW, gauW, shift, bkgd], x).astype('float64') 
        return model


    def resFit(self):
    
        x0 = np.array([200, 0.3, 0.3, 0.4, 0., 0.01])
        bounds = ([0., 0., 0., 0., -5, 0.], [2000, 1, 1, 1, 5, 0.5])

        for i, resFile in enumerate(self.dataList):
            resList = []
            for j, data in enumerate(resFile):
                resList.append(optimize.curve_fit(self.resFitFunc,
                                                 data.energies,
                                                 data.intensities,
                                                 sigma=data.errors,
                                                 #p0=x0,
                                                 bounds=bounds,
                                                 jac=self.resFitGrad,
                                                 max_nfev=10000000,
                                                 method='trf'))
                
                print('\n> Final result for qVal = %s: ' % data.qVal, flush=True)
                print('Fit normF : ' + str(resList[j][0][0]), flush=True)
                print('Fit S     : ' + str(resList[j][0][1]), flush=True)
                print('Fit lorW  : ' + str(resList[j][0][2]), flush=True)
                print('Fit gauW  : ' + str(resList[j][0][3]), flush=True)
                print('Fit shift : ' + str(resList[j][0][4]), flush=True)
                print('Fit bkgd  : ' + str(resList[j][0][5]) + '\n', flush=True)

            self.resFitList.append(resList)

        with open(os.path.dirname(os.path.abspath(__file__)) + '/resFunc_params', 'wb') as fittedParamsFile:
            myFile = pk.Pickler(fittedParamsFile)
            resFitList = self.resFitList
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
                           [x/self.resFitList[k][j][0][0] for x in qDatas.intensities],
                           [x/self.resFitList[k][j][0][0] for x in qDatas.errors], fmt='o')
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

                ax.plot(datas.energies, 
                            [val/normFact for val in datas.intensities], 
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
    
                    normF = self.resFitList[k][j][0][0]
                    result = resFunc(self.resFitList[k][j][0][:], qDatas.energies)

                    ax.plot(qDatas.energies, [val/normF for val in qDatas.intensities])
    
                    ax.plot(qDatas.energies, result/normF)


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
