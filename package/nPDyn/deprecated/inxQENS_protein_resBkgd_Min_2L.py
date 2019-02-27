import sys, os
import pickle as pk
import numpy as np
import inxBinQENS
import argParser
import re
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import optimize
from scipy.signal import fftconvolve
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib
import inxQENS_extract_resFunc_analysis as resFuncAnalysis
import inxENS_extract_normF as elasticNormF


class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resList = []
        self.resFitList = []
        self.scatFitList1 = []
        self.meanResGauWidth = []
        self.normF = []
        

        #_Get datas from the file and store them into self.dataList 
        for i, dataFile in enumerate(self.dataFiles):
            inxDatas = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxDatas)    

            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file to be used for resolution function fitting for file:\n'
                    + '...' + dataFile[dataFile.rfind('/'):]) 
            resFile = QFileDialog().getOpenFileName()[0]
            resData = inxBinQENS.inxBin(resFile, karg['binS'])
            self.resFitList.append(resFuncAnalysis.resFit(resData))


        #_Discard the selected index
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,.:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])
            for j, fileDatas in enumerate(self.dataFiles):
                for val in qDiscardList:
                    self.dataList[j].pop(int(val))
                    self.resFitList[j].pop(int(val))
        

        #_Keep the datas from wanted q-range
        qMin = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMin']) - x))
        qMax = min([datas.qVal for datas in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMax']) - x))
        for i, fileDatas in enumerate(self.dataFiles):
            self.resFitList[i] = [self.resFitList[i][qIdx] for qIdx, val in enumerate(self.dataList[i]) 
                                                                    if qMin <= val.qVal <= qMax]
            self.dataList[i] = [val for val in self.dataList[i] if qMin <= val.qVal <= qMax]


        #_Get values for normalization
        for i, dataFile in enumerate(self.dataFiles):
            if karg['elasticNormF']=='True':
                self.normF.append(elasticNormF.get_elastic_normF())
            else:
                self.normF.append([val[0][0] for val in self.resFitList[i]])

        if karg['new_fit']=='True':
            #_Get the file in which the fitted parameters are to be saved
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file in which to save the fitted parameters...') 
            self.paramsFile = QFileDialog().getSaveFileName()[0]
            self.scatFit()

        else:
            #_Get the file containing the fitted parameters
            message = QMessageBox.information(QWidget(), 'File selection',
                    'Please select the file containing the fitted parameters...') 
            paramsFile = QFileDialog().getOpenFileName()[0]
            with open(paramsFile, 'rb') as params:
                self.scatFitList1 = pk.Unpickler(params).load()


#_Construction of the GUI

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        

        #_Add some interactive elements
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        self.fitButton = QPushButton('Fit')
        self.fitButton.clicked.connect(self.fitPlot)

        self.analysisButton = QPushButton('Analysis')
        self.analysisButton.clicked.connect(self.analysisPlot)

        self.plot3DButton = QPushButton('3D Plot')
        self.plot3DButton.clicked.connect(self.plot3D)

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
        layout.addWidget(self.fitButton)
        layout.addWidget(self.analysisButton)
        self.setLayout(layout)




#_Everything needed for the fit
    def fitFunc(self, x, fileData, i):

        cost = 0

        gList   = x[0:2] 
        gList   = gList.reshape(gList.shape[0], 1)
        s0      = x[2]
        sList   = x[3:5]
        sList   = sList.reshape(sList.shape[0], 1)
        msd     = x[5]

        for j, data in enumerate(fileData):

            shift = self.resFitList[i][j][0][4]
            normF = self.resFitList[i][j][0][0]
            S     = self.resFitList[i][j][0][1]
            gauW  = self.resFitList[i][j][0][3]
            lorW  = self.resFitList[i][j][0][2]
            bkgd  = self.resFitList[i][j][0][5]

            
            X = data.energies
            
            #_Resolution function
            f_res = ((S * lorW / (lorW**2 + (X - shift)**2) / np.pi 
                    + (1-S) * np.exp(-(X - shift)**2 / (2*gauW**2)) / (gauW * np.sqrt(2*np.pi)) 
                    + bkgd))

            #_Lorentzians
            f_lor = gList*data.qVal**2 / (np.pi * (X**2 + (gList*data.qVal**2)**2 ))

            convolutions = np.array([np.convolve(val, f_res, mode='same') for val in f_lor])

            f = np.exp(-data.qVal**2*msd/3) * (s0 * f_res + np.sum(sList * convolutions, axis=0))

            cost += np.sum((data.intensities / self.normF[i][j] - f)**2 / (data.errors**2))


        return cost



    def scatFit(self):
    
        scatList1 = []
        for i, fileData in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            
            #_Minimization 
            scatList1.append(optimize.minimize(lambda x:
                    self.fitFunc(x, fileData, i),
                    [2.5, 80, 0.8, 0.1, 0.1, 1], 
                    bounds = [(0., 20), (0., 100), (0., 1), (0., 1), (0., 1), (0., 5.)],
                    options={'eps':1e-10, 'maxcor':500, 'maxls':150, 
                             'maxfun':100000, 'maxiter':100000},
                    callback=self.fitState))

            print('> Final result : ', flush=True)
            print('Fit g0 : ' + str(scatList1[i].x[0]), flush=True)
            print('Fit g1 : ' + str(scatList1[i].x[1]), flush=True)
            print('Fit s0 : ' + str(scatList1[i].x[2]), flush=True)
            print('Fit s1 : ' + str(scatList1[i].x[3]), flush=True)
            print('Fit s2 : ' + str(scatList1[i].x[4]), flush=True)
            print('Fit msd : ' + str(scatList1[i].x[5]), flush=True)


            self.scatFitList1 = scatList1

        with open(self.paramsFile, 'wb') as fittedParamsFile:
            myFile = pk.Pickler(fittedParamsFile)
            modelFitList = self.scatFitList1
            myFile.dump(modelFitList)


    #_Callback function for the algorithm
    def fitState(self, x):
        print('Step results :', flush=True)
        print('        {0:<50}= {1:.4f}'.format('g0', x[0]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('g1', x[1]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('s0', x[2]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('s1', x[3]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('s2', x[4]), flush=True)
        print('        {0:<50}= {1:.4f}'.format('msd', x[5]), flush=True)

    #_Function used to produce the plot of the fit
    def fittedFunc(self, data, shift, normFact, S, gauW, lorW, bkgd, j, x):

        s0         = x[0]
        gList      = x[1:3]
        gList      = gList.reshape(gList.shape[0], 1)
        sList      = x[3:4]
        sList      = sList.reshape(sList.shape[0], 1)
        msd        = x[5]

        X = data.energies
        
        #_Resolution function
        f_res = ((S * lorW / (lorW**2 + (X - shift)**2) / np.pi 
                + (1-S) * np.exp(-(X - shift)**2 / (2*gauW**2)) / (gauW * np.sqrt(2*np.pi)) 
                + bkgd))

        #_Lorentzians
        f_lor = gList*data.qVal**2 / (np.pi * (X**2 + (gList*data.qVal**2)**2 ))

        convolutions = np.array([np.convolve(val, f_res, mode='same') for val in f_lor])

        f = np.exp(-data.qVal**2*msd/3) * (s0 * f_res + np.sum(sList * convolutions, axis=0))

        return f


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
                           [x/self.resFitList[k][j][0][3] for x in qDatas.intensities],
                           [x/self.resFitList[k][j][0][3] for x in qDatas.errors], fmt='o')
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
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()

        self.canvas.draw()
    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        mplGrid = gridspec.GridSpec(2, 2)
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        xList = [x+1 for x in range(len(self.dataFiles))]

        g0List = []
        g1List = []
        s0List = []
        s1List = []
        s2List = []

        g0Err = []
        g1Err = []
        s0Err = []
        s1Err = []
        s2Err = []


        for k, fileDatas in enumerate(self.dataList):
            g0List.append(self.scatFitList1[k].x[0])
            g0Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[0])
            g1List.append(self.scatFitList1[k].x[1])
            g1Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[1])
            s0List.append(self.scatFitList1[k].x[2])
            s0Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[2])
            s1List.append(self.scatFitList1[k].x[3])
            s1Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[3])
            s2List.append(self.scatFitList1[k].x[4])
            s2Err.append(np.sqrt(np.diag(self.scatFitList1[k].hess_inv.todense()))[4])

            
            #_Plot of the g0 parameter of the fits
            ax = self.figure.add_subplot(mplGrid[:1,0])
            ax.errorbar(xList[k], g0List[k], g0Err[k], marker='o', 
                    linestyle='None', label='g0')
            ax.set_ylabel(r'$\Gamma 0$', fontsize=18)
            ax.set_ylim(0, 5)
            ax.set_xlim([0, len(self.dataFiles) +1])
            ax.grid(True)

            #_Plot of the g1 parameter of the fits
            ax = self.figure.add_subplot(mplGrid[:1,1])
            ax.errorbar(xList[k], g1List[k], g1Err[k], marker='o', 
                    linestyle='None', label='g1')
            ax.set_ylabel(r'$\Gamma 1$', fontsize=18)
            ax.set_xlim([0, len(self.dataFiles) +1])
            ax.set_ylim(0, 500)
            plt.legend(framealpha=0.5)
            ax.grid(True)

            #_Plot of the s0 parameter of the fits
            ax = self.figure.add_subplot(mplGrid[1:,0])
            ax.errorbar(xList[k],s0List[k], s0Err[k], marker='o', 
                    linestyle='None', label=self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.set_ylabel(r'$s0$', fontsize=18)
            ax.set_ylim(0, 1)
            plt.xticks(
                [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
                [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
                rotation=45, ha='right')
            ax.grid(True)

            #_Plot of the s1 and s2 parameters
            ax = self.figure.add_subplot(mplGrid[1:,1])
            ax.errorbar(xList[k], s1List[k], s1Err[k], marker='o', 
                    linestyle='None', label='s1')
            ax.errorbar(xList[k], s2List[k], s2Err[k], marker='o', 
                    linestyle='None', label='s2')
            ax.set_ylabel(r'$ Lorentzians \ contributions$', fontsize=18)
            ax.set_ylim(0, 1)
            plt.xticks(
                [0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
                [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
                rotation=45, ha='right')
            plt.legend(framealpha=0.5)
            ax.grid(True)

        plt.tight_layout()
        self.canvas.draw()

    def fitPlot(self):
                
        plt.gcf().clear()     
        qValToShow = min([datas.qVal for datas in self.dataList[0]], 
                            key = lambda x : abs(float(self.lineEdit.text()) - x))


        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))
        #_Plot the datas for selected q value normalized with integrated curves at low temperature
        for k, fileDatas in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k])
            for j, qDatas in enumerate(fileDatas):
                if qDatas.qVal == qValToShow:
                    resShift = self.resFitList[k][j][0][4]
                    normFact = self.resFitList[k][j][0][0]
                    lorS = self.resFitList[k][j][0][1]
                    GauR = self.resFitList[k][j][0][3]
                    lorR = self.resFitList[k][j][0][2]
                    bkgd = self.resFitList[k][j][0][5] 
                    
                    #_Plot of the measured QENS signal
                    ax.errorbar(qDatas.energies, 
                           [x/normFact for x in qDatas.intensities],
                           [x/normFact for x in qDatas.errors], fmt='o', zorder=2)

                    #_Plot of the resolution function
                    ax.plot(qDatas.energies, 
                            [resFuncAnalysis.resFunc(val, lorR, GauR, lorS, 1, resShift, 0) 
                                             for val in qDatas.energies], label='resolution')

                    #_Plot of the fitted lorentzian
                    g0 = self.scatFitList1[k].x[0] 
                    g1 = self.scatFitList1[k].x[1]
                    s0 = self.scatFitList1[k].x[2]
                    s1 = self.scatFitList1[k].x[3]
                    s2 = self.scatFitList1[k].x[4]
                    msd = self.scatFitList1[k].x[5]


                    ax.plot(qDatas.energies, [g0/(g0**2+val**2) * s1
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian 1')
                    ax.plot(qDatas.energies, [g1/(g1**2+val**2) * s2
                                        for val in qDatas.energies], 
                                        ls='--', label='lorentzian 2')


                    #_Plot of the fitted incoherent structure factor
                    ax.plot(qDatas.energies, self.fittedFunc(qDatas, resShift, normFact, lorS, 
                                        GauR, lorR, bkgd, j, self.scatFitList1[k].x), 
                            label='convolution', linewidth=2, color='orangered', zorder=5)
                    

            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.set_ylim(1e-3, 1.2)
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()
        
        plt.legend(framealpha=0.5)
        plt.tight_layout()
        self.canvas.draw()

    
if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)
    subW = Window() 
    subW.show()

    sys.exit(app.exec_())
