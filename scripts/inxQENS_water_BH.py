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
from scipy.special import spherical_jn
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib
from inxENS_extract_normF import get_elastic_normF


class Window(QDialog):
    def __init__(self):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resDataList = []
        self.resFitList = []
        self.meanResGauWidth = []
        self.scatFitList = []
        self.HO_dist = 0.96
        self.normF = []

        #_Parsing the additional parameters
        try:
            self.maxBesselOrder = int(karg['maxBesselOrder'])
        except ValueError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'Error: the "maxBesselOrder" parameter can\'t be converted to an' +
                            ' integer.')
            sys.exit()
        except KeyError:
            self.maxBesselOrder = 3
            print(('No value for the maximum order of Bessel function to be' +
                            ' used was given. \n' +
                            'Using maxBesselOrder=3 as default value.\n'), flush=True)

                    
        try:
            self.BH_iter = int(karg['BH_iter'])
        except ValueError:
            errorMessage = QMessageBox.critical(QWidget(), 'Parameter error',
                            'Error: the "BH_iter" parameter can\'t be converted to an' +
                            ' integer.')
            sys.exit()
        except KeyError:
            self.BH_iter = 300
            print('No value for the number of basinhopping iterations to be' +
                            ' used was given. \n' +
                            'Using BH_iter=300 as default value.\n', flush=True)

        #_Get the resolution function parameters
        with open(os.path.dirname(os.path.abspath(__file__)).rstrip('scripts') + 'params/resFunc_params', 
                  'rb') as resFit:
            resFitData = pk.Unpickler(resFit).load()
            self.resFitList = resFitData[0]
            #_If there is only of fit for the resolution function, we assume the same is used
            #_for all data files
            if len(self.resFitList) == 1:
                self.resFitList = np.tile(self.resFitList, len(self.dataFiles))
            self.meanResGauWidth = resFitData[1]

        #_Get datas from the file and store them into self.dataList 
        for i, dataFile in enumerate(self.dataFiles):
            inxDatas = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxDatas)    


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
                self.scatFitList = pk.Unpickler(params).load()

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

        self.qWisePlotButton = QPushButton('q-Wise Plot')
        self.qWisePlotButton.clicked.connect(self.qWisePlot)

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
        layout.addWidget(self.qWisePlotButton)
        self.setLayout(layout)


#_Everything needed for the fit
    def fitFunc(self, x, filedata, i):

        bkgdL = []

        chiSquared = 0
        gt = x[0] 
        gr = x[1]
        s0 = x[2]
        st = x[3]
        sr = x[4]
        msd = x[5]
        diffFactor = x[6]

        cost = 0

        for j, data in enumerate(filedata):

            shift = self.resFitList[i][j][0][4]
            normFact = self.resFitList[i][j][0][0]
            S = self.resFitList[i][j][0][1]
            gauW = self.resFitList[i][j][0][3]
            lorW = self.resFitList[i][j][0][2]
            bkgd = 0
            pBkgd = x[j + 7]

            bessel = np.array([spherical_jn(i, self.HO_dist * data.qVal) 
                                for i in range(0, self.maxBesselOrder + 1)])
            bessel = bessel.reshape((bessel.shape[0], 1))
            L = np.arange(1, bessel.shape[0]).reshape((bessel.shape[0]-1, 1))
            X = data.energies
            
            #_Resolution function
            f_res = ((S * lorW / (lorW**2 + (X - shift)**2) / np.pi 
                    + (1-S) * np.exp(-(X - shift)**2 / (2*gauW**2)) / (gauW * np.sqrt(2*np.pi)) 
                    + bkgd))

            #_Lorentzians
            f_lor_t = (gt * data.qVal**diffFactor / (X**2 + (gt * data.qVal**diffFactor)**2))

            f_lor_r = np.sum((2*L+1) * bessel[1:]**2 
                                  * L*(L+1) * gr / (np.pi * (X**2 + (L*(L+1) * gr)**2)), axis=0)

            model = (np.exp(-data.qVal**2*msd/3)
                    * ((s0 + sr * bessel[0]**2) * f_res
                    + st * np.convolve(f_res, f_lor_t, mode='same')
                    + sr * np.convolve(f_res, f_lor_r, mode='same')
                    + pBkgd))

            cost += np.sum((data.intensities / self.normF[i][j] - model)**2 
                            / (data.errors / self.normF[i][j])**2)


        return cost


    def scatFit(self):
    
        scatList1 = []
        for i, filedata in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            
            #_Minimization 
            x0 = [2, 4, 0.2, 0.4, 0.4, 1, 2] + [0.005 for val in range(len(filedata))]
            bounds = np.array([(0., 30), (0., 30), (0., 1), (0., 1), (0., 1), (0., 10), (0., 5)]
                     + [(0., 0.2) for val in range(len(filedata))])


            if i != 0: x0 = self.scatFitList[i-1].x
            scatList1.append(optimize.basinhopping(self.fitFunc,
                                                x0, 
                                                niter=self.BH_iter,
                                                T=0.5,
                                                interval=25,
                                                disp=True,
                                                niter_success=0.5*self.BH_iter,
                                                minimizer_kwargs={'bounds': bounds,
                                                                  'args': (filedata, i),
                                                                  'options': {'factr': 100,
                                                                              'maxcor': 100,
                                                                              'maxfun': 200000}}))

            print('> Final result : ', flush=True)
            print('Fit gt : ' + str(scatList1[i].x[0]), flush=True)
            print('Fit gr : ' + str(scatList1[i].x[1]), flush=True)
            print('Fit s0 : ' + str(scatList1[i].x[2]), flush=True)
            print('Fit st : ' + str(scatList1[i].x[3]), flush=True)
            print('Fit sr : ' + str(scatList1[i].x[4]), flush=True)
            print('Fit msd : ' + str(scatList1[i].x[5]), flush=True)
            print('Fit Lambda : ' + str(scatList1[i].x[6]) + '\n', flush=True)


            self.scatFitList = scatList1

        with open(self.paramsFile, 'wb') as fittedParamsFile:
            myFile = pk.Pickler(fittedParamsFile)
            modelFitList = self.scatFitList
            myFile.dump(modelFitList)

    #_Function used to produce the plot of the fit
    def fittedFunc(self, data, j, resP, modelP):


        bessel = np.array([spherical_jn(i, self.HO_dist * data.qVal) 
                            for i in range(0, self.maxBesselOrder + 1)])
        bessel = bessel.reshape((bessel.shape[0], 1))
        L = np.arange(1, bessel.shape[0]).reshape((bessel.shape[0]-1, 1))
        X = data.energies

        #_Getting the resolution function parameters
        normFact = resP[0]
        S = resP[1]
        lorW = resP[2]
        gauW = resP[3]
        shift = resP[4]
        bkgd = 0

        #_Getting the model parameters
        gt = modelP[0]
        gr = modelP[1]
        s0 = modelP[2]
        st = modelP[3]
        sr = modelP[4]
        msd = modelP[5]
        diffFactor = modelP[6]
        pBkgd = modelP[j + 7]

        
        #_Resolution function
        f_res = ((S * lorW / (lorW**2 + (X - shift)**2) / np.pi 
                + (1-S) * np.exp(-(X - shift)**2 / (2*gauW**2)) / (gauW * np.sqrt(2*np.pi)) 
                + bkgd))

        #_Lorentzians
        f_lor_t = (gt * data.qVal**diffFactor / (X**2 + (gt * data.qVal**diffFactor)**2))

        f_lor_r = np.sum((2*L+1) * bessel[1:]**2 
                              * L*(L+1) * gr / (np.pi * (X**2 + (L*(L+1) * gr)**2)), axis=0)

        model = (np.exp(-data.qVal**2*msd/3)
                * ((s0 + sr * bessel[0]**2) * f_res
                + st * np.convolve(f_res, f_lor_t, mode='same')
                + sr * np.convolve(f_res, f_lor_r, mode='same')
                + pBkgd))


        return model        


    def resFunc(self, x, normF, S, lorW, gauW, shift, bkgd):

        return  (normF * (S * lorW/(lorW**2 + (x-shift)**2) /np.pi 
                + (1-S) * np.exp(-((x-shift)**2) / (2*gauW**2)) / (gauW*np.sqrt(2*np.pi))
                + bkgd))  

#_Definitions of the slots for the plot window
    def plot(self):
	   
        plt.gcf().clear()     
        ax = self.figure.add_subplot(111)  
        
        qValToShow = min([data.qVal for data in self.dataList[0]], 
	                key = lambda x : abs(float(self.lineEdit.text()) - x))
        #_Plot the data for selected q value normalized with integrated curves at low temperature
        for k, filedata in enumerate(self.dataList):
            for j, qdata in enumerate(filedata):
                if qdata.qVal == qValToShow:
                    ax.errorbar(qdata.energies, 
                           [x / self.normF[k][j] for x in qdata.intensities],
                           [x / self.normF[k][j] for x in qdata.errors], fmt='o')
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
        #_3D plots of the data for each selected datafiles
        if len(self.dataList) <= 5:
            ax = [self.figure.add_subplot(1, len(self.dataList), i+1, projection='3d') 
                                        for i, val in enumerate(self.dataList)]
        else:
            ax = [self.figure.add_subplot(2, 5, i+1, projection='3d') 
                                        for i, val in enumerate(self.dataList)]

        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for k, filedata in enumerate(self.dataList):
            for i, data in enumerate(filedata):

                ax[k].plot(data.energies, [val / self.normF[k][i] for val in data.intensities], 
                                  data.qVal, zdir='y', c=cmap(normColors(data.qVal)))

            ax[k].set_xlabel(r'$\hslash \omega (\mu eV)$')
            ax[k].set_ylabel(r'$q$')
            ax[k].set_zlabel(r'$S_{300K}(q, \omega)$')
            ax[k].set_ylim((0, 2))
            ax[k].set_zlim((0, 1))
            ax[k].set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax[k].grid()

        plt.tight_layout()
        self.canvas.draw()
    
    #_Plot of the parameters resulting from the fit procedure
    def analysisPlot(self):

        plt.gcf().clear()     

        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223, sharex=ax1)
        ax4 = self.figure.add_subplot(224, sharex=ax1)

        #_Plot the data for selected q value normalized with integrated curves at low temperature
        xList = np.array([x+1 for x in range(len(self.dataFiles))])

        gtList = []
        grList = []
        s0List = []
        stList = []
        srList = []

        for k, filedata in enumerate(self.dataList):
            gtList.append(self.scatFitList[k].x[0])
            grList.append(self.scatFitList[k].x[1])
            s0List.append(self.scatFitList[k].x[2])
            stList.append(self.scatFitList[k].x[3])
            srList.append(self.scatFitList[k].x[4])

        s0List = np.array(s0List)
        stList = np.array(stList)
        srList = np.array(srList)

        sTot = s0List + stList + srList

        #_Plot of the gamma parameters of the fits
        ax1.errorbar(xList - 0.05, gtList, marker='o', 
                linestyle='-', label='translational')
        ax1.errorbar(xList, grList, marker='^', 
                linestyle='-', label='rotational')
        ax1.set_ylabel(r'$\Gamma$', fontsize=18)
        ax1.legend(framealpha=0.5, loc='upper center')
        ax1.grid(True)

        #_Plot of the s parameters
        ax2.errorbar(xList, stList/sTot, marker='o', 
                linestyle='-', label='st')
        ax2.set_ylabel(r'$ Lorentzians \ contributions$', fontsize=14)
        ax2.legend(framealpha=0.5, loc='upper center')
        ax2.grid(True)

        ax3.errorbar(xList, srList/sTot, marker='^', 
                linestyle='-', label='sr')
        ax3.set_ylabel(r'$ Lorentzians \ contributions$', fontsize=14)
        ax3.legend(framealpha=0.5, loc='upper center')
        plt.xticks([0] + [i + 1 for i, val in enumerate(self.dataFiles)] + [len(self.dataFiles)+1], 
                   [''] + [val[self.dataFiles[k].rfind('/'):] for val in self.dataFiles] + [''],
                   rotation=45, ha='right')
        ax3.grid(True)

        ax4.errorbar(xList, s0List/sTot, marker='x', 
                linestyle='-', label='s0')
        ax4.set_ylabel(r'$ Lorentzians \ contributions$', fontsize=14)
        ax4.legend(framealpha=0.5, loc='upper center')
        ax4.grid(True)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        self.canvas.draw()

    def fitPlot(self):
                
        plt.gcf().clear()     
        qValToShow = min([data.qVal for data in self.dataList[0]], 
                            key = lambda x : abs(float(self.lineEdit.text()) - x))

        if len(self.dataList) <= 5:
            ax = [self.figure.add_subplot(1, len(self.dataList), i+1) 
                                        for i, val in enumerate(self.dataList)]
        else:
            ax = [self.figure.add_subplot(2, 5, i+1) 
                                        for i, val in enumerate(self.dataList)]

        #_Plot the data for selected q value normalized with integrated curves at low temperature
        for k, filedata in enumerate(self.dataList):
            for j, qdata in enumerate(filedata):
                if qdata.qVal == qValToShow:

                    #_Plot of the measured QENS signal
                    ax[k].errorbar(qdata.energies, [x / self.normF[k][j] for x in qdata.intensities],
                           [x / self.normF[k][j] for x in qdata.errors], fmt='o')

                    #_Plot of the resolution function
                    ax[k].plot(qdata.energies,
                               [self.resFunc(val, 1, *self.resFitList[k][j][0][1:-1], 0) 
                               for val in qdata.energies], 
                               label='resolution')

                    #_Plot of the fitted lorentzian
                    gt = self.scatFitList[k].x[0] 
                    gr = self.scatFitList[k].x[1]
                    st = self.scatFitList[k].x[3]
                    sr = self.scatFitList[k].x[4]
                    pBkgd = self.scatFitList[k].x[j+7]

                    ax[k].plot(qdata.energies, [gt/(gt**2+val**2) * st
                                        for val in qdata.energies], 
                                        ls='--', label='translational')
                    ax[k].plot(qdata.energies, [gr/(gr**2+val**2) * sr
                                        for val in qdata.energies], 
                                        ls='--', label='rotational')
                    ax[k].axhline(y=pBkgd, label='background')                    


                    #_Plot of the fitted incoherent structure factor
                    ax[k].plot(qdata.energies, 
                               self.fittedFunc(qdata, j, self.resFitList[k][j][0], self.scatFitList[k].x),
                               label='convolution', linewidth=2, color='orangered', zorder=5)
                    

            ax[k].set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax[k].set_yscale('log')
            ax[k].set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax[k].set_ylim(1e-3, 1.2)
            ax[k].set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax[k].grid()
        
        [plt.setp(axis.get_xticklabels(), visible=False) for axis in ax[:5]]
        [plt.setp(axis.get_yticklabels(), visible=False) for axis in ax[1:5]]
        [plt.setp(axis.get_yticklabels(), visible=False) for axis in ax[6:]]
        ax[-1].legend(framealpha=0.5, loc='upper right', bbox_to_anchor=(1.5, 1))
        plt.tight_layout()
        self.canvas.draw()

    
    def qWisePlot(self):
	   
        plt.gcf().clear()     
        
        normColors = matplotlib.colors.Normalize(vmin=0, vmax=len(self.dataFiles))
        cmap = matplotlib.cm.get_cmap('rainbow')

        ax = self.figure.add_subplot(111)

        #_Plot the data for selected q value normalized with integrated curves at low temperature
        for k, filedata in enumerate(self.dataList):

            elastic_calc = []
            elastic_exp = []
            for j, qdata in enumerate(filedata):
                elastic_calc.append(max(self.fittedFunc(qdata, j, self.resFitList[k][j][0], 
                                                                  self.scatFitList[k].x)))
                elastic_exp.append(max(qdata.intensities / self.normF[k][j])) 

            ax.plot([val.qVal for val in self.dataList[0]],
                        elastic_calc,
                        marker='o', linestyle='-',
                        label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):],
                        c=cmap(normColors(k)))

            ax.plot([val.qVal for val in self.dataList[0]],
                        elastic_exp,
                        linestyle='--',
                        c=cmap(normColors(k)))
        ax.grid()
        ax.set_xlabel(r'$q-values$', fontsize=18)
        ax.set_ylabel(r'$Elastic \ intensities$', fontsize=18)   

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='upper right', framealpha=0.5)
        
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)
    subW = Window() 
    subW.show()

    sys.exit(app.exec_())
