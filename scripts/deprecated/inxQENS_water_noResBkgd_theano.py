import sys, os
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
from scipy.special import wofz, sph_jn
from scipy.stats import chisquare, bayes_mvs
from scipy.misc import factorial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gridspec
import matplotlib
import pickle as pk

import theano_functions as tf
modelFunc, modelCost, modelGrad = tf.modelFunc_water()
resFunc = tf.resFunc_only()

class Window(QDialog):
    def __init__(self, new_fit=True):
        super().__init__()
    
        arg, karg = argParser.argParser(sys.argv)

        self.dataFiles = arg[1:]
        self.dataList = []
        self.resDataList = []
        self.resFitList = []
        self.scatFitList = []
        self.HO_dist = 0.98
        self.new_fit = new_fit

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
            print('No value for the maximum order of Bessel function to be' +
                            ' used was given. \n' +
                            'Using maxBesselOrder=3 as default value.\n', flush=True)

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
        with open('resFunc_params', 'rb') as resFit:
            resFitData = pk.Unpickler(resFit).load()
            self.resFitList = resFitData[0]
            self.meanResGauWidth = resFitData[1]

        #_Correction for the normalization factor (optional)
        #for i, val in enumerate(self.resFitList[0]):
        #    val[0][0] = val[0][0] * (self.meanResGauWidth[0] / val[0][3])**0.8

        #_Get data from the file and store them into self.dataList 
        for i, dataFile in enumerate(self.dataFiles):
            inxdata = inxBinQENS.inxBin(dataFile, karg['binS'])
            self.dataList.append(inxdata)    

        #_Keep the data from wanted q-range
        qMin = min([data.qVal for data in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMin']) - x))
        qMax = min([data.qVal for data in self.dataList[0]], 
	                key = lambda x : abs(float(karg['qMax']) - x))
        for i, filedata in enumerate(self.dataFiles):
            self.dataList[i] = [val for val in self.dataList[i] if qMin <= val.qVal <= qMax]

        #_Discard the selected index
        if karg['qDiscard'] is not '':
            qDiscardPattern = re.compile(r'[ ,:;-]+')
            qDiscardList = qDiscardPattern.split(karg['qDiscard'])
            qDiscardList = [min([data.qVal for data in self.dataList[0]], 
	                key = lambda x: abs(float(val) - x)) for val in qDiscardList]
            for i, filedata in enumerate(self.dataFiles):
                self.dataList[i] = [val for val in self.dataList[i] 
                                    if val.qVal not in qDiscardList] 

        
        self.new_fit = True if karg['new_fit']=='True' else False
        if self.new_fit:
            self.scatFit()
        else:
            with open('modelFit_params_theano', 'rb') as modelFit:
                self.scatFitList = pk.Unpickler(modelFit).load()


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
    def model_cost(self, x, X, dataQ, dataI, dataErr, bessel, res_w):

        cost = modelCost(dataQ, res_w, x, X, bessel, dataI, dataErr)
        grad = modelGrad(dataQ, res_w, x, X, bessel, dataI, dataErr)

        return cost.astype('float64')#, grad.astype('float64').ravel()


    def scatFit(self):
    
        scatList = []

        for i, filedata in enumerate(self.dataList):
            print('>>> File : ' + self.dataFiles[i][self.dataFiles[i].rfind('/'):], flush=True)
            print(75*'_', flush=True)
            
            #_Minimization 
            x0 = [2, 4, 0.2, 0.4, 0.4, 1, 2] + [0.005 for val in range(len(filedata))]
            bounds = np.array([(0., 30), (0., 30), (0., 1), (0., 1), (0., 1), (0., 10), (0., 5)]
                     + [(0., 0.1) for val in range(len(filedata))])

            X = filedata[i].energies
            dataQ = np.hstack((data.qVal for data in filedata))
            dataI = np.stack((data.intensities/self.resFitList[0][k][0][0] for k, data in enumerate(filedata)))
            dataErr = np.stack((data.errors for k, data in enumerate(filedata)))
            bessel = np.stack((sph_jn(self.maxBesselOrder, self.HO_dist * qVal)[0] for qVal in dataQ))
            res_w = np.stack((np.hstack(([1], val[0][1:-1], [0])) for val in self.resFitList[0])) 

            scatList.append(optimize.basinhopping(self.model_cost,
                                   x0,
                                   niter=self.BH_iter,
                                   niter_success=0.5*self.BH_iter,
                                   disp=True,
                                   interval=25,
                                   T=0.5,
                                   minimizer_kwargs={'bounds': bounds, 
                                                     #'jac': True, 
                                                     'args': (X, dataQ, dataI, dataErr, bessel, res_w),
                                                     'options':{'factr': 100,
                                                                'maxcor': 100,
                                                                'maxfun': 200000}}))

            print('> Final result : ', flush=True)
            print('Fit s0 : ' + str(scatList[i].x[0]), flush=True)
            print('Fit st : ' + str(scatList[i].x[1]), flush=True)
            print('Fit gt : ' + str(scatList[i].x[2]), flush=True)
            print('Fit sr : ' + str(scatList[i].x[3]), flush=True)
            print('Fit gr : ' + str(scatList[i].x[4]), flush=True)
            print('Fit msd : ' + str(scatList[i].x[5]), flush=True)
            print('Fit lambda : ' + str(scatList[i].x[6]) + '\n', flush=True)


            self.scatFitList = scatList

            with open(os.path.dirname(os.path.abspath(__file__)) + '/modelFit_params_theano',
                                                                            'wb') as fittedParamsFile:
                myFile = pk.Pickler(fittedParamsFile)
                modelFitList = self.scatFitList
                myFile.dump(modelFitList)


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
                    normFact = self.resFitList[0][j][0][0]
                    ax.errorbar(qdata.energies, 
                           [x / normFact for x in qdata.intensities],
                           [x / normFact for x in qdata.errors], fmt='o')
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
        ax = [self.figure.add_subplot(2, 5, i+1, projection='3d') for i, val in enumerate(self.dataList)]

        normColors = matplotlib.colors.Normalize(vmin=0, vmax=2)
        cmap = matplotlib.cm.get_cmap('winter')

        for k, filedata in enumerate(self.dataList):
            for i, data in enumerate(filedata):
                normFact = self.resFitList[0][i][0][0]

                ax[k].plot(data.energies, [val/normFact for val in data.intensities], 
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
        ax1.grid(True)

        #_Plot of the a parameters
        ax2.errorbar(xList + 0.05, stList/sTot, marker='o', 
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


        mplGrid = gridspec.GridSpec(1, len(self.dataFiles))
        #_Plot the data for selected q value normalized with integrated curves at low temperature
        for k, filedata in enumerate(self.dataList):
            ax = self.figure.add_subplot(mplGrid[:,k])
            for j, qdata in enumerate(filedata):
                if qdata.qVal == qValToShow:
                    normFact = self.resFitList[0][j][0][0]
                    pBkgd = self.scatFitList[k].x[7+j]
                    
                    #_Plot of the measured QENS signal
                    ax.errorbar(qdata.energies, 
                           [x/normFact for x in qdata.intensities],
                           [x/normFact for x in qdata.errors], fmt='o')

                    #_Plot of the resolution function
                    ax.plot(qdata.energies, 
                            resFunc(np.hstack(([1.], self.resFitList[k][j][0][1:-1], [0.])), qdata.energies), 
                            label='resolution')

                    #_Plot of the fitted lorentzian
                    gt = self.scatFitList[k].x[2] 
                    gr = self.scatFitList[k].x[4]
                    st = self.scatFitList[k].x[1]
                    sr = self.scatFitList[k].x[3]


                    ax.plot(qdata.energies, [gt/(gt**2+val**2) * st
                                        for val in qdata.energies], 
                                        ls='--', label='translational')
                    ax.plot(qdata.energies, [gr/(gr**2+val**2) * sr
                                        for val in qdata.energies], 
                                        ls='--', label='rotational')
                    ax.axhline(y=pBkgd, label='background')                    


                    #_Plot of the fitted incoherent structure factor
                    bessel = sph_jn(self.maxBesselOrder, self.HO_dist * qdata.qVal)[0]
                    ax.plot(qdata.energies, modelFunc(np.array([qdata.qVal]), 
                                            np.array([np.hstack(([1.], self.resFitList[k][j][0][1:-1], [0.]))]),
                                            self.scatFitList[k].x,
                                            np.array(qdata.energies),
                                            np.array([bessel]))[0], 
                            label='convolution', linewidth=2, color='orangered')
                    

            ax.set_xlabel(r'$\hslash\omega (\mu eV)$', fontsize=18)
            ax.set_yscale('log')
            ax.set_ylabel(r'$S(' + str(qValToShow) + ', \omega)$', fontsize=18)   
            ax.set_ylim(1e-3, 1.2)
            ax.set_title('...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):])
            ax.grid()
        
        plt.legend(framealpha=0.5)
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
                    resShift = self.resFitList[0][j][0][4]
                    normFact = self.resFitList[0][j][0][0]
                    lorS = self.resFitList[0][j][0][1]
                    GauR = self.resFitList[0][j][0][3]
                    lorR = self.resFitList[0][j][0][2]
                    bkgd = 0

                    gt = self.scatFitList[k].x[0] 
                    gr = self.scatFitList[k].x[1]
                    s0 = self.scatFitList[k].x[2]
                    st = self.scatFitList[k].x[3]
                    sr = self.scatFitList[k].x[4]
                    msd = self.scatFitList[k].x[5]
                    diffFactor = self.scatFitList[k].x[6]
                    pBkgd = self.scatFitList[k].x[j + 7]

                    elastic_calc.append(max(self.fittedFunc(qdata, resShift, normFact, lorS, GauR, 
                                            lorR, gt, gr, s0, st, sr, bkgd, msd, pBkgd, diffFactor)))
                    elastic_exp.append(max(qdata.intensities / normFact)) 

            ax.plot([val.qVal for val in self.dataList[0]],
                        elastic_calc,
                        marker='o', linestyle='-',
                        label='...' + self.dataFiles[k][self.dataFiles[k].rfind('/'):],
                        c=cmap(normColors(k)))

            ax.plot([val.qVal for val in self.dataList[0]],
                        elastic_exp,
                        linestyle='--',
                        label='experimental',
                        c=cmap(normColors(k)))
        ax.grid()
        ax.set_xlabel(r'$q-values$', fontsize=18)
        ax.set_ylabel(r'$Elastic \ intensities$', fontsize=18)   

        ax.legend(loc='upper right', framealpha=0.5)

        
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)
    subW = Window() 
    subW.show()

    sys.exit(app.exec_())
