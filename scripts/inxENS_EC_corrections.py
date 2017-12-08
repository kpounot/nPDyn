import sys, os
import numpy as np
import inxBinENS
import argParser
import re
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple
from scipy import optimize
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QMessageBox
import inxQENS_extract_resFunc_analysis as resAnalysis

def EC_Correction(dataFile, binS, totalRange=True):
    
    dataList = inxBinENS.inxBin(dataFile, binS)

    qMin = float(kargList['qMin'])
    qMax = float(kargList['qMax'])

    qList = list(map(float, [i.qVal for i in dataList]))
    reducedqList = [val for val in qList if qMin < val < qMax]
    reducedIndex = [i for i, val in enumerate(qList) if qMin < val < qMax]

    if totalRange == False:
        dataList = [dataList[i] for i in reducedIndex]

    for i, val in enumerate(dataList):
        normFact = resFitList[i][0][0]
        val = val._replace(intensities=val.intensities - 0.95*normFact)

    return dataList


if __name__ == '__main__':

    argList, kargList = argParser.argParser(sys.argv)
    app = QApplication(sys.argv)

    messageWidget = QMessageBox()
    messageWidget.information(QWidget(), 'File selection',
            'Please select the file containing the empty cell data...') 

    resFitList, meanWidth, dataList = resAnalysis.getFitList()

    dataList = EC_Correction(sys.argv[1], kargList['binS'])

    r = ''                                          
    for i, val in enumerate(dataList):
        r += '%d    1    2    0   0   0   0   %d\n' % (len(val.temps)+3, len(val.temps))
        r += '    title...\n'
        r += '     %s    %s    0    0    0    0\n' % (val.qVal, dataList[-1].qVal)
        r += '    0    0    0\n'
        for j, values in enumerate(val.temps):
            r += '    %.5f    %.5f    %.5f\n' % (values, val.intensities[j], val.errors[j])
    r = r[:-2]
    print(r)

    messageWidget.information(QWidget(), 'Processing done',
            'Data correction\'s done successfully.')

    messageWidget.done(0)




