import sys, os
import numpy as np
import inxBinENS
import inxBinQENS
import argParser
import re

from collections import namedtuple

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QMessageBox

def EC_Correction(dataFile, emptyCellFile, binS, totalRange=True):
    
    dataList = inxBinENS.inxBin(dataFile, binS)
    ECData = inxBinQENS.inxBin(emptyCellFile, 1)

    qMin = float(kargList['qMin'])
    qMax = float(kargList['qMax'])

    qList = list(map(float, [i.qVal for i in dataList]))
    reducedqList = [val for val in qList if qMin < val < qMax]
    reducedIndex = [i for i, val in enumerate(qList) if qMin < val < qMax]

    EC_elastic = [qData.intensities[np.argmin(np.abs(qData.energies))] for qData in ECData]

    if totalRange == False:
        dataList = [dataList[i] for i in reducedIndex]
        EC_elastic = [EC_elastic[i] for i in reducedIndex]

    for i, val in enumerate(dataList):
        val = val._replace(intensities=val.intensities - 0.90*EC_elastic[i])

    return dataList


if __name__ == '__main__':

    argList, kargList = argParser.argParser(sys.argv)
    app = QApplication(sys.argv)

    message = QMessageBox.information(QWidget(), 'File selection',
            'Please select the file containing the empty cell data...') 
    ECFile = QFileDialog().getOpenFileName()[0]

    dataList = EC_Correction(sys.argv[1], ECFile, kargList['binS'])

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




