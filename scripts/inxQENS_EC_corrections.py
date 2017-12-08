import sys
import numpy as np
import inxBinQENS
import argParser
import re
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple
from scipy import optimize
from PyQt5.QtWidgets import QApplication, QFileDialog

def EC_Correction(dataFile, binS, totalRange=True):
    
    dataList = inxBinQENS.inxBin(dataFile, binS)

    fileToOpen = QFileDialog().getOpenFileName()[0]
    ECDataList = inxBinQENS.inxBin(fileToOpen, binS)

    qMin = float(kargList['qMin'])
    qMax = float(kargList['qMax'])

    qList = list(map(float, [i.qVal for i in dataList]))
    reducedqList = [val for val in qList if qMin < val < qMax]
    reducedIndex = [i for i, val in enumerate(qList) if qMin < val < qMax]

    if totalRange == False:
        dataList = [dataList[i] for i in reducedIndex]

    for i, val in enumerate(dataList):
        dataList[i] = val._replace(intensities=[intensities - 0.95*ECDataList[i].intensities[j]
                                    for j, intensities in enumerate(dataList[i].intensities)])

    return dataList


if __name__ == '__main__':

    argList, kargList = argParser.argParser(sys.argv)
    app = QApplication(sys.argv)

    dataList = EC_Correction(sys.argv[1], 1)

    r = ''                                          
    for i, val in enumerate(dataList):
        r += '%d    1    2    0   0   0   0   %d\n' % (len(val.energies)+3, len(val.energies))
        r += '    title...\n'
        r += '     %s    %s    0    0    0    0\n' % (val.qVal, dataList[-1].qVal)
        r += '    0    0    0\n'
        for j, values in enumerate(val.energies):
            r += '    %.5f    %.5f    %.5f\n' % (values, val.intensities[j], val.errors[j])
    r = r[:-2]
    print(r)

    sys.exit(app.exec_()) 

