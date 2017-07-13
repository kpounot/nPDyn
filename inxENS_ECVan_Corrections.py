import sys
import numpy as np
import inxConvertENS
import argParser
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QLabel, 
                             QLineEdit, QDialog, QPushButton, QVBoxLayout, QFrame)
from PyQt5 import QtGui
from collections import namedtuple


def inxBin(data, binS):

    dataList = inxConvertENS.convert(data)
    binS = int(binS)

    qDataBin = namedtuple('qDataBin', 'qVal temps intensities errors')
    
    listQ = []
    dataBinList = []
    for j, val in enumerate(dataList):
        listQ.append(val.qVal)
        listT = []
        listI = []
        listE = []
        i = 0
        while (i+1)*binS < len(val.data):
            listT.append(sum([float(value[0]) for l, value 
                              in enumerate(val.data) if i*binS <= l < (i+1)*binS])/binS)          
            listI.append(sum([float(value[1]) for l, value 
                              in enumerate(val.data) if i*binS <= l < (i+1)*binS])/binS)
            listE.append(sum([float(value[2]) for l, value
                              in enumerate(val.data) if i*binS <= l < (i+1)*binS])/binS)
            i += 1            
            
        listT.append(sum([float(value[0]) for l, value in enumerate(val.data) 
                          if i*binS <= l < len(val.data)])/(len(val.data)-i*binS))
        listI.append(sum([float(value[1]) for l, value in enumerate(val.data) 
                          if i*binS <= l < len(val.data)])/(len(val.data)-i*binS))
        listE.append(sum([float(value[2]) for l, value in enumerate(val.data) 
                          if i*binS <= l < len(val.data)])/(len(val.data)-i*binS))

        dataBinList.append(qDataBin(listQ[j], listT, listI, listE))    

    return dataBinList

def backgroundRemoval(datas, dataList):

    bkgdMessage = QMessageBox.information(QWidget(), 'Background file selection',
                        'Please select the file containing the background measurements')
    bkgdFile = QFileDialog().getOpenFileName()[0] 
    bkgdDatas = inxBin(bkgdFile, karg['binS'])
 
    for i, val in enumerate(dataList):
        val.intensities = [I - bkgdDatas[i].intensities[I] for I in val.intensities]

    return dataList
    
def vanaNormalization(datas, dataList):
    
    vanaMessage = QMessageBox.information(QWidget(), 'Background file selection',
                        'Please select the file containing the background measurements')
    vanaFile = QFileDialog().getOpenFileName()[0] 
    vanaDatas = inxBin(bkgdFile, karg['binS'])

if __name__ == '__main__':

    app = QApplication(sys.argv)

    arg, karg = argParser.argParser(sys.argv)


    dataFile = sys.argv[1]
    dataList = inxBin(sys.argv[1], karg['binS'])
    
    #with open(dataFile[dataFile.rfind('.')] + '_binned.inx', 'w') as fichier:    
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
    #fichier.write(r)

    sys.exit(app.exec_())
