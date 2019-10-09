'''
This script extracts the elastic intensities at low temperature within the given q-range.
These can be used as normalization factors for elastic or QENS data. '''

import sys
import numpy as np
import inxBinENS
import argParser
import re
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple
from scipy import optimize
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

def get_elastic_normF():
   
    argList, karg = argParser.argParser(sys.argv)

    message = QMessageBox.information(QWidget(), 'File selection',
            'Please select the file from which extract the normalization factors...') 
    dataFile = QFileDialog().getOpenFileName()[0]

    qMin = float(karg['qMin'])
    qMax = float(karg['qMax'])
    normFact = int(karg['normFactor'])
    binS = int(karg['binS'])

    dataList = inxBinENS.inxBin(dataFile, binS)

    qList = list(map(float, [i.qVal for i in dataList]))
    reducedqList = [val for val in qList if qMin < val < qMax]
    reducedIndex = [i for i, val in enumerate(qList) if qMin < val < qMax]

    #_Discard the selected index
    if karg['qDiscard'] is not '':
        qDiscardPattern = re.compile(r'[ ,:;-]+')
        qDiscardList = qDiscardPattern.split(karg['qDiscard'])
        qDiscardList = [min(qList, key = lambda x: abs(float(val) - x)) for val in qDiscardList]
        zipList = [(i + min(reducedIndex), val) for i, val in enumerate(reducedqList) 
                            if val not in qDiscardList] 
        reducedIndex, reducedqList = list(zip(*zipList))[0], list(zip(*zipList))[1]

    dataList = [val for i, val in enumerate(dataList) if i in reducedIndex]



    #_Extract the intensities and errors for the same temperature depending on q values 
    normIList = [np.mean(q.intensities[:normFact]) for q in dataList]

    return normIList
