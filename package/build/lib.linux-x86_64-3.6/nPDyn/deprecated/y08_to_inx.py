import sys
import numpy as np
import re
from collections import namedtuple

def convert(datafile):
    datafile = datafile

    fullDataList = []
    qData = namedtuple('qData', 'qVal xVec intensities errors')

    with open(datafile, 'r') as fileinput:
        data = fileinput.read().splitlines()

    dataList = []
    qList = []
    tempList = []
    intensList = []
    errorsList = []

    dataPattern = re.compile(r'^([ ]+[-]?[0-9]+[.]?[0-9]*){3}$')
    for i, value in enumerate(data):
        if re.search(re.compile(r'[ ]*z0:[ ]+[0-9]+[.]?[0-9]+'), value):
            qList.append(float(value.split()[1]))
            if len(qList) > 1:
                dataList.append([tempList, intensList, errorsList])
                tempList = []
                intensList = []
                errorsList = []
        elif re.search(dataPattern, value):
            tempList.append(float(value.split()[0]))
            intensList.append(float(value.split()[1]))
            errorsList.append(float(value.split()[2]))
    dataList.append([tempList, intensList, errorsList])

    #_Create namedtuples containing lists of temperatures/energies, intensities and errors
    #_for each q-values
    for i, val in enumerate(qList):
        fullDataList.append(qData(val, dataList[i][0], dataList[i][1], dataList[i][2]))

    return fullDataList    

if __name__ == '__main__':

    dataList = convert(sys.argv[1])

    r = ''                                          
    for i, val in enumerate(dataList):
        r += '%d    1    2    0   0   0   0   %d\n' % (len(val.xVec)+3, 
                                                       len(val.xVec))
        r += '    title...\n'
        r += '     %s    %.3f    0    0    0    0\n' % (val.qVal, dataList[-1].qVal)
        r += '    0    0    0\n'
        for j, values in enumerate(val.xVec):
            r += '    %.5f    %.5f    %.5f\n' % (values, val.intensities[j], val.errors[j])
    r = r[:-2]
    print(r)
