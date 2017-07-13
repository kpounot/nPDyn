import sys
import numpy as np
from collections import namedtuple

def convert(datafile):
    datafile = datafile

    qData = namedtuple('qData', 'qVal energies intensities errors')

    with open(datafile, 'r') as fileinput:
        data = fileinput.read().splitlines()

    #_Get the index of the data corresponding to each q-value
    indexList = []
    for i, value in enumerate(data):
        if value == data[0]:
            indexList.append(i)


    #_Create namedtuples containing lists of temperatures, intensities and errors
    #_for each q-values
    data = [val.split() for i, val in enumerate(data)]
    dataList = []
    for i, value in enumerate(indexList):
        if i < len(indexList)-1:
            dataList.append(qData(data[value+2][0], 
                 [val[0] for j, val in enumerate(data) if indexList[i] + 3 < j < indexList[i+1]],
                 [val[1] for j, val in enumerate(data) if indexList[i] + 3 < j < indexList[i+1]],
                 [val[2] for j, val in enumerate(data) if indexList[i] + 3 < j < indexList[i+1]],
                 ))
        else:
            dataList.append(qData(data[value+2][0], 
                            [val[0] for j, val in enumerate(data) if j > indexList[i]+3],
                            [val[1] for j, val in enumerate(data) if j > indexList[i]+3],
                            [val[2] for j, val in enumerate(data) if j > indexList[i]+3]))

    return dataList    

if __name__ == '__main__':

    dataList = convert(sys.argv[1])
    outQ = []
    for i, val in enumerate(dataList):
        outQ.append(dataList[i].qVal)

    r = (6*'\t').join(outQ)
    r += '\n\n'
    i = 0
    while i < len(dataList[0].data):
        for j, value in enumerate(dataList):
            r += '%8s'%(value.data[i][0]) + '\t' + '%8s'%(value.data[i][1]) 
            r += '\t' + '%8s'%(value.data[i][2]) + '\t' 
        r += '\n'
        i += 1

    print(r)
    #with open(sys.argv[1][:sys.argv[1].find('.')] + '.txt', 'w') as saveFile:
    #    saveFile.write(r)        
