import sys
import numpy as np
import inxConvertQENS
import argParser
from collections import namedtuple


def inxBin(data, binS):

    dataList = inxConvertQENS.convert(data)
    binS = int(binS)

    qDataBin = namedtuple('qDataBin', 'qVal energies intensities errors')

    dataBinList = []
    listQ = []
    for j, val in enumerate(dataList):
        listQ.append(float(val.qVal))
        listE = []
        listI = []
        listEr = []
        i = 0
        while (i+1)*binS < len(val.energies):
            listE.append(sum([float(value) for l, value 
                              in enumerate(val.energies) if i*binS <= l < (i+1)*binS])/binS)
            listI.append(sum([float(value) for l, value 
                              in enumerate(val.intensities) if i*binS <= l < (i+1)*binS])/binS)
            listEr.append(sum([float(value) for l, value
                              in enumerate(val.errors) if i*binS <= l < (i+1)*binS])/binS)
            i += 1            
            
        listE.append(sum([float(value) for l, value in enumerate(val.energies) 
                          if i*binS <= l < len(val.energies)])/(len(val.energies)-i*binS))
        listI.append(sum([float(value) for l, value in enumerate(val.intensities) 
                          if i*binS <= l < len(val.intensities)])/(len(val.intensities)-i*binS))
        listEr.append(sum([float(value) for l, value in enumerate(val.errors) 
                          if i*binS <= l < len(val.errors)])/(len(val.errors)-i*binS))


        listE = listE[:-1]
        listI = listI[:-1]
        listEr = listEr[:-1]

        dataBinList.append(qDataBin(listQ[j], np.array(listE), 
                                              np.array(listI), 
                                              np.array(listEr)))    

    return dataBinList


if __name__ == '__main__':

    arg, karg = argParser.argParser(sys.argv)
    dataFile = sys.argv[1]
    dataList = inxBin(sys.argv[1], karg['binS'])

    #with open(dataFile[dataFile.rfind('.')] + '_binned.inx', 'w') as fichier:    
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
    #fichier.write(r)