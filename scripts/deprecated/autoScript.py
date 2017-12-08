import sys, os
import re


def autoScript(fileName):
    fileName = fileName

    if re.search('.inx', fileName):
        with open(fileName, 'r') as openFile:
            openFile = openFile.read().splitlines()
            if float(openFile[0].split()[0]) < 2000:
                return 'inxPlotQENS.py'
            else:
                return 'inxPlotENS.py'

    elif re.search('.y08', fileName):
        return 'y08Convert.py'

    elif re.search('fasta', fileName):
        return 'protParser.py'

    else:
        raise Exception('File format not recognized')
