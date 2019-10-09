''' This module takes the sys.argv list which is passed to the script
    and extract the listed arguments (usually the files to process) 
    and the keyword arguments which are identified by the following 
    format: argName=argValue.
    The function returns a list and a dictionary for the listed arguments and 
    the keyword arguments respectively. '''

import re
import sys

def argParser(argList):
    
    arg = []
    karg = dict()

    for i, val in enumerate(argList):
        if re.search('=', val):
            if val == '=':
                arg.pop(len(arg)-1)
                argKey = argList[i-1].strip()
                argVal = argList[i+1].strip()
                argList.remove(argList[i+1])
                karg[argKey] = argVal
            else:
                argKey = val[:val.find('=')].strip()
                argVal = val[val.find('=')+1:].strip()
                karg[argKey] = argVal
        else:
            arg.append(val)

    return arg, karg

if __name__ == '__main__':
    
    argList, kargList = argParser(sys.argv)
    print(argList)
    print(kargList)
