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
