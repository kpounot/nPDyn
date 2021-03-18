"""Helper module to parse strings given to raw data importation routines"""

import os

import re

import glob

import numpy as np


def parseString(s):
    """Take a string s and convert it to a useful argument for data parsers"""
    scanList = np.array([])
    if re.match(r"[\/]", s[-1]):
        files = glob.glob(s + "*")
        scanList = [val for val in files if os.path.isfile(val)]
        scanList.sort()
    else:
        path = os.path.dirname(s)
        if len(path) > 0:
            path += "/"
        files, ext = os.path.basename(s).split(".")
        inputList = re.split("[+,;]", files)
        for idx, inp in enumerate(inputList):
            scans = np.array(re.split("[:-]", inp)).astype(int)
            if scans.size > 1:
                scans = np.arange(scans[0], scans[1] + 1, dtype=int)
            scanList = np.concatenate((scanList, scans)).astype(int)
        scanList = [path + str(val) + "." + ext for val in scanList]

    return scanList


def arrayToString(arr):
    """Take an array of scan numbers and generate a string."""
    arr = np.sort(arr)
    out = ""
    scans = []
    continousSerie = []
    for idx, val in enumerate(arr):
        if idx == 0:
            continousSerie.append(val)
        elif val == continousSerie[-1] + 1:
            continousSerie.append(val)
        else:
            scans.append(continousSerie)
            continousSerie = [val]
    scans.append(continousSerie)

    for val in scans:
        if len(val) > 1:
            out += ",%d:%d" % (val[0], val[-1])
        else:
            out += ",%d" % val[0]
    return out.strip(",")
