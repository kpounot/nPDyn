import sys
import argparse

from PyQt5.QtWidgets import QApplication 

from nPDyn.Dataset import Dataset
from nPDyn.dataTypes.models import *
from nPDyn.MDParser import MDData


#_Defining options for nPDyn call
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--QENS", nargs='*', 
                help="List of files corresponding to sample Quasi-Elastic Neutron Scattering (QENS) data")
parser.add_argument("-f", "--FWS", nargs='*',
                help="List of files corresponding to sample Fixed-Window Scan (FWS) data")
parser.add_argument("-tr", "--TempRamp", nargs='*', 
                help="List of files corresponding to temperature ramp elastic data")
parser.add_argument("-res", "--resolution", nargs='*', 
                                help="Specify the file(s) to be used for resolution function fitting.")
parser.add_argument("-ec", "--empty-cell", nargs='?', 
                                help="Specify the file containing QENS empty cell data")
parser.add_argument("-fec", "--fixed-empty-cell", nargs='?',
                                help="Specify the file containing FWS empty cell data")
parser.add_argument("-tec", "--TempRamp-empty-cell", nargs='?',
                                help="Specify the file containing temperature ramp empty cell data")
parser.add_argument("-d", "--D2O", nargs='?', help="Specify the file containing QENS D2O data")
parser.add_argument("-fd", "--fixed-D2O", nargs='?', help="Specify the file containing FWS D2O data")


args = parser.parse_args()


#_Initialize a first instance of Dataset
data = Dataset(args.QENS, args.FWS, args.TempRamp, args.empty_cell, args.fixed_empty_cell, 
                                        args.TempRamp_empty_cell, args.resolution, args.D2O, args.fixed_D2O)





