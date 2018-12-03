import sys
import argparse

from PyQt5.QtWidgets import QApplication 

from package.Dataset import Dataset

app = QApplication(sys.argv)

#_Defining options for nPDyn call
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--QENS", nargs='*', 
                help="List of files corresponding to sample Quasi-Elastic Neutron Scattering (QENS) data")
parser.add_argument("-f", "--FWS", nargs='*',
                help="List of files corresponding to sample Fixed-Window Scan (FWS) data")
parser.add_argument("-res", "--resolution", nargs='*', 
                                help="Specify the file(s) to be used for resolution function fitting.")
parser.add_argument("-ec", "--empty-cell", nargs='?',
                                help="Specify the file containing QENS empty cell data")
parser.add_argument("-d", "--D2O", nargs='?', help="Specify the file containing QENS D2O data")


args = parser.parse_args()

data = Dataset(args.QENS, args.FWS, args.empty_cell, args.resolution, args.D2O)


sys.exit(app.exec_())



