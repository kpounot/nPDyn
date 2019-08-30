import numpy as np

from collections import namedtuple

from ..ECType import ECType, DataTypeDecorator
from ...fit.ECFit_pseudoVoigtModel import ECFit, fitFunc 



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = fitFunc
        self.params     = None
        self.paramsNames = ["normF", "S", "lorW", "gauW", "shift", "bkgd"] #_For plotting purpose


    def fit(self):
        self.params = ECFit(self.data)
