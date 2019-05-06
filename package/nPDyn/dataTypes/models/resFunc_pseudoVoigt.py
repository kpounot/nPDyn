import numpy as np

from collections import namedtuple

from ..resType import ResType, DataTypeDecorator
from ...fit import resFit_pseudoVoigtModel as resModel 



class Model(DataTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(dataType)

        self.model      = resModel.resFunc
        self.params     = None
        self.paramsNames = ["normF", "S", "lorW", "gauW", "shift", "bkgd"] #_For plotting purpose


    def fit(self, p0=None, bounds=None):
        self.params = resModel.resFit(self.data, p0, bounds)
