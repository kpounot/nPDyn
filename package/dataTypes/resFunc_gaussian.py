import numpy as np

from collections import namedtuple

from .baseType import BaseType, BaseTypeDecorator
from ..fit import resFit_gaussianModel as resModel 



class ResFunc_gaussian(BaseTypeDecorator):
    """ This class stores data as resolution function related. It allows to perform a fit using a 
        pseudo-voigt profile as a model for instrument resolution. """

    def __init__(self, dataType):
        super().__init__(self, dataType)

        self.model      = resModel.resFunc
        self.params     = None
        self.paramNames = ["normF", "S", "g0", "g1", "shift", "bkgd"] #_For plotting purpose


    def fit(self):
        self.params = resModel.resFit(self.data)
