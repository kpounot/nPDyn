import numpy as np

from collections import namedtuple

from .baseType import BaseType


class D2OType(BaseType):

    def __init__(self, fileName=None, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)



    def substractEC(self, ECFunc, subFactor):
        """ This method can be used to substract empty cell data from QENS data.
            Empty cell signal is rescaled using the given 'subFactor' prior to substraction. """

        #_Substracting the empty cell intensities from experimental data
        self.data = self.data._replace(intensities = self.data.intensities - ECFunc)


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)
 


class DataTypeDecorator(D2OType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)


