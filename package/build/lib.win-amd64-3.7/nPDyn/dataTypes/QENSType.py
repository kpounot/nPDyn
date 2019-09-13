import numpy as np

from collections import namedtuple

from .baseType import BaseType


class QENSType(BaseType):

    def __init__(self, fileName, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


        self.BH_iter    = 100
        self.disp       = True

        self.globalFit  = True


class DataTypeDecorator(QENSType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)

