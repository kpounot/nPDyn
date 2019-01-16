import numpy as np

from collections import namedtuple

from .baseType import BaseType


class ECType(BaseType):

    def __init__(self, fileName=None, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


