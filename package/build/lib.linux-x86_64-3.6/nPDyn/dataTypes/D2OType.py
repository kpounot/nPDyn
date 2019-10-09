"""

Classes
^^^^^^^

"""

import numpy as np

from collections import namedtuple

from nPDyn.dataTypes.baseType import BaseType


class D2OType(BaseType):
    """ This class inherits from :class:`baseType` class.

        No extra or redefined methods, compared to :class:`baseType` are present for now.

    """

    def __init__(self, fileName=None, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)




class DataTypeDecorator(D2OType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)


