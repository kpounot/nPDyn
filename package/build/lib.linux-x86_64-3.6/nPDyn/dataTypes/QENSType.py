"""

Classes
^^^^^^^

"""

import numpy as np

from collections import namedtuple

from nPDyn.dataTypes.baseType import BaseType


class QENSType(BaseType):
    """ This class inherits from :class:`baseType` class.

        No extra or redefined methods, compared to :class:`baseType` are present for now.

        Additional attributes are present for Scipy basinhopping routine:

            - BH_iter - number of basinhopping iteration to perform (default 100)
            - disp    - if True, print basinhopping progression while it's running (default True)

    """

    def __init__(self, fileName, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


        self.BH_iter    = 100
        self.disp       = True


class DataTypeDecorator(QENSType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)

