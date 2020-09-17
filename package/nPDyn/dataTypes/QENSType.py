"""

Classes
^^^^^^^

"""

from nPDyn.dataTypes.baseType import BaseType


class QENSType(BaseType):
    """ This class inherits from :class:`baseType` class.

        No extra or redefined methods, compared to
        :class:`baseType` are present for now.

    """

    def __init__(self, fileName, data=None, rawData=None,
                 resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)



class DataTypeDecorator(QENSType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data,
                         dataType.rawData, dataType.resData,
                         dataType.D2OData, dataType.ECData)
