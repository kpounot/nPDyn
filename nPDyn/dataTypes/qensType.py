"""Specialized type for quasi-elastic neutron scattering (QENS) data."""

from nPDyn.dataTypes.baseType import BaseType


class QENSType(BaseType):
    def __init__(
        self,
        fileName,
        data=None,
        rawData=None,
        resData=None,
        D2OData=None,
        ECData=None,
        model=None,
    ):
        """This class inherits from :class:`baseType` class.

        No extra or redefined methods, compared to
        :class:`baseType` are present for now.

        """
        super().__init__(
            fileName, data, rawData, resData, D2OData, ECData, model
        )
