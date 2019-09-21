"""

Classes
^^^^^^^

"""

import numpy as np

from collections import namedtuple

from nPDyn.dataTypes.baseType import BaseType


class ECType(BaseType):
    """ This class inherits from :class:`baseType` class.

    """

    def __init__(self, fileName=None, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)




    def discardNonElastic(self, deltaE=0.4):
        """ Can be used to set to zero in intensity the region that does not pertain to the elastic
            peak. This can be useful for proper empty cell signal substraction.

            :arg deltaE: energy offset corresponding to the end of the elastic peak, everything that
                         is beyond + or - deltaE will be set to zero. 

        """


        toDiscard = np.where( np.abs(self.data.X) > deltaE)

        newIntensities = self.data.intensities
        newIntensities[:,toDiscard] = 0

        self.data = self.data._replace(intensities = newIntensities)





class DataTypeDecorator(ECType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)
