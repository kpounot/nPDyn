import numpy as np

from collections import namedtuple

from .baseType import BaseType


class ECType(BaseType):

    def __init__(self, fileName, data=None):
        super().__init__(fileName, data)


