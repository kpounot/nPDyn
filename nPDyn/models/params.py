"""The module contains a Parameter class to be used with the Model class.

"""

from copy import deepcopy

from collections import OrderedDict, namedtuple

import numpy as np


def pTuple(value=1, bounds=(-np.inf, np.inf), fixed=False, error=0.0):
    """Helper function to create a namedtuple with default values."""
    t = namedtuple("param", "value bounds fixed error")
    _checkTupleEntries(value, bounds, fixed, error)
    return t(value, bounds, fixed, error)


def _checkTupleEntries(
    value=1, bounds=(-np.inf, np.inf), fixed=False, error=0.0
):
    """Check for entries in the 'pTuple'."""
    # check value attribute
    if isinstance(value, list):
        if not all(isinstance(ele, (int, float)) for ele in value):
            raise ValueError(
                "Elements in parameter value should be integer or float."
            )
    elif isinstance(value, np.ndarray):
        if not all(isinstance(ele, (int, float)) for ele in value.flatten()):
            raise ValueError(
                "Elements in parameter value should be integer or float."
            )
    else:
        if not isinstance(value, (int, float)):
            raise ValueError("The parameter value should be integer or float.")

    # check bounds attribute
    if not isinstance(bounds, (list, tuple)):
        raise ValueError(
            "Parameter bounds should be a 2-tuple or 2-list of min and max."
        )
    else:
        for side in bounds:
            if not isinstance(side, (int, float)):
                raise ValueError(
                    "Parameter bounds should contain only integers or floats."
                )

    # check fixed attribute
    if not isinstance(fixed, bool):
        raise ValueError("Parameter 'fixed' should be either True or False.")

    # check error attribute
    if isinstance(error, list):
        if not all(isinstance(ele, (float, int)) for ele in error):
            raise ValueError(
                "Elements in parameter error should be integer or float."
            )
    elif isinstance(error, np.ndarray):
        if not all(isinstance(ele, (float, int)) for ele in error.flatten()):
            raise ValueError(
                "Elements in parameter error should be integer or float."
            )
    else:
        if not isinstance(error, (int, float)):
            raise ValueError("The parameter error should be integer or float.")


class Parameters(OrderedDict):
    """A parameter class that handles names, values and bounds.

    Parameters
    ----------
    params : dict of dict
        A dictionary of parameter names, each being associated with a
        namedtuple containing the 'value', the 'bounds', the
        'fixed', and the 'error' attributes.
    kwargs : keywords
        Additional keywords argument to set parameter names, values
        (and possibly bounds and fixed attributes).
        Can override params too.

    """

    def __init__(self, params=None, **kwargs):
        super().__init__()

        if params is None:
            params = {}

        self.update(**params)

        if kwargs is not None:
            for key, val in kwargs.items():
                if isinstance(val, (list, tuple)):
                    self[key] = pTuple(*val)
                elif isinstance(val, dict):
                    self[key] = pTuple(**val)
                else:
                    self[key] = pTuple(val)

        self._applyBounds()

    def set(self, name, **kwargs):
        """Set a parameter entry with given attributes in 'kwargs'.

        Parameters
        ----------
        name : str
            Parameter name to be updated.
        kwargs : dict of float, tuple or namedtuple
            Parameters to be updates with the associated attributes.
            The call should be of the form:

            >>> params.set('amplitude', value=1.2, fixed=True)
            >>> params.set('width', value=2.3, bounds=(0., np.inf))

        """
        # parse kwargs to keep only attributes from pTuple
        kwargs = {
            key: val
            for key, val in kwargs.items()
            if key in ["value", "bounds", "fixed", "error"]
        }

        _checkTupleEntries(**kwargs)

        if name in self.keys():
            self[name] = self[name]._replace(**kwargs)
        else:
            self[name] = pTuple(**kwargs)

        self._applyBounds()

    def update(self, **kwargs):
        """Update the parameters."""
        for key, val in kwargs.items():
            if isinstance(val, type(pTuple)):
                self[key] = val
                self._applyBounds()
            elif isinstance(val, (list, tuple)):
                self[key] = pTuple(*val)
                self._applyBounds()
            elif isinstance(val, dict):
                self.set(key, **val)
            else:
                self.set(key, **{"value": val})

    def _applyBounds(self):
        """Apply bounds on parameter values."""
        for key, entry in self.items():
            pMin, pMax = entry.bounds
            if isinstance(entry.value, (list, np.ndarray)):
                newVal = np.array(entry.value)
                shape = newVal.shape
                for val in newVal.flatten():
                    val = pMin if val < pMin else val
                    val = pMax if val > pMax else val
                self[key] = self[key]._replace(value=newVal.reshape(shape))
            else:  # assume int or float
                newVal = pMin if entry.value < pMin else entry.value
                newVal = pMax if entry.value > pMax else entry.value
                self[key] = self[key]._replace(value=newVal)

    def _paramsToList(self):
        """Converts the dictionary of parameters to a list of values.

        Do not return the parameters for which the 'fixed' entry is True.

        Returns
        -------
        A 2-tuple in which the first entry contains the values and the
        second the associated bounds.

        """
        params = []
        bounds = []
        for key, val in self.items():
            if not val.fixed:
                if isinstance(val.value, (list, np.ndarray)):
                    params.append(np.array(val.value).flatten())
                    size = val.value.size
                else:  # assumes a integer or float
                    params.append(np.array([val.value]))
                    size = 1

                if val.bounds is not None:
                    [bounds.append(val.bounds) for i in range(size)]
                else:
                    [bounds.append(val.bounds) for i in range(size)]

        params = np.concatenate(params)

        return params, bounds

    def _listToParams(self, pList, errList=None):
        """Use the given list to convert a list of parameters to
        a dictionary similar to the current one.

        """
        params = deepcopy(self)
        pList = list(pList)
        if errList is not None:
            errList = list(errList)
        else:
            errList = list(np.zeros_like(pList))

        for key, val in params.items():
            if not val.fixed:
                if isinstance(val.value, np.ndarray):
                    valShape = val.value.shape
                    valSize = val.value.size
                    newVal = [pList.pop(0) for idx in range(valSize)]
                    newErr = [errList.pop(0) for idx in range(valSize)]
                    newVal = np.array(newVal).reshape(valShape)
                    params.set(key, value=newVal, error=newErr)

                elif isinstance(val.value, list):
                    valSize = len(val)
                    newVal = [pList.pop(0) for idx in range(valSize)]
                    newErr = [errList.pop(0) for idx in range(valSize)]
                    params.set(key, value=newVal, error=newErr)

                else:  # assumes integer or float
                    params.set(key, value=pList.pop(0), error=errList.pop(0))

        return params

    def __repr__(self):
        """Fancy representation for the class."""
        out = ""
        for key, val in self.items():
            out += "\n" + 20 * "_" + key + ":\n"
            for pKey, pVal in val._asdict().items():
                if isinstance(pVal, np.ndarray):
                    pVal = np.copy(pVal)
                    pVal = np.array2string(
                        pVal, prefix="{key}: ".format(key=pKey), threshold=10
                    )
                out += "{key}: {val}".format(key=pKey, val=pVal)
                out += "\n"

        return out
