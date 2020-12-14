"""This module provides a class to handle workspace in Mantid."""

from functools import wraps

from dateutil.parser import parse as tparse

import numpy as np

try:
    from mantid.api import WorkspaceGroup, MatrixWorkspace

    _FOUND_MANTID = True
except ImportError:
    _FOUND_MANTID = False


def validate_mantid(cls):
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if not _FOUND_MANTID:
            print(
                "Cannot import Mantid API. Please verify your Mantid "
                "installation or your PATH variable."
            )
            return
        else:
            return cls(*args, **kwargs)

    return wrapper


@validate_mantid
class WorkspaceHandler:
    """A class to handle Mantid workspaces.

    Parameters
    ----------
    ws : mantid MatrixWorkspace or group of MatrixWorkspaces.
        Mantid Workspace from which the data are to be read.
    fws : bool
        Whether the data comes from fixed-window scans (FWS) or not.
    obsName : {'time', 'temperature'}
        For QENS data, observable to be used when multiple MatrixWorkspace
        are loaded.

    """

    _mutableKeys = [
        "name",
        "qVals",
        "qIdx",
        "times",
        "temps",
        "observable",
        "observable_name",
    ]
    _immutableKeys = ["energies", "intensities", "errors", "norm"]

    def __init__(self, ws, fws=False, obsName="time"):
        self._name = ""
        self._energies = []
        self._qVals = []
        self._qIdx = []
        self._temps = []
        self._times = []
        self._observable = []
        self._observable_name = ""
        self._norm = False

        self._ws = ws
        self._fws = fws
        self._obsName = obsName

        if isinstance(ws, WorkspaceGroup):
            for i in range(ws.getNumberOfEntries()):
                self._getWorkspaceInfo(ws[i])
            for idx, t in enumerate(self._times):
                self._times[idx] = (t - self._times[0]).total_seconds() / 3600
        elif isinstance(ws, MatrixWorkspace):
            self._ws = ws
            self._getWorkspaceInfo(ws)
            for idx, t in enumerate(self._times):
                self._times[idx] = (t - self._times[0]).total_seconds() / 3600

    def _getWorkspaceInfo(self, ws):
        """Helper function to extract useful information from a workspace."""
        self._name = ws.getRun()["subtitle"].value
        wavelength = ws.getRun()["wavelength"].value

        qVals = ws.getAxis(1).extractValues()
        qCaption = ws.getAxis(1).getUnit().caption()
        self._qVals = qVals
        if qCaption == "Scattering angle":
            self._qVals = 4 * np.pi * np.sin(np.pi * qVals / 360) / wavelength
        if qCaption == "Q2":
            self._qVals = np.sqrt(qVals)
        self._qIdx = np.arange(self._qVals.size)

        time = ws.getRun()["start_time"].value.split(",")[0]
        self._times = np.append(self._times, tparse(time))
        self._temps = np.append(
            self._temps, np.mean(ws.getRun()["sample.temperature"].value)
        )

        xaxis = ws.getAxis(0)
        xcaption = xaxis.getUnit().caption()

        if self._fws:
            self._observable = ws.extractX()
            self._observable_name = xcaption
            self._energies = np.append(
                self._energies,
                ws.getRun()["Doppler.maximum_delta_energy"].value,
            )
        else:
            self._observable = (
                self._times if self._obsName == "time" else self._temps
            )
            self._observable_name = self._obsName
            self._energies = ws.extractX

    @property
    def intensities(self):
        """Accessor for 'intensities' attribute."""
        if isinstance(self._ws, WorkspaceGroup):
            out = []
            for i in range(self._ws.getNumberOfEntries()):
                out.append(self._ws[i].extractY())
            out = np.array(out)
            if self._fws:
                out = out.T
        else:
            out = self._ws.extractY()[np.newaxis, :, :]

        return out[:, self._qIdx]

    @property
    def errors(self):
        """Accessor for 'errors' attribute."""
        if isinstance(self._ws, WorkspaceGroup):
            out = []
            for i in range(self._ws.getNumberOfEntries()):
                out.append(self._ws[i].extractE())
            out = np.array(out)
            if self._fws:
                out = out.T
        else:
            out = self._ws.extractE()[np.newaxis, :, :]

        return out[:, self._qIdx]

    @property
    def energies(self):
        """Accessor for 'energies' attribute."""
        if self._fws:
            out = self._energies
        else:
            out = self._energies()[0] * 1e3
            # match the shapes of intensities and errors
            out = out[1:]

        return out

    @property
    def temps(self):
        """Accessor for 'temps' attribute."""
        out = self._temps
        return out

    @property
    def times(self):
        """Accessor for 'times' attribute."""
        out = self._times
        return out

    @property
    def name(self):
        """Accessor for 'name' attribute."""
        out = self._name
        return out

    @property
    def qVals(self):
        """Accessor for 'qVals' attribute."""
        out = self._qVals
        return out

    @property
    def qIdx(self):
        """Accessor for 'qIdx' attribute."""
        out = self._qIdx
        return out

    @property
    def observable(self):
        """Accessor for 'observable' attribute."""
        out = self._observable
        return out

    @property
    def observable_name(self):
        """Accessor for 'observable_name' attribute."""
        out = self._observable_name
        return out

    @property
    def norm(self):
        """Accessor for the 'norm' attribute."""
        return self._norm

    def copy(self):
        """Return a copy of the WorkspaceHandler."""
        out = WorkspaceHandler(self._ws, self._fws)

        out._name = self.name
        out._qVals = self.qVals
        out._qIdx = self.qIdx
        out._temps = self.temps
        out._times = self.times
        out._observable = self.observable
        out._observable_name = self.observable_name

        return out

    def _replace(self, **kwargs):
        """This method partly reproduces the bahvior of `namedtuple._replace`
        method.

        """
        out = self.copy()

        for key in kwargs.keys():
            if key in self._mutableKeys:
                out.__setattr__("_" + key, kwargs[key])
            elif key in self._immutableKeys:
                print(
                    "Attribute %s cannot be changed with the "
                    "WorkspaceHandler class as it is read from "
                    "the Mantid workspace on each request.\n"
                    "Please use Mantid API to modify it." % key
                )
                continue
            else:
                raise AttributeError(
                    "WorkspaceHandler object has no " "attribute %s" % key
                )

        return out
