"""Helper module to group the scans together."""

from collections import namedtuple

import h5py

import numpy as np

from nPDyn.dataParsers.stringParser import parseString, arrayToString


def printGroup(groups):
    """Fancy printing for scan groups."""
    for group in groups:
        for key, val in group.items():
            if key in ("title", "subtitle"):
                print(str(val[0].decode("utf-8")))
            elif key == "runNumber":
                print("\tscans: ", end="")
                print(arrayToString(val))
            else:
                print("\t%s: %s" % (key, str(val[0])))
        print("\n")


def groupScansIN16B(scans, speedTol=0.02):
    """Group the scans together based on properties in the files.

    The following properties are compared and the scans which share the
    same values are grouped together:

        - title
        - subtitle
        - temperature setpoint
        - pressure setpoint
        - field setpoint
        - wavelength
        - doppler delta energy
        - doppler speed

    For the speed values, the argument *speedTol* gives a tolerance
    for two values to be considered equal (except for doppler).

    Parameters
    ----------
    scans : string
        A string given the scans to be grouped.
    speedTol : float, optional
        Tolerance for the'speed' values to be considered equal
        (default 0.02)

    """
    scans = parseString(scans)
    out = []
    for sIdx, scan in enumerate(scans):
        prop = _readPropertiesIN16B(scan)
        if len(out) == 0:
            out.append(prop)
        else:
            for val in out:
                equals = val["title"][0] == prop["title"][0]
                equals &= val["subtitle"][0] == prop["subtitle"][0]
                equals &= val["setTemperature"][0] == prop["setTemperature"][0]
                equals &= val["setField"][0] == prop["setField"][0]
                equals &= val["setPressure"][0] == prop["setPressure"][0]
                equals &= val["wavelength"][0] == prop["wavelength"][0]
                equals &= np.array_equal(
                    val["doppler_delta_energy"][0],
                    prop["doppler_delta_energy"][0],
                    True,
                )
                equals &= np.allclose(
                    val["doppler_speed"], prop["doppler_speed"], speedTol
                )
                if equals:
                    val["runNumber"] = np.append(
                        val["runNumber"], prop["runNumber"]
                    )
                    break
            if not equals:
                out.append(prop)

    return out


def _readPropertiesIN16B(scan):
    """Extract the properties from a HDF5 file generated on IN16B."""
    f = h5py.File(scan, "r")
    pMap = {
        "title": "entry0/title",
        "subtitle": "entry0/subtitle",
        "runNumber": "entry0/run_number",
        "setTemperature": "entry0/sample/setpoint_temperature",
        "setField": "entry0/sample/setpoint_field",
        "setPressure": "entry0/sample/setpoint_pressure",
        "wavelength": "entry0/wavelength",
        "doppler_delta_energy": "entry0/instrument/Doppler/"
        "maximum_delta_energy",
        "doppler_speed": "entry0/instrument/Doppler/doppler_speed",
    }

    for key, val in pMap.items():
        pMap[key] = f[val][()]

    f.close()

    return pMap
