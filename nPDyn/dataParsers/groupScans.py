"""Helper module to group the scans together."""

from collections import namedtuple

import h5py

import numpy as np

from nPDyn.dataParsers.stringParser import parseString, arrayToString


def printGroup(groups):
    """Fancy printing for scan groups."""
    out = ""
    for group in groups:
        for key, val in group.items():
            if key in ("title", "subtitle"):
                out += str(val[0].decode("utf-8")) + "\n"
            elif key == "runNumber":
                out += "\tscans: "
                out += arrayToString(val)
                out += "\n"
            else:
                out += "\t%s: %s\n" % (key, str(val[0]))
        out += "\n"

    return out


def groupScansIN16B(scans, speedTol=0.02, groupFWS=True):
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
    groupFWS : bool
        If True, all scans having a value for doppler speed
        different from 4.5 will be considered as FWS and the
        values for 'setTemperature', 'setField', 'setPressure',
        'doppler delta energy' and 'doppler speed' won't be compared.

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
                equals &= val["wavelength"][0] == prop["wavelength"][0]
                if prop["doppler_delta_energy"] < 15:
                    if not groupFWS:
                        equals &= (
                            val["setTemperature"][0]
                            == prop["setTemperature"][0]
                        )
                        equals &= val["setField"][0] == prop["setField"][0]
                        equals &= (
                            val["setPressure"][0] == prop["setPressure"][0]
                        )
                        equals &= (
                            val["doppler_delta_energy"][0]
                            == prop["doppler_delta_energy"][0]
                        )
                        equals &= np.allclose(
                            val["doppler_speed"],
                            prop["doppler_speed"],
                            speedTol,
                        )
                    else:
                        equals &= prop["doppler_delta_energy"][0] < 15
                        equals &= val["doppler_delta_energy"][0] < 15
                else:
                    equals &= (
                        val["setTemperature"][0] == prop["setTemperature"][0]
                    )
                    equals &= val["setField"][0] == prop["setField"][0]
                    equals &= val["setPressure"][0] == prop["setPressure"][0]
                    equals &= (
                        val["doppler_delta_energy"][0]
                        == prop["doppler_delta_energy"][0]
                    )
                    equals &= np.allclose(
                        val["doppler_speed"], prop["doppler_speed"], speedTol
                    )
                if equals:
                    val["runNumber"] = np.append(
                        val["runNumber"], prop["runNumber"]
                    )
                    val["setTemperature"] = prop["setTemperature"]
                    val["setField"] = prop["setField"]
                    val["setPressure"] = prop["setPressure"]
                    break
            if not equals:
                out.append(prop)

    return out


def _readPropertiesIN16B(scan):
    """Extract the properties from a HDF5 file generated on IN16B."""
    try:
        f = h5py.File(scan, "r")
    except Exception as e:
        print("Error with scan %s:\n" % scan)
        print(e)
        return
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

    if not np.isfinite(pMap["doppler_delta_energy"]):
        pMap["doppler_delta_energy"] = np.array([-1])

    f.close()

    return pMap
