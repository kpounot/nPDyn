"""Definitions and geometries for IN16B at the ILL."""


import numpy as np

from collections import namedtuple


diffDetectors = {
    "number": 8,
    "size": 685,
    "angles": np.array([12, 31, 50, 69, 88, 107, 126, 145]),
    "radius": 1845,
    "nbrPixels": 256,
}


def getDiffDetXAxis(wavelength):
    """Helper function to get the momentum transfer values for diffraction."""
    halfAngleRange = (
        np.arcsin(diffDetectors["size"] / (2 * diffDetectors["radius"]))
        / (2 * np.pi)
        * 360
    )

    startAngle = diffDetectors["angles"][0] - halfAngleRange
    lastAngle = diffDetectors["angles"][-1] + halfAngleRange
    axisSize = diffDetectors["number"] * diffDetectors["nbrPixels"]
    step = (lastAngle - startAngle) / axisSize
    angles = np.arange(startAngle, lastAngle, step) / 360 * np.pi
    angles = 4 * np.pi / wavelength * np.sin(angles)

    return angles
