"""nPDyn is a Python API for neutron incoherent scattering data analysis.

The API includes features for data files parsing, including .inx files
and .nxs files as generated by Mantid. In addition, raw data files from
IN16B at the ILL can be read and processed directly.

It also includes modelling and fitting capabilities that are based on
the lmfit-py library.

Finally, several methods allow for easy plotting of the data and
the fit results.

"""
from nPDyn.sample import Sample
from nPDyn.plot import plot

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
