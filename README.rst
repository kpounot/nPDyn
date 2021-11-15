.. image:: https://readthedocs.org/projects/npdyn/badge/?version=latest
    :target: https://npdyn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://circleci.com/gh/kpounot/nPDyn.svg?style=svg
    :target: https://circleci.com/gh/kpounot/nPDyn

.. image:: https://codecov.io/gh/kpounot/nPDyn/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/kpounot/nPDyn

.. image:: https://app.codacy.com/project/badge/Grade/c7300a6a87b54eebb887c6acadb4672e
    :target: https://www.codacy.com/gh/kpounot/nPDyn/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kpounot/nPDyn&amp;utm_campaign=Badge_Grade

|

.. image:: https://img.shields.io/pypi/v/nPDyn.svg
   :target: https://pypi.org/project/nPDyn

.. image:: https://zenodo.org/badge/97102616.svg
   :target: https://zenodo.org/badge/latestdoi/97102616



nPDyn
=====
nPDyn is a Python based API for analysis of neutron backscattering data.

The API aims at providing a lightweight, user-friendly and modular tool
to process and analyze quasi-elastic neutron scattering (QENS) and
fixed-window scans (FWS) obtained with backscattering spectroscopy.

nPDyn can be used in combination with other software for neutron data analysis
such as `Mantid <https://www.mantidproject.org>`_.

An important feature of nPDyn is the modelling interface, which is designed
to be highly versatile and intuitive for multidimensional dataset with global
and non-global parameters.
The modelling in nPDyn is provided by builtin classes,
``params.Parameters``, ``model.Model`` and
``model.Component``.
nPDyn provides also some helper functions to use
`lmfit <https://lmfit.github.io/lmfit-py/>`_ as modelling backend.
See *Fit data* section in documentation for details.

Eventually, some plotting methods are available to examine processed data,
model fitting and optimized parameters.


Installation:
-------------
Prior to building on Windows, the path to Gnu Scientific Library (GSL) should
be given in setup.cfg file (required by libabsco)

If not, the package can still be installed but paalman-ping corrections won't
work.


Unix and Windows
^^^^^^^^^^^^^^^^
For installation within your python framework:

- with pip:

.. code:: bash

    python3 -m pip install nPDyn

- with source code:

.. code:: bash

    git clone https://github.com/kpounot/nPDyn npdyn
    cd npdyn
    python3 setup.py install


Full documentation
------------------
See https://npdyn.readthedocs.io/en/latest/


Support
-------
A `google group <https://groups.google.com/g/npdyn>`_ is available for any
question, discussion on features or comment.

In case of bugs or obvious change to be done in the code use GitHub Issues.


Contributions
-------------
See `contributing <https://github.com/kpounot/nPDyn/blob/master/contributing.rst>`_.


Getting started
---------------
The nPDyn API is organized around a `Sample` class.
This class inherits from the NumPy ndarray class with some extra
features added, such as neutron scattering-specific attributes, binning,
data correction algorithm, automatic error propagation and data fitting.

In a neutron backscattering experiment, there is not only the measurement of
samples but also some calibration measurements like vanadium, empty cell
and solvent signal (often :math:`\rm D_2O`). Some methods of the
`Sample` class can be used to perform normalization or
absorption correction using the dataset corresponding to vanadium
or empty cell, respectively. These calibration dataset can be used also
in the `fit` function to automatically add a background or perform
a convolution with the resolution function.

Details regarding importation of data are available in the `dataImport`
section of the documentation.

Importantly, nPDyn provides versatile tools for model building and fitting
to the data. See the section `dataFitting` for details.

Finally, a `plot` method is provided for easy visualisation
of the data and the fit results.
