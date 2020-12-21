.. image:: https://readthedocs.org/projects/npdyn/badge/?version=latest
    :target: https://npdyn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://circleci.com/gh/kpounot/nPDyn.svg?style=svg
    :target: https://circleci.com/gh/kpounot/nPDyn

.. image:: https://codecov.io/gh/kpounot/nPDyn/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/kpounot/nPDyn

.. image:: https://app.codacy.com/project/badge/Grade/c7300a6a87b54eebb887c6acadb4672e
    :target: https://www.codacy.com/gh/kpounot/nPDyn/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kpounot/nPDyn&amp;utm_campaign=Badge_Grade



nPDyn
=====
nPDyn is a Python based API for analysis of neutron backscattering data.

The API aims at providing a lightweight, user-friendly and modular tool
to process and analyze quasi-elastic neutron scattering (QENS) and
fixed-window scans (FWS) obtained with backscattering spectroscopy.

nPDyn can be used in combination with other software for neutron data analysis
such as `Mantid <https://www.mantidproject.org>`_. The API provides an interface
to Mantid workspaces for that.

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
For installation within your python framework, use:

.. code:: bash

    make install

or

.. code:: bash

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
The nPDyn API is organized around a ``dataset.Dataset`` class.
This class has a ``Dataset.dataList`` attribute used to store the experimental
data. Each measurement in ``Dataset.dataList`` consists in a class that
inherits from ``baseType.BaseType``.

In a neutron backscattering experiment, there is not only the measurement of
samples but also some calibration measurements like vanadium, empty cell
and solvent signal (often D2O).
The ``dataset.Dataset`` can handle these in the special attributes
``Dataset.resData``, ``Dataset.ECData`` and ``Dataset.D2OData``, respectively.
Each data in ``Dataset.dataList`` can have some calibration data associated
with it in the ``BaseType.resData``, ``BaseType.ECData``
and ``BaseType.D2OData`` attributes.

In the current state of nPDyn, only one file can be loaded for empty cell
and solvent calibration measurements. For the resolution function, the
``Dataset.resData`` attribute is actually a list that can contain
several measurements. The reason for this is that the resolution
function can be obtained by measuring the samples at very low temperature
instead of using a single vanadium measurement. Hence, each data in
``Dataset.dataList`` can be associated with a resolution measurement
in ``Dataset.resData``.

The aforementioned structure of the API is sketched below for two samples,
measured at temperatures t1 and t2 each,
with a measurement for the resolution function at 10K for each sample,
one measurement of empty cell and one of D2O background:

.. image:: docs/fig/sketch_structure_01.png
    :width: 600

Details regarding importation of data are available in the documentation
section of the documentation.

The ``baseType.BaseType`` base class and its derivatives
``qensType.QENSType`` and ``fwsType.FWSType`` contain
several methods for data processing (see *Process data* in documentation) and
fitting (see *Fit data* section in documentation).
In addition the class ``dataset.Dataset`` contains some shortcut
methods to apply data processing and fitting algorithm quickly on the
sample and calibration data. It also contains plotting methods to examine
data and the fitted model and its optimized parameters.

Importantly, nPDyn provides versatile tools for model building and fitting
to the data. See the section *Fit data* in documentation for details.
