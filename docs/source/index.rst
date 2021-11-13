.. nPDyn documentation master file, created by
   sphinx-quickstart on Fri Sep 20 17:40:40 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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
:py:class:`params.Parameters`, :py:class:`model.Model` and
:py:class:`model.Component`.
nPDyn provides also some helper functions to use
`lmfit <https://lmfit.github.io/lmfit-py/>`_ as modelling backend.
See :doc:`dataFitting` for details.

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


Getting started
---------------
The nPDyn API is organized around a :py:class:`sample.Sample` class.
This class inherits from the NumPy ndarray class with some extra
features added, such as neutron scattering-specific attributes, binning,
data correction algorithm, automatic error propagation and data fitting.

In a neutron backscattering experiment, there is not only the measurement of
samples but also some calibration measurements like vanadium, empty cell
and solvent signal (often :math:`\rm D_2O`). Some methods of the
:py:class:`sample.Sample` class can be used to perform normalization or
absorption correction using the dataset corresponding to vanadium
or empty cell, respectively. These calibration dataset can be used also
in the `fit` function to automatically add a background or perform
a convolution with the resolution function.

Details regarding importation of data are available in the :doc:`dataImport`
section of the documentation.

Importantly, nPDyn provides versatile tools for model building and fitting
to the data. See the section :doc:`dataFitting` for details.

Finally, a :py:method:`plot.plot` method is provided for easy visualisation
of the data and the fit results.


Documentation
-------------
.. toctree::
    :maxdepth: 2

    dataImport
    dataProcessing
    dataFitting
    dataPlotting
    api/api
    license
    help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
