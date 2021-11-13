Process data
============
nPDyn provides several data processing methods, which includes
binning, normalization, scaling, empty cell correction,
Paalman-Pings coefficient calculation and detector selection.

These are described below.

Arithmetic operations
^^^^^^^^^^^^^^^^^^^^^
The :py:class:`sample.Sample` is essentially a NumPy array, so
arithmetic operations can be used as for any array.

>>> from nPDyn.dataParsers import processNexus
>>> sample1 = processNexus('mySample1.nxs')
>>> sample2 = processNexus('mySample2.nxs')
>>> corrected_sample = sample1 - 0.95 * sample2

Again, the errors are automatically propagated for most of the commonly
used operators (addition, subtraction multiplication, division,
exponentiation, logarithm, power, square, square root).

Binning
^^^^^^^
The dataset can be binned along any axis.
This can be done using the method :py:meth:`sample.Sample.bin`.

Here is an example code with quasi-elastic neutron scattering (QENS) data:

>>> from nPDyn.dataParsers import processNexus
>>> sample = processNexus('myData.nxs')
>>> sample.shape
(1, 18, 1004)
>>> # 1 observable, 18 detectors/q values and 1004 energy transfers
>>> sample = sample.bin(5, axis=2)  # bins of 5 points on the energy axis
>>> sample.shape
(1, 18, 200)
>>> sample.energies.shape
(200,)

Normalization
^^^^^^^^^^^^^
Normalization of data can be done by dividing by the integration
of themselves, of vanadium or of data at low temperature.

The following:

>>> from nPDyn.dataParsers import processNexus
>>> sample = processNexus('myData.nxs')
>>> sample = sample.normalize()

will apply normalization using the integration of the 'sample' dataset.

Using

>>> from nPDyn.dataParsers import processNexus
>>> sample = processNexus('myData.nxs')
>>> vanadium = processNexus('vanadium.nxs')
>>> sample = sample.normalize(vanadium)

The signal of the vanadium will be integrated and used for normalization.
If a fitted model exists for the vanadium, it will be used instead of the
experimental data.

Background corrections
^^^^^^^^^^^^^^^^^^^^^^
The correction of background, often using empty cell signal, can be
done either using simple arithmetic operators or using the
:py:meth:`sample.Sample.absorptionCorrection` method.

For instance,

>>> from nPDyn.dataParsers import processNexus
>>> sample = processNexus('mySample.nxs')
>>> empty_cell = processNexus('empty_cell.nxs')
>>> sample = sample.absorptionCorrection(
...     empty_cell,
...     canScaling=0.95,
...     canType='tube',
...     useModel=False
... )

will computes the Paalman-Ping coefficient for a tubular sample
holder, scale the empty cell data provided factor and apply
the absorption correction to the dataset.


Selection of data range
^^^^^^^^^^^^^^^^^^^^^^^
The user will very likely want to restrain the analysis to a specific
range of momentum transfers q or observable values.
To this end, some self-explaining methods are provided to select a
range based on values instead of indices:

>>> from nPDyn.dataParsers import processNexus
>>> sample = processNexus('mySample.nxs')
>>> sample.shape
(10, 18, 1004)
>>> sample = sample.get_q_range(0.3, 1.7)
>>> sample = sample.get_observable_range(280, 320)
>>> sample.shape
(4, 14, 1004)
