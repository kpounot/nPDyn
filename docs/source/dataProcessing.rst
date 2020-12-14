Process data
============
nPDyn provides several data processing methods, which includes
binning, normalization, scaling, empty cell correction,
Paalman-Pings coefficient calculation and detector selection.

These are described below.

Binning
^^^^^^^
The dataset can be binned along two axes, energies and observable.
This can be done using the method :py:meth:`dataset.Dataset.binAll`,
:py:meth:`dataset.Dataset.binDataset` or :py:meth:`dataset.Dataset.binResData`
(for resolution data only).

Here is an example code with quasi-elastic neutron scattering (QENS) data:

>>> import nPDyn
>>> exp = nPDyn.Dataset()
>>> exp.importFiles(QENSFiles=['myData.nxs'])
>>> exp.dataList[0].data.intensities.shape
(1, 18, 1004)
>>> # 1 observable, 18 detectors/q values and 1004 energy transfers
>>> exp.binAll(5, axis='energies')  # bins of 5 points on the energy axis
>>> exp.dataList[0].data.intensities.shape
(1, 18, 200)
>>> exp.dataList[0].data.energies.shape
(200,)

Use ``exp.binAll(5, axis='observable')`` to bin along the observable axis.

Normalization
^^^^^^^^^^^^^
Normalization of data can be done by dividing the by integration
of themselves, of vanadium or of data at low temperature.
These normalizations can be performed using the
:py:meth:`dataset.Dataset.normalize_usingSelf`,
:py:meth:`dataset.Dataset.normalize_usingResFunc`, or
:py:meth:`dataset.Dataset.normalize_usingLowTemp`, respectively.

The following:

>>> import nPDyn
>>> exp = nPDyn.Dataset(
...     QENSFiles=['myData01.nxs', 'myData02.nxs', 'myData0".nxs],
...     resFiles=['vanadium.nxs'])
>>> exp.normalize_usingResFunc(0, 1)

will apply normalization using the vanadium data on the first and second
entries (index 0 and 1) in ``exp.dataList``, while writing simply

>>> exp.normalize_usingResFunc()

will apply normalization to all data in ``exp.dataList``.
The same apply for the other normalization methods.


Scaling
^^^^^^^
The user can also apply a scaling factor to the data.
To this end, use:

>>> import nPDyn
>>> exp = nPDyn.Dataset(
...     QENSFiles['myData01.nxs', myData02.nxs', 'myData03.nxs'])
>>> exp.scaleData(0.5, 0, 1)

to apply a scaling factor of 0.5 on 'myData01.nxs' and 'myData02.nxs'.

Background corrections
^^^^^^^^^^^^^^^^^^^^^^
nPDyn provides two types of background corrections, namely empty cell
subtraction (see :py:meth:`dataset.Dataset.subtract_EC`),
and paalman-pings coefficients
(see :py:meth:`dataset.Dataset.absorptionCorrection`).

For instance:

>>> import nPDyn
>>> exp = nPDyn.Dataset(
...     QENSFiles['myData01.nxs', myData02.nxs', 'myData03.nxs'],
...     ECFile='empty_cell.nxs',
...     D2OFile='D2O.nxs')
>>> exp.subtract_EC(0,2, subFactor=0.9, subD2O=True, useModel=False)

will scale the empty cell data by 0.9 and subtract it from 'myData01.nxs'
and 'myData03.nxs' as well as from the D2O data.


Detector selection
^^^^^^^^^^^^^^^^^^
The user will very likely want to restrain the analysis to a specific
range of momentum transfers q or to discard some detectors.
To this end, the :py:class:`dataset.Dataset` provides the following
methods, :py:meth:`dataset.Dataset.setQRange`
and :py:meth:`dataset.Dataset.discardDetectors`.

Reset data
^^^^^^^^^^
All the data can be restored to their original state (the one at importation)
using :py:meth:`dataset.Dataset.resetData` or
:py:meth:`dataset.Dataset.resetAll`.
