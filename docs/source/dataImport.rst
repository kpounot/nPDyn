Import data
===========
nPDyn provides various ways to handle data.

The imported data are available within the :class:`dataset.Dataset` class.
This class act as a master class that stores all loaded data and provides
methods for data processing, fitting and plotting.

Sample data are stored in the :py:attr:`Dataset.dataList` attribute.
Resolution data are stored in the :py:attr:`Dataset.resData` attribute.
Similarly, data for empty cell and solvent background are stored
in the :py:attr:`Dataset.ECData` and :py:attr:`Dataset.D2OData`, respectively if
they are obtained from quasi-elastic neutron scattering (QENS) or
in the :py:attr:`Dataset.fECData` and :py:attr:`Dataset.fD2OData` if they are obtained
from fixed-window scans (FWS).

Assignment of calibration data to sample data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each data in ``Dataset.dataList`` can have resolution, empty cell
and solvent background calibration data associated with it in the
``baseType.resData``, ``baseType.ECData`` and ``baseType.D2OData``
attributes.
nPDyn will try to assign calibration data automatically on importation
but the user can still manually assign them.

For instance, a case where there are two samples measured at two different
temperatures. The resolution function is obtained by measuring some vanadium
and empty cell and :math:`\rm D_2O` background are also measured.
Then using the following:

>>> import nPDyn
>>> exp = nPDyn.Dataset(
...     QENSFiles=['sample1_T1.nxs', 'sample1_T2.nxs',
...                'sample2_T1.nxs'; 'sample2_T2.nxs'],
...     resFiles=['vanadium.nxs'],
...     ECFile='empty_cell.nxs',
...     D2OFile='D2O.nxs')

will result automatically in the following structure on import:

.. image:: ../fig/sketch_structure_02.png

Now, let us assume there are four samples, each associated with
one measurement for resolution and again, empty cell
and :math:`\rm D_2O`. Then using:

>>> import nPDyn
>>> exp = nPDyn.Dataset(
...     QENSFiles=['sample1.nxs', 'sample2.nxs',
...                'sample3.nxs', 'sample4.nxs'],
...     resFiles=['resolution1.nxs', 'resolution2.nxs',
...               'resolution3.nxs', 'resolution4.nxs'],
...     ECFile='empty_cell.nxs',
...     D2OFile='D2O.nxs')

will result automatically in the following on import:

.. image:: ../fig/sketch_structure_03.png

To obtain the following structure:

.. image:: ../fig/sketch_structure_01.png

the resolution measurements have to be assigned manually using:

>>> import nPDyn
>>> exp = nPDyn.Dataset(
...     QENSFiles=['sample1_T1.nxs', 'sample1_T2.nxs',
...                'sample2_T1.nxs'; 'sample2_T2.nxs'],
...     resFiles=['measure1.nxs', 'measure2.nxs'],
...     ECFile='empty_cell.nxs',
...     D2OFile='D2O.nxs')
>>> data.dataList[0].resData = data.resData[0]
>>> data.dataList[1].resData = data.resData[0]
>>> data.dataList[2].resData = data.resData[1]
>>> data.dataList[3].resData = data.resData[1]


Access the data values
^^^^^^^^^^^^^^^^^^^^^^
Each imported data consists in a class that inherits from :class:`baseType.BaseType`
class. It has resolution data, empty cell data and D2O background data associated
with it. If all the aforementioned type of data are imported together, nPDyn will
automatically associate resolution, empty cell and D2O background to each
data loaded in ``dataset.Dataset.dataList``.
The class contains several methods for processing and fitting
and also a ``data`` attribute which contains the experimental data.

For a dataset created using ``exp = nPDyn.Dataset()`` and
experimental data loaded in ``exp.dataList[0]``, the following attributes
are available:

- **exp.dataList[0].data.name**,
  The name used in the experiment for the scans.
- **exp.dataList[0].data.energies**,
  The energy transfers measured during the experiment.
- **exp.dataList[0].data.intensities**,
  The intensities/counts measured during the experiment.
- **exp.dataList[0].data.errors**,
  The errors associated with the intensities.
- **exp.dataList[0].data.qVals**,
  The momentum transfer *q* values measured during the experiment.
- **exp.dataList[0].data.qIdx**,
  The index associated *q* values
  (mainly for internal use, i.e. discard detectors, q-range selection).
- **exp.dataList[0].data.observable**,
  The values of the observable of the experiment, if any.
  It can be temperature, time, run number.
- **exp.dataList[0].data.observable_name**,
  A string giving the name of the observable.
- **exp.dataList[0].data.temps**,
  The sample temperature measured in Kelvins during the experiment.
- **exp.dataList[0].data.times**,
  The time delta from the first scan in hours.
- **exp.dataList[0].data.norm**,
  For internal use, a boolean indicating whether data were
  normalized by integration of themselves or vanadium or not.


Raw data
^^^^^^^^
Raw dataset, as generated
on IN16B at the ILL, can be imported directly. The algorithm has several
options allowing for detector grouping, unmirroring, integrating and
summation of the scans.

See :class:`in16b_qens_scans_reduction.IN16B_QENS` or
:class:`in16b_fws_scans_reduction.IN16B_FWS` for example.

To import raw data, the following can be used:

.. code-block:: python

    import nPDyn

    exp = nPDyn.Dataset()

    # we can use a path to a folder or a list of strings
    # here for FWS data where we only keep elastic scans
    # and we choose the observable to be the temperature
    exp.importRawData('myDataFolder/', 'IN16B', 'FWS',
                       kwargs={'offset': 0.0, 'observable': 'temperature'})

    # ...and here for QENS data with .xml detector grouping file
    exp.importRawData(['scan01.nxs', 'scan02.nxs', 'scan03.nxs'],
                       'IN16B',
                       'QENS',
                       kwargs={'detGroup': 'IN16B_detGroup.xml'})

The imported dataset are then accessible using:

>>> exp.dataList
[<nPDyn.dataTypes.fwsType.FWSType at 0x7eff2fd75c88>,
 <nPDyn.dataTypes.qensType.QENSType at 0x7eff2f36df60>]


Different methods and properties of the dataset are accessible
through this list, e.g., the momentum transfers using:

>>> exp.dataList[1].data.qVals
array([0.19102381, 0.29274028, 0.43543718, 0.56747019, 0.69687497,
0.82305221, 0.94541753, 1.0634042 , 1.17646584, 1.28407863,
1.38574439, 1.48099215, 1.5693807 , 1.65050083, 1.72397668,
1.78946811, 1.84667172, 1.89532256])


Nexus (hdf5) files
^^^^^^^^^^^^^^^^^^
Nexus files as generated by `Mantid <https://www.mantidproject.org>`_ can
be read by nPDyn using the :py:meth:`dataset.Dataset.importFiles` method.

The file will be assumed to be a Nexus file if the extension is '.nxs',
hence the following:

.. code-block:: python

    import nPDyn

    exp = nPDyn.Dataset()
    exp.importFiles(QENSFiles=['mySample01.nxs', 'mySample02.nxs'],
                    resFiles=['vanadium.nxs'],
                    ECFile='empty_cell.nxs', D2OFile='d2O_background.nxs')

will import all files using the Nexus file parser. The files 'mySample01.nxs'
and 'mySample02.nxs' will by imported into
:py:attr:`dataset.Dataset.dataList` list, 'vanadium.nxs' into
:py:attr:`dataset.Dataset.resData` list, 'empty_cell.nxs' into
:py:attr:`dataset.Dataset.ECData` attribute and 'd2O_background.nxs' into
:py:attr:`dataset.Dataset.D2OData` attribute.

In case the user want to force a specific file format, the following
can be used: ``exp.importFiles(fileFormat='nexus', ...)``.


.inx files
^^^^^^^^^^
Similarly to Nexus files, nPDyn can read '.inx' files as generated by the
software SLAW available at the MLZ in Garching, Germany.
The usage is essentially the same as for Nexus file and the file format can
be forced using: ``exp.importFiles(fileFormat='inx', ...)``


Mantid workspace
^^^^^^^^^^^^^^^^
In order to provide a good integration within
`Mantid`_, nPDyn contains also
a class to handle `Mantid`_ workspaces
generated by the ``IndirectILLReductionQENS`` and ``IndirectILLReductionFWS``
algorithms.
Therefore, all data importations and processing can be performed using
Mantid and the modelling and fitting with nPDyn directly in the
`Mantid`_ workbench window.

Again, the workspace can be imported directly using:

.. code-block:: python

    import nPDyn

    exp = nPDyn.Dataset()
    exp.importFiles(
        fileFormat='mantid',  # optional
        QENSFiles=[ws1, ws2],
        resFiles=[vana],
        ECFile=empty_cell, D2OFile=d2O_background)

where the variables 'ws1', 'ws2', 'vana', 'empty_cell', and 'd2O_background'
are MatrixWorkspace or WorkspaceGroup from
`Mantid`_.

.. warning::
    When Mantid workspace are used, the nPDyn methods for data processing
    won't affect the intensities, errors or energies of the datasets.
    Only the other attributes (qVals, qIdx, name, observable, observable_name,
    times, temps) can be changed.
    Use Mantid algorithms to process intensities, errors or energies.
    Note also that if you further process your data with Mantid after having
    imported them in nPDyn, you may have to import the new OutputWorkspace
    from the Mantid algorithm.
    Indeed, nPDyn does not monitor what Mantid does and it will not update the
    workspace being used after an algorithm is applied.
