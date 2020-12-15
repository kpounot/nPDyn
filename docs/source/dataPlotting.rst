Plot data
=========

nPDyn provides some plotting windows for quasi-elastic neutron scattering
(QENS) and elastic/inelastic fixed-window scans (E/IFWS) data.

Using the result of the fitting procedure presented in the :doc:`dataFitting`
section, the data, the fitted model model and the parameters can be
examined using the window shown below for QENS:

.. figure:: ../fig/qensPlot_plot.png
    :width: 600

    The experimental data are plotted alone with their errors for the
    selected observable and momentum transfer q value.


.. figure:: ../fig/qensPlot_plot_with_model.png
    :width: 600

    Here, the fitted model and its components are added by clicking on the
    associated checkboxes.


.. figure:: ../fig/qensPlot_3d.png
    :width: 600

    An 3D view of all spectra is available by clicking on the '3D plot'
    button.


.. figure:: ../fig/qensPlot_analysis.png
    :width: 600

    The optimized parameters can be plotted by clicking on the 'Analysis'
    button. The global parameters (which are unique for all q-values) are
    represented by a single horizontal line.


And for E/IFWS data:

.. figure:: ../fig/fwsPlot_plot_with_model.png
    :width: 600

    The data are plotted along the momentum-transfer q-values.
    The fitted model, which is used to extract the mean-squared displacement
    is added.


.. figure:: ../fig/fwsPlot_3D.png
    :width: 600

    The whole dataset can be plotted using the '3D plot' button.


.. figure:: ../fig/fwsPlot_analysis_with_errors.png
    :width: 600

    The optimized parameters can be plotted along different axis
    (observable, energy, q-values). Here, the uncertainty on the
    parameters is represented by the blue shaded area around the curve.
