Fit data
========

nPDyn relies on a builtin implementation to model and fit data, but provides
also some methods to fit your data using 
`lmfit <https://lmfit.github.io/lmfit-py/>`_ as modelling and fitting backend.

In the following, we will introduce data modelling using two type of 
data and analysis, the first being the fit of quasi-elastic neutron
scattering (QENS) measurement on a protein solution sample and the 
second the fit of elastic fixed-window scans (EFWS) of a protein
powder sample.

The QENS data will be modelled using the following:

.. math::
    :label: eq:QENS

    \rm S(q, \hbar \omega) = R(q, \hbar \omega) \otimes
    \beta_q \left[\alpha \mathcal{L}_{\Gamma} +
    (1 - \alpha) \mathcal{L}_{\Gamma + \gamma} \right]
    + \beta_{D_2O} \mathcal{L}_{D_2O}

where :math:`\rm q` is the momentum transfer, :math:`\rm \hbar \omega`
the energy transfer, :math:`\rm R(q, \hbar \omega)` is the resolution
function (here a pseudo-Voigt profile), 
:math:`\rm \beta_q` a vector of scalars accounting
for detector efficiency (one scalar for each q), :math:`\rm \alpha` a 
scalar between 0 and 1, :math:`\rm \mathcal{L}_{\Gamma}` a Lorentzian
of accounting for center-of-mass diffusion with a explicit q-dependent
width :math:`\rm \Gamma = D_s q^2`, where :math:`\rm D_s` is the 
self-diffusion coefficient, :math:`\rm \mathcal{L}_{\Gamma + \gamma}` is a
Lorentzian accounting for internal dynamics with 
:math:`\rm \gamma = \frac{D_i q^2}{1 + D_i q^2 \tau}` (see [#]_) and 
:math:`\rm \beta_{D_2O} \mathcal{L}_{D_2O}` accounting for the signal
from the :math:`\rm D_2O`.

The EFWS data will be modelled using a simple Gaussian to extract the 
mean-squared displacement (MSD) as a function of temperature:

.. math::
    :label: eq:EFWS

    \rm S(q, 0) = e^{-\frac{q^2 MSD}{6}}


We use the sample data in the the test suite of nPDyn (from package 
root directory, use ``cd nPDyn/tests/sample_data/`` and we initiate
our dataset using, for QENS::

    >>> from nPDyn import Dataset
    >>> import numpy as np
    >>> qens = Dataset(
    ...     QENSFiles=['lys_part_01_QENS_before_280K.nxs'], 
    ...     resFiles=['vana_QENS_280K.nxs'], 
    ...     ECFile='empty_cell_QENS_280K.nxs', 
    ...     D2OFile='D2O_QENS_280K.nxs')
    >>>
    >>> # Perform some data processing
    >>> qens.binAll(5)
    >>> qens.subtract_EC()
    >>> qens.setQRange(0.4, 1.8)
    >>>
    >>> # Extract momentum transfers for modelling and make it 2D
    >>> q = qens.dataList[0].data.qVals[:, np.newaxis]

and for EFWS::

    >>> efws = Dataset(FWSFiles=['D_syn_fibers_elastic_10to300K.inx'])

Using builtin model backend
---------------------------
The builtin modelling interface has been designed to be easy to use
and adapted to the multi-dimensional dataset obtained with neutron
backscattering spectroscopy and a mix of global and non-global parameters. 

The basic workflow is as follows:

#. Create a **Parameter** instance with parameters that can be 
   scalar, 1D, 2D or any shaped arrays.
#. Create a **Model** instance that is initiated with the prviously
   created parameters.
#. Add several **Component** or other **Model** to this model.
   Each component is associated with a Python function, the 
   arguments of which can be dynamically defined at the creation
   of the component using an expression as a string as shown below.
#. Fit your data!

For the QENS data, we first model the resolution function using 
a pseudo-voigt profile. To this end, we use the 
:py:func:`builtins.modelPVoigt` builtin model from nPDyn.
The same is done for :math:`\rm D_2O` background using the 
:py:func:`builtins.modelD2OBackground` builtin model.

Simply use::

    >>> from nPDyn.models.builtins import modelPVoigt
    >>> from nPDyn.models.builtins import modelD2OBackground
    >>> qens.fitRes(model=modelPVoigt(q, 'resolution'))
    >>> qens.D2OData.fit(model=modelD2OBackground(q, temperature=280))

With a little anticipation on this documentation, you can use
the following to look at the fit result::

    >>> qens.plotResFunc()
    >>> qens.plotD2OFunc()


Create parameters
^^^^^^^^^^^^^^^^^
For the QENS sample, there are 6 parameters, namely :math:`\rm \beta_q`,
:math:`\rm \alpha`, :math:`\rm D_s`, :math:`\rm D_i`, :math:`\rm \tau`,
and :math:`\rm \beta_{D_2O}`.

We can thus create the **Parameters** instance::

    >>> from nPDyn.models import Parameters
    >>> pQENS = Parameters(
    ...     beta={'value': np.zeros_like(q) + 1, 'bounds': (0., np.inf)},
    ...     alpha={'value': 0.5, 'bounds': (0., 1)},
    ...     Ds={'value': 5, 'bounds': (0., 100)},
    ...     Di={'value': 20, 'bounds': (0., 100)},
    ...     tau={'value': 1, 'bounds': (0., np.inf)},
    ...     bD2O={'value': 0.1, 'bounds': (0., np.inf)})

For the EFWS sample, we only have the MSD and we use a slightly different
way to instantiate the **Parameters** instance for demonstration purpose::

    >>> from nPDyn.models import Model
    >>> pEFWS = Parameters(msd=0.5)
    >>> pEFWS.set('msd', bounds=(0., np.inf), fixed=False)

Instantiate a Model
^^^^^^^^^^^^^^^^^^^
Instantiating a **Model** is very straightforward, just use::

    >>> modelQENS = Model(pQENS, 'QENS')  # for QENS data
    >>> modelEFWS = Model(pEFWS, 'EFWS')  # for EFWS data

Add components
^^^^^^^^^^^^^^
The ``modelQENS`` model should contain three components, or three lineshapes,
as we can see in equation :eq:`eq:QENS`, namely a Lorentzian for 
center-of-mass diffusion, a Lorentzian for internal dynamics and the model
we used for :math:`\rm D_2O` background.
We can add them using::

    >>> from nPDyn.models import Component
    >>> from nPDyn.models.presets import lorentzian
    >>> modelQENS.addComponent(Component(
    ...     'center-of-mass', 
    ...     lorentzian,
    ...     scale='beta * alpha',  # will find the parameters values in pQENS
    ...     width='Ds * q**2',  # we will give q on the call to the fit method
    ...     center=0))  # we force the center to be at 0 
    ...                 # (as it is given by the convolution with resolution)
    >>> # we can add, subtract, multiply or divide a model using a Component or
    >>> # another Model 
    >>> internal = Component(
    ...     'internal', 
    ...     lorentzian,
    ...     scale='beta * (1 - alpha)', 
    ...     width='Di * q**2 / (1 + Di * q**2 * tau)', 
    ...     center=0)  # we force the center to be at 0 
    ...                # (as it is given by the convolution with resolution)
    >>> modelQENS += internal
    >>> # for the D2O signal, we use a lambda function to include the scaling
    >>> modelQENS.addComponent(Component(
    ...     '$D_2O$',  # we can use LaTeX for the component and model names
    ...     lambda x, scale: scale * qens.D2OData.fit_best(x=x)[0], 
    ...     scale='bD2O',
    ...     skip_convolve=True))  # we do not want to convolve this 
    >>>                           # component with resolution

The ``modelEFWS`` model uses the momentum transfer q as independent
variable, which will be passed later upon fitting and it contains 
only one component. Here, we use::

    >>> from nPDyn.models.presets import gaussian
    >>> modelEFWS.addComponent(Component(
    ...     'EISF',
    ...     lambda x, scale, msd: scale * np.exp(-x**2 * msd / 6)))

Fit data
^^^^^^^^
The class :py:class:`dataset.Dataset` provides a method to fit all data
in ``Dataset.dataList`` attribute at once. It simply calls the 
:py:meth:`baseType.BaseType.fit` method for each selected data.

Here, we use it and write for QENS::

    >>> qens.fitData(
    ...     model=modelQENS, q=q, convolveRes=True,
    ...     fit_method='basinhopping', fit_kws={'niter': 10, 'disp': True})

and for EFWS, where we set the independent variable to a column vector
containing the momentum transfer q values::
    
    >>> efws.fitData(
    ...     model=modelEFWS, 
    ...     x=efws.dataList[0].data.qVals[:, np.newaxis])


Using *lmfit* backend
---------------------
In addition to the builtin model interface of nPDyn, the API also
provides some helper functions to use the 
`lmfit <https://lmfit.github.io/lmfit-py/>`_ package.
This package is more advanced and exhaustive than the builtin
model interface but it is less adapted to multi-dimensional 
dataset with global and non-global parameters.

This is where the presets and builtin models in nPDyn come into
play, to make it easier to use within the analysis workflow of 
neutron backscattering data.

The interface with `lmfit <https://lmfit.github.io/lmfit-py/>`_
relies on the :py:func:`lmfit_presets.build_2D_model` function.

Build model
^^^^^^^^^^^

Fit data
^^^^^^^^

References
----------
.. [#] https://doi.org/10.1103/PhysRev.119.863
