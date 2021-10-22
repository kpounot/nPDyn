"""Helper functions for nPDyn."""


from collections import namedtuple

import numpy as np

from nPDyn import Sample


def groupBy_qValues(data, qVals, newQVals):
    """Group data based on momentum transfers q.

    This method is a helper function that can be used to group
    diffraction data based on momentum transfer q-values to match
    the quasi-elastic datasets.

    The algorithm simply take the *qVals* array and will group
    the *data* based on the distance of *qVals* values to values
    in the *newQVals* array.

    Parameters
    ----------
    data : np.ndarray
        The array ocntaining the data to be grouped.
    qVals : 1D numpy array
        The q-values corresponding to the data.
    newQVals : 1D numpy array
        The new q-values to be used for grouping.

    """
    idxList = []
    for idx, val in enumerate(qVals):
        minIdx = np.argmin((val - newQVals) ** 2)
        idxList.append(minIdx)

    idxList = np.array(idxList)
    newData = []
    for idx, val in enumerate(np.unique(idxList)):
        ids = np.where(idxList == val)[0]
        newData.append(np.mean(data[ids]))

    return np.array(newData)


def dataset_from_array(
    intensities,
    errors,
    energies,
    qVals,
    name=None,
    temp=None,
    time=None,
    observable=None,
    norm=False,
    diffraction=None,
    diff_qVals=None,
):
    r"""Create a :py:class:`baseType.BaseType` instance from arrays.

    The function allows to create a new dataset to be added to the data list
    using user provided arrays for intensities, errors, energies,
    and q values.
    Additional keywords can be passed to the function to define temperature,
    time and observable as well as diffraction pattern.

    Parameters
    ----------
    intensities : numpy.ndarray
        The intensities / values for the scatterinf function
        :math:`S(q, \omega)`. The shape should be of the form
        (number of observables, number of q values, number of energy channels)
    errors : np.ndarray
        The errors associated with the intentities, with the same shape.
    energies : np.ndarray
        The energies associated with the third axis of the intensities.
    qVals : np.ndarray
        The momentum transfer q values associated with the second axis of
        the intensities.
    name : str
        A name for this dataset.
    temp : np.ndarray
        A 1D array giving the temperature(s).
    time : np.ndarray
        A 1D array giving the time(s).
    observable : np.ndarray
        The name of the observable.
    norm : bool
        Whether the intensities should be considered as normalized or not.
    diffraction : np.ndarray
        The diffraction pattern for the sample, if available.
    diff_qVals : np.ndarray
        The momentum transfer q values associated with the diffraction pattern.

    """
    out = Sample(
        intensities,
        errors=errors,
        energies=energies,
        temperature=temp,
        time=time,
        name=name,
        q=qVals,
        observable=observable,
        diffraction=diffraction,
        qdiff=diff_qVals,
    )

    return out


def get_stokes_einstein_curves(
    diff_coeffs, temperatures, eta=None, npoints=100, minT=273, maxT=310
):
    """Compute the temperature dependence of the self-diffusion coefficient.

    The methods can be used to compute the theoretical Stokes-Einstein law
    for a particle whose radius is obtained from a self-diffusion coefficient
    at a given temperature.

    Parameters
    ----------
    diff_coeffs : list
        The self-diffusion coefficient obtained at different temperatures
        in units of meter squared per second.
    temperatures : list
        The temperatures corresponding to each diffusion coefficient in
        units of Kelvin.
    eta : list, optional
        Values for the viscosity corresponding to the diffusion coefficients.
        If None, the viscosity of pure D2O is used for the given
        temperatures.
        (default, None)
    npoints : int, optional
        The number of points for the output curves.
        (default, 100)
    minT : int, optional
        The minimum temperature value for the curve.
        (default, 273)
    maxT : int, optional
        The maximum temperature value for the curve.
        (default, 310)

    Returns
    -------
    curves : np.ndarray
        An array of theoretical Stokes-Einstein law as a function of
        temperature for each diffusion coefficient given in input.
    temps : np.array
        The temperatures used to generate the curves.
    radii : list
        The radii corresponding to the computed curves.

    """

    def getViscosityD2O(T):
        """Computes the viscosity of D2O at given temperature.

        The viscosity eta(T) of D2O is in units of [Pa*s] for given
        temperature in units of [K].
        Dependence of viscosity on temperature was determined by
        Cho et al. (1999)
        "Thermal offset viscosities of liquid H2O,
        D2O and T2O", J.Phys. Chem. B 103(11):1991-1994
        eta(T) is valid from 280K up to 400K.

        """
        C = 885.60402
        a = 2.799e-3
        b = -1.6342e-5
        c = 2.9067e-8
        g = 1.55255
        T0 = 231.832
        t = T - T0
        eta = 1e-3 * C * (t + a * t ** 2 + b * t ** 3 + c * t ** 4) ** (-g)
        return eta

    kb = 1.38064852e-23
    radii = []
    if eta is None:
        temperatures = np.array(temperatures)
        eta = getViscosityD2O(temperatures)

    for idx, val in enumerate(diff_coeffs):
        radius = (
            kb
            * temperatures[idx]
            / (6 * np.pi * val * getViscosityD2O(eta[idx]))
        )
        radii.append(radius)

    curves = []
    temps = np.arange(minT, maxT, (maxT - minT) / npoints)
    for idx, r in enumerate(radii):
        D = kb * temps / (6 * np.pi * eta * r)
        curves.append(D)

    return curves, temps, radii
