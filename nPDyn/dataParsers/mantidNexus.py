import numpy as np
import h5py as h5
from collections import namedtuple

from dateutil.parser import parse

from scipy.interpolate import interp1d


def processNexus(dataFile, FWS=False):
    """This script is meant to be used with IN16B data
    pre-processed (reduction, (EC correction)
    and vanadium centering) with Mantid.

    It can handle both QENS and fixed-window scans.

    Then the result is stored as a namedtuple containing several
    members (all being numpy arrays).
        - intensities   - 3D array of counts values for each frame
                          (axis 0), q-value (axis 1) and energy channels
                          (axis 2)
        - errors        - 3D array of errors values for each frame
                          (axis 0), q-value (axis 0) and energy channels
                          (axis 2)
        - energies      - 1D array of energy offsets used
        - temps         - 2D array of temperatures, the first dimension
                          is of size 1 for QENS, and of the same size
                          as the number of energy offsets for FWS. The
                          second dimensions represents the frames
        - times         - same structure as for temps but representing
                          the time
        - name          - name that is stored in the 'subtitle' entry
        - qVals         - 1D array of q-values used
        - qIdx          - same as selQ but storing the indices
        - observable    - data for the observable used for data series
                          ('time' or 'temperature')
        - observable_name - name of the observable used for data series
        - norm          - boolean, whether data were normalized or not

    """

    dataTuple = namedtuple(
        "dataTuple",
        "intensities errors energies "
        "temps times name qVals "
        "qIdx observable "
        "observable_name norm",
    )

    h5File = h5.File(dataFile, "r")

    interpObs = False
    interpEnergies = False
    name = h5File["mantid_workspace_1/logs/subtitle/value"][()].astype(str)
    wavelength = h5File["mantid_workspace_1/logs/wavelength/value"][()]
    listQ = h5File["mantid_workspace_1/workspace/axis2"][()]
    observable_name, spectrum_axis = _processAlgoInfo(h5File)
    if spectrum_axis in ["SpectrumNumber", "2Theta"]:  # converts to q
        listQ = np.array(4 * np.pi / wavelength * np.sin(np.pi * listQ / 360))
    if spectrum_axis == "Q2":  # converts to q
        listQ = np.sqrt(np.array(listQ))
    if spectrum_axis == "decNbr":  # IndirectILLEnergyTransfer, no axis2...
        listQ = _getMomentumTransfers(h5File, listQ, wavelength)

    dataList = []  # Stores the full dataset for a given data file

    # Initialize some lists and store energies,
    # intensities and errors in them
    listI = []
    listErr = []
    energies = []
    temps = []
    times = []
    nbrWorkspace = len(h5File.keys())
    workspaces = [
        "mantid_workspace_%i" % i for i in range(1, nbrWorkspace + 1)
    ]
    for workspace in workspaces:
        temps.append(h5File[workspace + "/logs/sample.temperature/value"][()])
        times.append(
            h5File[workspace + "/logs/start_time/value"][(0)].decode()
        )
        listI.append(h5File[workspace + "/workspace/values"][()])
        listErr.append(h5File[workspace + "/workspace/errors"][()])
        if FWS:
            energies.append(
                h5File[workspace + "/logs/Doppler.maximum_delta_energy/value"][
                    ()
                ]
            )
        else:
            energies = h5File[workspace + "/workspace/axis1"][()]
            energies *= 1e3

    times = np.array([parse(t.split(",")[0]) for t in times])
    times = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
    # process observable name
    if observable_name == "time":
        listObs = np.array(times)
    if observable_name == "temperature":
        listObs = np.array(temps)

    for data in listI:
        if FWS and data.shape[0] != listObs.shape[0]:
            interpObs = True
        if not FWS and data.shape[-1] != len(energies):
            interpEnergies = True

    if interpObs:
        listObs, listI, listErr = _interpFWS(listObs, listI, listErr)
    if interpEnergies:
        energies, listI, listErr = _interpEnergies(energies, listI, listErr)

    # converts intensities and errors to numpy and array and transpose
    # to get (# frames, # qVals, # energies) shaped array
    listI = np.array(listI)
    listErr = np.array(listErr)
    if FWS:
        listI = listI.T
        listErr = listErr.T
        energies = np.array(energies)[:, 0]

    dataList = dataTuple(
        listI,
        listErr,
        energies,
        temps,
        times,
        str(name),
        listQ,
        np.arange(listQ.size),
        listObs,
        observable_name,
        False,
    )

    return dataList


def _interpFWS(listX, listI, listErr):
    """In the case of different sampling for the energy transfers
    used in FWS data, the function interpolates the smallest arrays to
    produce a unique numpy array of FWS data.

    """
    maxSize = 0
    maxX = None

    # Finds the maximum sampling in the list of dataset
    for k, data in enumerate(listX):
        if data.shape[0] >= maxSize:
            maxSize = data.shape[0]
            maxX = data

    # Performs an interpolation for each dataset that has a sampling
    # rate smaller than the maximum
    for k, data in enumerate(listI):
        if data.shape[0] != maxSize:
            interpI = interp1d(
                listX[k],
                listI[k],
                kind="linear",
                fill_value=(listI[k][:, 0], listI[k][:, -1]),
                bounds_error=False,
            )

            interpErr = interp1d(
                listX[k],
                listErr[k],
                kind="linear",
                fill_value=(listErr[k][:, 0], listErr[k][:, -1]),
                bounds_error=False,
            )

            listI[k] = interpI(maxX)
            listErr[k] = interpErr(maxX)

    return listX, listI, listErr


def _interpEnergies(energies, listI, listErr):
    """In the case of different sampling for the energy transfers
    in QENS data, the function interpolates the data and produces
    data with the same length in the energy dimension.

    """
    maxSize = 0
    maxX = None

    # Finds the maximum sampling in the list of dataset
    for k, data in enumerate(listI):
        if data.shape[0] >= maxSize:
            maxSize = data.shape[-1]
            maxX = np.arange(maxSize)

    # Performs an interpolation for each dataset that has a sampling
    # rate smaller than the maximum
    for k, data in enumerate(listI):
        if data.shape[0] != maxSize:
            interpI = interp1d(
                np.arange(data.shape[-1]),
                listI[k],
                kind="linear",
                fill_value=(listI[k][:, 0], listI[k][:, -1]),
                bounds_error=False,
            )

            interpErr = interp1d(
                np.arange(data.shape[-1]),
                listErr[k],
                kind="linear",
                fill_value=(listErr[k][:, 0], listErr[k][:, -1]),
                bounds_error=False,
            )

            listI[k] = interpI(maxX)
            listErr[k] = interpErr(maxX)

    energies = interp1d(
        np.arange(energies.size),
        energies,
        kind="linear",
        fill_value=(energies[0], energies[-1]),
        bounds_error=False,
    )(maxX)

    return energies, listI, listErr


def _getMomentumTransfers(h5File, listQ, wavelength):
    """Obtain the list of q-values from HDF file and length of listQ"""
    anglesPSD = [
        h5File["mantid_workspace_1/logs/PSD.PSD angle %i/value" % i][()]
        for i in range(1, 17)
    ]
    anglesSD = [
        h5File["mantid_workspace_1/logs/SingleD.SD%i angle/value" % i][()]
        for i in range(1, 9)
    ]
    nbrSD = listQ.size - 16

    angles = np.concatenate((anglesSD[:nbrSD], anglesPSD))
    angles = np.array(4 * np.pi / wavelength * np.sin(np.pi * angles / 360))
    return angles.flatten()


def _processAlgoInfo(f):
    """This function is used to extract information about the parameters
    that were used in Mantid for data reduction.

    In particular, it extracts the observable used as well as the
    spectrum axis (either 'SpectrumNumber', '2Theta', 'Q', 'Q2')

    Note
    ----
    The function assumes that the configuration is the same for
    all workspaces in the file and that the first algorithm is
    the one corresponding to data reduction.

    """

    algoConfig = f["mantid_workspace_1/process/MantidAlgorithm_1/data"][(0)]
    algoConfig = algoConfig.decode().splitlines()

    algo = "IndirectILLEnergyTransfer"
    obs = "time"  # default value in case no observable is found
    specAx = ""

    for i in algoConfig:
        if "Observable" in i:
            obs = i.split(",")[1]
            obs = obs[obs.find(":") + 2 :]
        if "SpectrumAxis" in i:
            specAx = i.split(",")[1]
            specAx = specAx[specAx.find(":") + 2 :]
    for i in algoConfig:
        if algo in i:
            specAx = "decNbr"

    if obs == "start_time":
        obs = "time"

    return obs, specAx
