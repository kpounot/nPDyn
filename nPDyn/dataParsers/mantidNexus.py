import numpy as np
import h5py as h5
from collections import namedtuple

from dateutil.parser import parse

from scipy.interpolate import interp1d


def processNexus(dataFile, FWS=False, averageTemp=True):
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

    # Fixed window scan processing
    if FWS is True:
        interp = False
        name = h5File["mantid_workspace_1/logs/subtitle/value"][()].astype(str)
        times = h5File["mantid_workspace_1/logs/start_time/value"][
            (0)
        ].decode()
        wavelength = h5File["mantid_workspace_1/logs/wavelength/value"][()]
        listQ = h5File["mantid_workspace_1/workspace/axis2"][()]
        observable_name, spectrum_axis = _processAlgoInfo(h5File)
        if spectrum_axis in ["SpectrumNumber", "2Theta"]:  # converts to q
            listQ = np.array(
                4 * np.pi / wavelength * np.sin(np.pi * listQ / 360)
            )
        if spectrum_axis == "Q2":  # converts to q
            listQ = np.sqrt(np.array(listQ))

        dataList = []  # Stores the full dataset for a given data file

        # Initialize some lists and store energies,
        # intensities and errors in them
        listObs = []
        listI = []
        listErr = []
        deltaE = []
        listT = []
        for j, workspace in enumerate(h5File):
            listObs.append(h5File[workspace + "/workspace/axis1"][()])
            listI.append(h5File[workspace + "/workspace/values"][()])
            listErr.append(h5File[workspace + "/workspace/errors"][()])
            deltaE.append(
                h5File[workspace + "/logs/Doppler.maximum_delta_energy/value"][
                    ()
                ]
            )
            listT.append(
                h5File[workspace + "/logs/sample.temperature/value"][()]
            )

        for data in listObs:
            if data.shape[0] != listObs[0].shape[0]:
                interp = True

        if interp:
            listObs, listI, listErr = _interpFWS(listObs, listI, listErr)

        # converts intensities and errors to numpy and array and transpose
        # to get (# frames, # qVals, # energies) shaped array
        listObs = np.array(listObs).mean(0)
        listI = np.array(listI).T
        listErr = np.array(listErr).T
        deltaE = np.array(deltaE)[:, 0]
        listT = np.array(listT).mean(0)

        # process observable name
        if observable_name == "time":
            listObs = listObs / 3600  # converts to hours
            times = np.copy(listObs)
        if observable_name == "temperature":
            listT = np.copy(listObs)
            times = np.array([parse(times.split(",")[0])])
            times = np.array(
                [(t - times[0]).total_seconds() / 3600 for t in times]
            )

        dataList = dataTuple(
            listI,
            listErr,
            deltaE,
            listT,
            times,
            name,
            listQ,
            np.arange(listQ.size),
            listObs,
            observable_name,
            False,
        )

        return dataList

    # Assumes QENS data
    else:
        wavelength = h5File["mantid_workspace_1/logs/wavelength/value"][()]
        name = h5File["mantid_workspace_1/logs/subtitle/value"][()].astype(str)
        observable_name, spectrum_axis = _processAlgoInfo(h5File)
        temps = np.array(
            [h5File["mantid_workspace_1/logs/sample.temperature/value"][()]]
        )
        times = h5File["mantid_workspace_1/logs/start_time/value"][
            (0)
        ].decode()
        times = np.array([parse(times.split(",")[0])])
        times = np.array(
            [(t - times[0]).total_seconds() / 3600 for t in times]
        )

        listQ = h5File["mantid_workspace_1/workspace/axis2"][()]

        if spectrum_axis in ["SpectrumNumber", "2Theta"]:  # converts to q
            listQ = np.array(
                4 * np.pi / wavelength * np.sin(np.pi * listQ / 360)
            )
        if spectrum_axis == "Q2":  # converts to q
            listQ = np.sqrt(np.array(listQ))

        listObs = np.array([])
        listE = h5File["mantid_workspace_1/workspace/axis1"][()][:-1] * 1e3

        listI = h5File["mantid_workspace_1/workspace/values"][()]
        listErr = h5File["mantid_workspace_1/workspace/errors"][()]

        listI = listI[np.newaxis, :, :]
        listErr = listErr[np.newaxis, :, :]

        # process observable name
        if observable_name == "time":
            listObs = np.copy(times)
        if observable_name == "temperature":
            listObs = np.copy(temps)

        dataList = dataTuple(
            listI,
            listErr,
            listE,
            temps,
            times,
            name,
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

    # Performs an interpolation for each dataset with a sampling
    # rate smaller than the maximum
    for k, data in enumerate(listX):
        if data.shape[0] != maxSize:
            interpI = interp1d(
                data,
                listI[k],
                kind="linear",
                fill_value=(listI[k][:, 0], listI[k][:, -1]),
                bounds_error=False,
            )

            interpErr = interp1d(
                data,
                listErr[k],
                kind="linear",
                fill_value=(listErr[k][:, 0], listErr[k][:, -1]),
                bounds_error=False,
            )

            data = maxX
            listI[k] = interpI(maxX)
            listErr[k] = interpErr(maxX)

            data = maxX

    return listX, listI, listErr


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

    obs = "time"  # default value in case no observable is found
    specAx = ""

    for i in algoConfig:
        if "Observable" in i:
            obs = i.split(",")[1]
            obs = obs[obs.find(":") + 2 :]
        if "SpectrumAxis" in i:
            specAx = i.split(",")[1]
            specAx = specAx[specAx.find(":") + 2 :]

    if obs == "start_time":
        obs = "time"

    return obs, specAx
