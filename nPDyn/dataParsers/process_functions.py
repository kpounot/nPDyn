""" Process functions for raw data from position sensitive detectors.

"""

from dateutil.parser import parse

import numpy as np

from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from nPDyn.dataParsers.xml_detector_grouping import IN16B_XML

from nPDyn import Sample
from nPDyn.sample import concatenate


def normalizeToMonitor(data, peaks=None, monPeaks=None, fws=False):
    """Normalize the data by divinding by the monitor.

    If `peaks` an `monPeaks` are not None, the data are aligned
    to monitor peaks for each momenum transfer prior to normalization.
    For FWS data, only the values at peak positions are used.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`
    peaks : np.ndarray
        The position of the peak(s) for each momentum transfer.
        Requires 'monPeaks' as well.
    monPeaks : np.ndarray
        The position of the peak(s) in monitor.
        Requires 'peaks' as well.
    fws : bool
        Whether data are FWS or not.

    """
    data = data.swapaxes(0, data.axes.index("q"))
    lastAxis = "channels" if "channels" in data.axes else "energies"
    data = data.swapaxes(-1, data.axes.index(lastAxis))
    normData = []
    for qIdx, qData in enumerate(data):
        if peaks is not None and monPeaks is not None:
            monPeaks = np.asarray(monPeaks.flatten())
            if fws:
                normData.append(
                    np.sum(qData[peaks[qIdx]] / qData.monitor[monPeaks], 0)
                )
            else:
                qData = qData.roll(int(monPeaks - peaks[qIdx]))
                normData.append(qData / qData.monitor)
        else:
            new_data = qData / qData.monitor
            if fws:
                np.place(new_data, ~np.isfinite(new_data), 0)
                new_data = new_data.sum(0)
            normData.append(new_data)

    normData = np.stack(normData, "q")
    if normData.ndim > 1:
        normData = normData.swapaxes(1, data.axes.index("q"))

    return normData


def findPeaksFWS(data):
    """Find peaks in FWS data.

    For arrays with more than one dimension, the function assumes that
    the first axis is the momentum tansfer q values ('q') and the second
    the recorded channels ('channels').

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`

    """
    if np.asarray(data).ndim == 1:
        return find_peaks(data, distance=data.shape[0] / 2)[0]
    else:
        return np.array(
            [find_peaks(val, distance=data.shape[1] / 2)[0] for val in data]
        )


def findPeaks(data, peakFindingMask=None):
    """Find the peak for each momentum transfer in data.

    The function always return a single peak for each momentum
    transfer value. Hence, it should be called twice for mirrored data,
    once for each wing, before unmirroring.

    The data are expected to have the momentum transfer q-values in the
    first dimension, the channels in the second dimension and, for 3D
    arrays, the momentum transfer in vertical position qz in the third
    dimension.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`
    peakFindingMask : np.ndarray (optional)
        A mask to exclude some points from peak search.
        (default None, use a small window centered on the 'channel' axis)

    """
    if data.ndim > 1:
        data = data.swapaxes(0, data.axes.index("q"))
        lastAxis = "channels" if "channels" in data.axes else "energies"
        data = data.swapaxes(1, data.axes.index(lastAxis))
        if data.ndim == 3:
            data = data.sum(2)

        if "qz" in data.axes:
            data = data.sum(data.axes.index("qz"))

    arr = np.asarray(data)
    np.place(arr, ~np.isfinite(arr), 0)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    nbrChannels = arr.shape[1]
    middle = int(nbrChannels / 2)
    window = (middle - nbrChannels / 5, middle + nbrChannels / 5)

    if peakFindingMask is None:
        mask = np.zeros_like(arr)
        mask[:, int(window[0]) : int(window[1])] = 1

    maskedData = arr * mask

    # Finds the peaks by using a Gaussian function to fit the data
    gaussian = lambda x, normF, gauW, shift: (
        normF
        * np.exp(-((x - shift) ** 2) / (2 * gauW ** 2))
        / (gauW * np.sqrt(2 * np.pi))
    )

    # Finds the peaks using a Savitsky-Golay filter to
    # smooth the data, and extract the position of the maximum
    filters = np.array(
        [
            savgol_filter(maskedData, 5, 4),
            savgol_filter(maskedData, 11, 4),
            savgol_filter(maskedData, 19, 3),
            savgol_filter(maskedData, 25, 5),
        ]
    )
    savGolPeaks = np.mean(np.argmax(filters, 2), 0)

    findPeaks = []
    for qData in maskedData:
        qPeaks = find_peaks(
            qData,
            distance=qData.shape[0] / 2,
            prominence=0.5 * qData.max(),
        )[0]
        selData = qData[qPeaks]
        findPeaks.append(qPeaks[np.argmax(selData)].squeeze())
    findPeaks = np.array(findPeaks)

    try:
        gaussPeaks = []
        for qData in maskedData:
            qData = np.convolve(qData, np.ones(12), mode="same")
            params = curve_fit(
                gaussian,
                np.arange(nbrChannels),
                qData,
                bounds=(0.0, np.inf),
                p0=[
                    qData.max(),
                    1,
                    nbrChannels / 2,
                ],
                max_nfev=10000,
            )
            gaussPeaks.append(params[0][2])
        gaussPeaks = np.array(gaussPeaks)

        return np.rint(
            0.5 * gaussPeaks + 0.25 * savGolPeaks + 0.25 * findPeaks
        ).astype(int)

    except RuntimeError:
        return np.rint(0.5 * savGolPeaks + 0.5 * findPeaks).astype(int)


def detGrouping(data, detGroup=None):
    """The function performs a sum along detector tubes using the provided
    range to be kept.

    It makes use of the :arg detGroup: argument.


    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`
    detGroup : int, list, file path
        If the argument is a scalar, it sums over all
        values that are in the range
        [center of the tube - detGroup : center of the tube + detGroup].

        If the argument is a list of integers, then each element of the
        list is assumed to correspond to a range for each corresponding
        detector in ascending order.

        If the argument is a mantid-related xml file (a python string),
        the xml_detector_grouping module is then used to parse the xml
        file and the provided values are used to define the range.

    """
    axis = data.axes.index("qz")
    data.swapaxes(0, data.axes.index("q"))
    data_SD = np.sum(data[: data.nbr_single_dec], axis)
    data_PSD = data[data.nbr_single_dec :]
    if isinstance(detGroup, int):
        midPos = int(data.shape[1] / 2)
        data_PSD = data_PSD.take(
            np.arange(midPos - detGroup, midPos + detGroup),
            axis,
        )
        return np.sum(data, axis)

    elif detGroup is None:
        return np.sum(data, axis)

    elif isinstance(detGroup, (list, tuple, np.ndarray)):
        midPos = int(data.shape[axis] / 2)
        new_arr = []
        for detId, detData in enumerate(data_PSD):
            slice_axis = detData.axes.index("qz")
            new_arr.append(
                detData.take(
                    np.arange(
                        midPos - detGroup[detId],
                        midPos + detGroup[detId],
                        dtype="int64",
                    ),
                    slice_axis,
                ).sum(slice_axis)
            )

    elif isinstance(detGroup, str):
        numTubes = data_PSD.shape[data_PSD.axes.index("q")]
        xmlData = IN16B_XML(detGroup, numTubes)

        detRanges = xmlData.getPSDValues()

        new_arr = []
        for detId, detData in enumerate(data_PSD):
            slice_axis = detData.axes.index("qz")
            new_arr.append(
                detData.take(
                    np.arange(detRanges[detId][0], detRanges[detId][1]),
                    slice_axis,
                ).sum(slice_axis)
            )

    data_PSD = np.stack(new_arr, "q")
    out = np.concatenate((data_SD, data_PSD), 0)

    return out


def alignGroups(data, position=None):
    """Align the peaks along the z-axis of the detectors.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`.
        First axis is assumed to be q-values.
    position : int (optional)
        Position of the center along the 'channels' axis.
        (default, None, is determined automatically)

    Returns
    -------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample` for which the data maxima
        were aligned along the z direction.
    center : int
        The center determined by the algorithm, which can then be used
        to convert the time-of-flight to energies as it defines the elastic
        peak.

    """
    modData = np.asarray(data[data.nbr_single_dec :].sum(0))
    modData = np.array(
        [np.convolve(val, np.ones(20), mode="same") for val in modData]
    )
    nbrPixels = modData.shape[1]
    nbrChannels = modData.shape[0]
    window = (int(nbrChannels / 2 - 250), int(nbrChannels / 2 + 250))
    mask = np.zeros_like(modData)
    mask[window[0] : window[1]] = 1
    modData *= mask

    f = lambda x, d, offset, shift: (d / np.cos((x - shift) / x.size) + offset)

    res = curve_fit(
        f,
        np.arange(nbrPixels),
        modData.argmax(0),
        p0=[nbrPixels / 2, 1024, nbrPixels / 2],
        loss="cauchy",
        method="trf",
    )

    maxPos = np.rint(f(np.arange(nbrPixels), *res[0])).astype(int)

    if position is None:
        center = np.min(maxPos)
    else:
        center = position

    shifts = center - maxPos
    for qIdx, qData in enumerate(data):
        if qIdx >= data.nbr_single_dec:
            data[qIdx] = np.array(
                [np.roll(val, shifts[idx]) for idx, val in enumerate(qData.T)]
            ).T
            data.errors[qIdx] = np.array(
                [np.roll(val, shifts[idx]) for idx, val in enumerate(qData.T)]
            ).T

    return data, center


def unmirror(data, refPeaks=None):
    """Unmirror data using the elastic peak as a reference.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`.
    refPeaks : np.ndarray (optional)
        Reference peak positions for the elastic signal.
        Should have one entry for each momentum transfer q-values
        in the first dimension and two entries in the second dimension
        for the peak in the left and right wing, respectively.
        (default None, will be determined automatically)

    """
    data = data.swapaxes(0, data.axes.index("q"))
    data = data.swapaxes(1, data.axes.index("channels"))

    nbrChannels = data.shape[1]
    midChannel = int(nbrChannels / 2)

    if refPeaks is None:
        leftPeaks = findPeaks(data[:, :midChannel])
        rightPeaks = findPeaks(data[:, midChannel:])
    else:
        leftPeaks = refPeaks[:, 0]
        rightPeaks = refPeaks[:, 1]

    newData = []
    for qIdx, qData in enumerate(data):
        newData.append(
            np.roll(qData, midChannel - leftPeaks[qIdx])
            + np.roll(qData, midChannel - (midChannel + rightPeaks[qIdx]))
        )
    newData = np.stack(newData, "q")

    leftPeak = int(np.mean(leftPeaks))
    rightPeak = int(np.mean(rightPeaks))
    newMonitor = np.roll(data.monitor, midChannel - leftPeak) + np.roll(
        data.monitor, midChannel - (midChannel + rightPeak)
    )
    newData.monitor = newMonitor[int(midChannel / 2) : 3 * int(midChannel / 2)]

    return newData[:, int(midChannel / 2) : 3 * int(midChannel / 2)]


def convertChannelsToEnergy(
    data,
    type,
    refDist=33.388,
    tElastic=None,
):
    """Convert the 'channels' axis to 'energies'

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`.
    type : {'qens', 'fws', 'bats'}
        Type of dataset that is being processed.
    refDist : float (optional)
        Reference distance from the pulse chopper used in BATS mode
        to the sample.
    tElastic : int (optional)
        Reference value of time-of-flight for the elastic signal.

    """
    data = data.swapaxes(0, data.axes.index("q"))
    data = data.swapaxes(1, data.axes.index("channels"))
    if type == "qens":
        data.energies = np.linspace(
            -data.energies, data.energies, data.shape[1]
        )
        data.axes[1] = "energies"
    elif type == "fws":
        data.axes[1] = "energies"
    elif type == "bats":
        if tElastic is None:
            center = int(np.mean(findPeaks(data)))
            tElastic = data.tof[center]
        refEnergy = 1.3106479439885732e-40 / (data.wavelength * 1e-10) ** 2
        refVel = np.sqrt(2 * refEnergy / 1.67493e-27)
        refTime = refDist / refVel
        dt = data.tof - tElastic
        velocities = refDist / (refTime + dt * 1e-6)
        energies = 1.67493e-27 * velocities ** 2 / 2
        energies -= refEnergy
        energies *= 6.241509e18 * 1e6
        data.energies = energies
        data.axes[1] = "energies"
    else:
        return "Argument 'type' should be one of {'qens', 'fws', 'bats'}"

    data = data.swapaxes(1, data.axes.index("q"))
    data = data.swapaxes(-1, data.axes.index("energies"))

    return data


def mergeDataset(dataList, observable="time"):
    """Produce a single dataset from multiple FWS data.

    In the case of different sampling for the energy transfers
    used in FWS data, the function interpolates the smallest arrays
    to produce a unique numpy array of FWS data.

    Parameters
    ----------
    data : list of :py:class:`sample.Sample`
        list of instances of :py:class:`sample.Sample`.
    observable : {'time', 'temperature', 'pressure'} (optional)
        The name of the observable used for series of data.
        (default, 'time')

    """
    energyMap = {}
    for data in dataList:
        if np.asarray(data.energies).max() not in energyMap:
            energyMap[np.asarray(data.energies).max()] = []
        energyMap[np.asarray(data.energies).max()].append(data.copy())

    # Finds the maximum sampling in the list of dataset
    listSizes = [len(val) for val in energyMap.values()]
    maxKey = list(energyMap.keys())[np.argmax(listSizes)]
    times = np.array([val.time for val in energyMap[maxKey]])
    times = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
    temperatures = np.array([val.temperature for val in energyMap[maxKey]])
    pressures = np.array([val.pressure for val in energyMap[maxKey]])
    monitors = np.array([val.monitor for val in energyMap[maxKey]])

    dataset = []
    for val in energyMap.values():
        tmpSample = np.stack(val, observable)
        if tmpSample.shape[0] > 1:
            tmpTime = np.array([entry.time for entry in val])
            tmpTime = np.array(
                [(t - tmpTime[0]).total_seconds() / 3600 for t in tmpTime]
            )
            interpArr = interp1d(
                tmpTime,
                np.asarray(tmpSample),
                axis=0,
                bounds_error=False,
                fill_value=(tmpSample[0], tmpSample[-1]),
            )
            interpErr = interp1d(
                tmpTime,
                tmpSample.errors,
                axis=0,
                bounds_error=False,
                fill_value=(tmpSample.errors[0], tmpSample.errors[-1]),
            )
            tmpSample._arr = interpArr(times)
            tmpSample.errors = interpErr(times)
        dataset.append(tmpSample)

    if "energies" in dataset[0].axes or "channels" in dataset[0].axes:
        out = dataset[0]
    else:
        out = np.stack(dataset, "energies")

    out.time = times.flatten()
    out.temperature = temperatures.flatten()
    out.pressure = pressures.flatten()
    out.monitor = monitors
    out = out.swapaxes(0, out.axes.index(observable))
    out = out.swapaxes(1, out.axes.index("q"))
    out.observable = observable

    return out


def alignToZero(data, peaks=None):
    """Align data peaks to the zero of energy transfers.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`.
    peaks : np.ndarray (optional)
        Array of peak positions for each momentum transfer q value.
        (default, None - will be determined automatically)

    """
    peaks = peaks
    if peaks is None:
        peaks = findPeaks(data)

    data = data.swapaxes(0, data.axes.index("q"))
    data = data.swapaxes(1, data.axes.index("energies"))
    zeroIdx = np.argmin(data.energies ** 2)
    out = []
    for qIdx, qData in enumerate(data):
        out.append(qData.roll(peaks[qIdx] - zeroIdx, 2))

    np.stack(out, "q")
    out = out.swapaxes(2, data.axes.index("energies"))
    out = out.swapaxes(1, data.axes.index("q"))

    return out


def sumAlongObservable(data):
    """Sum a single dataset along with monitor over the observable.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`.
    peaks : np.ndarray (optional)
        Array of peak positions for each momentum transfer q value.
        (default, None - will be determined automatically)

    """
    data = np.sum(data, data.axes.index(data.observable))[None, :]
    data.monitor = (
        np.sum(data.monitor, 0) if data.monitor.ndim > 1 else data.monitor
    )
    data.axes[0] = data.observable

    return data


def avgAlongObservable(data):
    """Average a single dataset along with monitor over the observable.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`.
    peaks : np.ndarray (optional)
        Array of peak positions for each momentum transfer q value.
        (default, None - will be determined automatically)

    """
    data = np.mean(data, data.axes.index(data.observable))[None, :]
    data.monitor = (
        np.mean(data.monitor, 0) if data.monitor.ndim > 1 else data.monitor
    )
    data.axes[0] = data.observable

    return data


def alignTo(data, refPos, peaks=None):
    """Align data peaks to zero energy transfer.

    Parameters
    ----------
    data : :py:class:`sample.Sample`
        Instance of :py:class:`sample.Sample`.
    refPos : int
        Reference index on energy/channels axis.
    peaks : np.ndarray (optional)
        Array of peak positions for each momentum transfer q value.
        (default, None - will be determined automatically)

    """
    peaks = peaks
    if peaks is None:
        peaks = findPeaks(data)

    qIdx = data.axes.index("q")
    data = data.swapaxes(0, qIdx)
    axName = "energies" if "energies" in data.axes else "channels"
    refAx = data.axes.index(axName)
    new_data = []
    for idx, val in enumerate(data):
        new_data.append(val.roll(refPos - peaks[idx], refAx - 1))

    new_data = np.stack(new_data, "q")
    new_data = new_data.swapaxes(0, qIdx)

    return new_data
