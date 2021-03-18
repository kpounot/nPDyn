"""This module is used for importation of raw data from IN16B instrument.

"""

import os

from dateutil.parser import parse

import h5py
import numpy as np

from collections import namedtuple

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from nPDyn.dataParsers.xml_detector_grouping import IN16B_XML
from nPDyn.dataParsers.stringParser import parseString


class IN16B_BATS:
    """This class can handle raw data from IN16B-BATS
    at the ILL in the hdf5 format.

    Parameters
    ----------
    scanList : string or list
        A string or a list of files to be read and
        parsed to extract the data.
        It can be a path to a folder as well.
    sumScans : bool
        Whether the scans should be summed or not.
    detGroup : string, int
        Detector grouping, i.e. the channels that
        are summed over along the position-sensitive
        detector tubes. It can be an integer, then the
        same number is used for all detectors, where
        the integer defines a region (middle of the
        detector +/- detGroup). It can be a list of
        integers, then each integers of the list
        should corresponds to a detector. Or it can
        be a string, defining a path to an xml file
        as used in Mantid.
        If set to `no`, no detector gouping is performed
        and the data represents the signal for each
        pixel on the detectors. In this case, the
        observable become the momentum transfer q in
        the vertical direction.
    normalize:
        Whether the data should be normalized to the monitor
    strip:
        An integer defining the number of points that
        are ignored at each extremity of the spectrum.
    observable:
        The observable that might be changing over scans.
        It can be `time` or `temperature`.
    tElastic: int, float
        Time for the elastic peak.
        Optional, if None, will be guessed from peak fitting.
    monitorOffset: int, float
        Time offset for the monitor data.
        Optional, if None, either **tElastic** is used
        or the fitted peak position in the data.
    monitorCutoff:
        Cutoff with respect to monitor maximum to discard data.
    pulseChopper : {'C12', 'C34'}
        Chopper pair that is used to define the pulse.
    slidingSum : int, optional
        If not None, the size of the window to perform a sliding sum
        over the observables (default None).

    """

    def __init__(
        self,
        scanList,
        sumScans=True,
        peakFindingMask=None,
        detGroup=None,
        normalize=True,
        strip=None,
        observable="time",
        tElastic=None,
        monitorOffset=None,
        monitorCutoff=0.80,
        pulseChopper="C34",
        slidingSum=None,
    ):

        self.data = namedtuple(
            "data",
            "intensities errors energies "
            "temps times name qVals "
            "qIdx observable "
            "observable_name norm "
            "tof diffraction",
        )

        self.scanList = parseString(scanList)

        self.sumScans = sumScans

        self.peakFindingMask = peakFindingMask
        self.detGroup = detGroup
        self.normalize = normalize
        self.observable = observable
        self.strip = strip
        self.tElastic = tElastic
        self.monitorOffset = monitorOffset
        self.monitorCutoff = monitorCutoff
        self.pulseChopper = pulseChopper
        self._refDist = {"C12": 34.300, "C34": 33.388}
        self.slidingSum = slidingSum

        self.dataList = []
        self.diffList = []
        self.errList = []
        self.tofList = []
        self.energyList = []
        self.qList = []
        self.tempList = []
        self.qzList = []

        self.outTuple = None

    def process(self):
        """Extract data from the provided files and
        reduce them using the given parameters.

        """
        self.dataList = []
        self.diffList = []
        self.errList = []
        self.energyList = []
        self.qList = []
        self.qzList = []
        self.tempList = []
        self.startTimeList = []
        self.monitor = []
        self.tofList = []

        for dIdx, dataFile in enumerate(self.scanList):
            dataset = h5py.File(dataFile, mode="r")

            data = dataset["entry0/data/PSD_data"][()]

            tof = dataset["entry0/instrument/PSD/time_of_flight"][()]
            tof = (np.arange(tof[1]) + 0.5) * tof[0] + tof[2]

            if "dataDiffDet" in dataset["entry0"].keys():
                diffraction = dataset["entry0/dataDiffDet/DiffDet_data"][()]
                self.diffList.append(diffraction.squeeze())

            self.name = dataset["entry0/subtitle"][(0)].astype(str)

            monitor = (
                dataset["entry0/monitor/data"][()].squeeze().astype("float")
            )
            self.monitor.append(monitor)

            # compute the time-of-flight for the monitor
            monTof = dataset["entry0/monitor/time_of_flight"][()]
            monTof = (np.arange(monTof[1]) + 0.5) * monTof[0] + monTof[2]

            wavelength = dataset["entry0/wavelength"][()]
            refEnergy = 1.3106479439885732e-40 / (wavelength * 1e-10) ** 2

            if dIdx == 0:
                data, center = self._alignGroups(data)
            else:
                data, center = self._alignGroups(data, center)

            if self.detGroup == "no":
                sampleToDec = dataset[
                    "entry0/instrument/PSD/distance_to_sample"
                ][()]
                tubeHeight = dataset[
                    "entry0/instrument/PSD/tubes_opening_ver"
                ][()]
                self.observable = "$q_z$"
                nbrDet = data.shape[0]
                nbrYPixels = int(data.shape[1])
                qZ = np.arange(nbrYPixels) - nbrYPixels / 2
                qZ *= tubeHeight / nbrYPixels
                qZ *= 4 * np.pi / (sampleToDec * wavelength)
                self.qzList = qZ
            else:
                # Sum along the selected region of the tubes
                data = self._detGrouping(data)
                nbrDet = data.shape[0]

            angles = np.array(
                [
                    dataset[
                        "entry0/instrument/PSD/PSD angle %s" % int(val + 1)
                    ][()]
                    for val in range(nbrDet)
                ]
            )

            nbrPSD = data.shape[0]
            data, angles = self._getSingleD(dataset, data, angles)
            nbrSingleD = data.shape[0] - nbrPSD

            angles = (
                4
                * np.pi
                * np.sin(np.pi * np.array(angles).squeeze() / 360)
                / wavelength
            )

            temp = dataset["entry0/sample/temperature"][()]
            time = parse(dataset["entry0/start_time"][0])

            self.dataList.append(np.copy(data))
            self.qList.append(np.copy(angles))
            self.tempList.append(np.copy(temp))
            self.startTimeList.append(time)

            dataset.close()

        if self.slidingSum is not None:
            self._slidingSum(self.slidingSum)

        if self.sumScans:
            self.monitor = [np.sum(np.array(self.monitor), 0)]
            self.dataList = [np.sum(np.array(self.dataList), 0)]
            self.diffList = [np.sum(np.array(self.diffList), 0)]

        # get the elastic time
        if self.tElastic is not None:
            tElastic = self.tElastic
        else:
            tElastic = tof[center]

        # convert time-of-flight to energies
        refDist = self._refDist[self.pulseChopper]
        refVel = np.sqrt(2 * refEnergy / 1.67493e-27)
        refTime = refDist / refVel
        dt = tof - tElastic
        velocities = refDist / (refTime + dt * 1e-6)
        energies = 1.67493e-27 * velocities ** 2 / 2
        energies -= refEnergy
        energies *= 6.241509e18 * 1e6
        refPos = np.argmin(energies ** 2)

        for idx, data in enumerate(self.dataList):
            for qIdx, qData in enumerate(data):
                if qIdx < nbrSingleD:
                    peak = self._findPeaks(qData)[0]
                else:
                    peak = center
                data[qIdx] = np.roll(qData, int(refPos - peak))

            errList = np.sqrt(data)
            if self.normalize:
                monCenter = self._findPeaks(self.monitor[idx])
                if self.monitorOffset is not None:
                    monRefPos = np.argmin((self.monitorOffset - monTof) ** 2)
                    self.monitor[idx] = np.roll(
                        self.monitor[idx],
                        int(monRefPos - monCenter),
                    )
                else:
                    self.monitor[idx] = np.roll(
                        self.monitor[idx], int(refPos - monCenter)
                    )

                np.place(self.monitor[idx], self.monitor[idx] <= 0, -np.inf)

                if self.detGroup == "no":
                    monitor = np.copy(self.monitor[idx][:, :, np.newaxis])
                else:
                    monitor = np.copy(self.monitor[idx])

                data = data / monitor
                errList = errList / monitor

            self.dataList[idx] = data[:, ::-1]
            self.monitor[idx] = self.monitor[idx][::-1]
            self.errList.append(errList[:, ::-1])
            self.energyList.append(np.copy(energies[::-1]))

        self.tofList = tof
        self._convertDataset()

    def _getSingleD(self, dataset, data, angles):
        """Determines the number of single detector used and add the data
        and angles to the existing data and angles arrays.

        :returns: data and angles arrays with the single detector data.

        """
        dataSD = []
        anglesSD = []

        keysSD = [
            "SD1 angle",
            "SD2 angle",
            "SD3 angle",
            "SD4 angle",
            "SD5 angle",
            "SD6 angle",
            "SD7 angle",
            "SD8 angle",
        ]

        for idx, key in enumerate(keysSD):
            angle = dataset["entry0/instrument/SingleD/%s" % key][()]

            if angle > 0:
                anglesSD.append(angle)
                dataSD.append(
                    dataset["entry0/instrument/SingleD/data"][(idx)].squeeze()
                )

        if self.observable == "$q_z$":
            tmpSD = np.zeros((len(dataSD), data.shape[1], data.shape[2]))
            tmpSD[:, 64] += np.array(dataSD)
            data = np.row_stack((tmpSD, data)).transpose(0, 2, 1)
        else:
            data = np.row_stack((np.array(dataSD).squeeze(), data))

        angles = np.concatenate((anglesSD, angles))

        return data, angles

    def _convertDataset(self):
        """Converts the data lists of the class into namedtuple(s)
        that can be directly used by nPDyn.

        """
        if self.strip is None:
            maxMon = self.monitor[0].max()
            monSignal = self.monitor[0] >= maxMon * self.monitorCutoff
            data = np.array(self.dataList)[:, :, monSignal]
            errors = np.array(self.errList)[:, :, monSignal]
            energies = self.energyList[0][monSignal]
            tof = self.tofList[monSignal]
        elif self.strip > 0:
            data = np.array(self.dataList)[:, :, self.strip : -self.strip]
            errors = np.array(self.errList)[:, :, self.strip : -self.strip]
            energies = self.energyList[0][self.strip : -self.strip]
            tof = self.tofList[self.strip : -self.strip]
        else:
            data = np.array(self.dataList)
            errors = np.array(self.errList)
            energies = self.energyList[0]
            tof = self.tofList

        # converts the times to hours
        times = np.array(self.startTimeList)

        if self.sumScans:
            temps = np.array([np.mean(self.tempList)])
            times = np.array([0.0])
        else:
            times = times - times[0]
            for idx, t in enumerate(times):
                times[idx] = t.total_seconds() / 3600
            temps = np.array(self.tempList).squeeze()

        if self.observable == "time":
            Y = times
        elif self.observable == "temperature":
            Y = temps
        elif self.observable == "$q_z$":
            data = data.transpose(3, 1, 2, 0).squeeze()
            errors = errors.transpose(3, 1, 2, 0).squeeze()
            Y = self.qzList
            midPos = int(data.shape[0] / 2)
            data = (data[:midPos][::-1] + data[midPos:]) / 2
            errors = (errors[:midPos][::-1] + errors[midPos:]) / 2
            Y = (-Y[:midPos][::-1] + Y[midPos:]) / 2

        self.outTuple = self.data(
            data,
            errors,
            energies,
            temps,
            times,
            self.name,
            self.qList[0],
            np.arange(self.qList[0].shape[0]),
            Y,
            self.observable,
            False,
            tof,
            np.array(self.diffList),
        )

    def _findPeaks(self, data):
        """Find the peak in each time-of-flight measurement."""
        data = np.array(data)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        # if detGroup is none, sum over all vertical positions to find
        # peaks more effectively
        if data.ndim == 3:
            data = data.sum(1)

        nbrChannels = data.shape[1]
        middle = int(nbrChannels / 2)
        window = (middle - nbrChannels / 5, middle + nbrChannels / 5)

        if self.peakFindingMask is None:
            mask = np.zeros_like(data)
            mask[:, int(window[0]) : int(window[1])] = 1

        maskedData = data * mask

        # Finds the peaks by using a Gaussian function to fit the data
        Gaussian = lambda x, normF, gauW, shift: (
            normF
            * np.exp(-((x - shift) ** 2) / (2 * gauW ** 2))
            / (gauW * np.sqrt(2 * np.pi))
        )
        try:
            gaussPeaks = []
            for qData in maskedData:
                qData = np.convolve(qData, np.ones(12), mode="same")
                errors = np.sqrt(qData)
                np.place(errors, errors == 0, np.inf)
                params = curve_fit(
                    Gaussian,
                    np.arange(nbrChannels),
                    qData,
                    sigma=errors,
                    bounds=(0.0, np.inf),
                    p0=[
                        qData.max(),
                        1,
                        nbrChannels / 2,
                    ],
                    max_nfev=10000,
                )
                gaussPeaks.append(params[0][2])
            return np.rint(gaussPeaks).astype(int)

        except RuntimeError:
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
                    distance=qData.size / 2,
                    prominence=0.5 * qData.max(),
                )[0]
                selData = qData[qPeaks]
                findPeaks.append(qPeaks[np.argmax(selData)].squeeze())

            return np.rint(
                0.5 * savGolPeaks + 0.5 * np.array(findPeaks)
            ).astype(int)

    def _detGrouping(self, data):
        """The method performs a sum along detector tubes using the provided
        range to be kept.

        It makes use of the :arg detGroup: argument.

        If the argument is a scalar, it sums over all
        values that are in the range
        [center of the tube - detGroup : center of the tube + detGroup].

        If the argument is a list of integers, then each element of the
        list is assumed to correspond to a range for each corresponding
        detector in ascending order.

        If the argument is a mantid-related xml file (a python string),
        the xml_detector_grouping module is then used to parse the xml
        file and the provided values are used to define the range.

        :arg data:  PSD data for the file being processed

        """
        if isinstance(self.detGroup, int):
            midPos = int(data.shape[1] / 2)
            data = data[:, midPos - self.detGroup : midPos + self.detGroup, :]
            out = np.sum(data, 1)

        elif isinstance(self.detGroup, (list, tuple, np.ndarray)):
            midPos = int(data.shape[1] / 2)

            out = np.zeros((data.shape[0], data.shape[2]))
            for detId, detData in enumerate(data):
                detData = detData[
                    midPos
                    - self.detGroup[detId] : midPos
                    + self.detGroup[detId]
                ]
                out[detId] = np.sum(detData, 0)

        elif isinstance(self.detGroup, str):
            numTubes = data.shape[0]
            xmlData = IN16B_XML(self.detGroup, numTubes)

            detRanges = xmlData.getPSDValues()

            out = np.zeros((data.shape[0], data.shape[2]))
            for detId, vals in enumerate(detRanges):
                out[detId] = data[detId, vals[0] : vals[1]].sum(0)

        elif self.detGroup is None:
            out = np.sum(data, 1)

        return out.astype("float")

    def _alignGroups(self, data, position=None):
        """Align the peaks along the z-axis of the detectors."""
        modData = data.sum(0)
        modData = np.array(
            [np.convolve(val, np.ones(20), mode="same") for val in modData.T]
        )
        nbrPixels = modData.shape[1]
        nbrChannels = modData.shape[0]
        window = (int(nbrChannels / 2 - 250), int(nbrChannels / 2 + 250))
        mask = np.zeros_like(modData)
        mask[window[0] : window[1]] = 1
        modData *= mask

        f = lambda x, d, offset, shift: (
            d / np.cos((x - shift) / x.size) + offset
        )

        res = curve_fit(
            f,
            np.arange(nbrPixels),
            modData.argmax(0),
            p0=[nbrPixels / 2, 1024, nbrPixels / 2],
        )

        maxPos = np.rint(f(np.arange(nbrPixels), *res[0])).astype(int)

        if position is None:
            center = np.min(maxPos)
        else:
            center = position

        for qIdx, qData in enumerate(data):
            data[qIdx] = np.array(
                [
                    np.roll(val, (center - maxPos)[idx])
                    for idx, val in enumerate(qData)
                ]
            )

        return data, center

    def _slidingSum(self, size):
        """Performs a sliding sum over scans.

        Parameters
        ----------
        size : int
            The size of the window to be summed over.
        data : list
            The list of datasets/scans.
        monitor : list
            The list of monitor data corresponding to each dataset.

        """
        outData = []
        outDiff = []
        outMonitor = []
        outTemp = []

        data = np.array(self.dataList)
        diff = np.array(self.diffList)
        monitor = np.array(self.monitor)
        temp = np.array(self.tempList)
        for idx, val in enumerate(data[:-size]):
            outData.append(np.sum(data[idx : idx + size], 0))
            outDiff.append(np.sum(diff[idx : idx + size], 0))
            outMonitor.append(np.sum(monitor[idx : idx + size], 0))
            outTemp.append(np.mean(temp[idx : idx + size], 0))

        self.dataList = outData
        self.diffList = outDiff
        self.monitor = outMonitor
        self.tempList = outTemp
        self.startTimeList = self.startTimeList[:-size]
