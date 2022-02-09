""" Parser for .nxs files from IN16B

"""

from dateutil.parser import parse

import h5py
import numpy as np

from collections import namedtuple

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from nPDyn.dataParsers.stringParser import parseString
from nPDyn.dataParsers.instruments.in16b import getDiffDetXAxis

from nPDyn import Sample


class IN16B_nexus:
    """This class can handle raw data from IN16B at
    the ILL in the hdf5 format.

    :arg scanList:       a string or a list of files to be read
                         and parsed to extract the data.
                         It can be a path to a folder as well.
    :arg sumScans:       whether the scans should be summed or not
    :arg alignPeaks:     if True, will try to align peaks of the monitor
                         with the ones from the PSD data.
    :arg peakFindWindow: the size (in number of channels) of the window
                         to find and align the peaks
                         of the monitor to the peaks of the data.
    :arg detGroup:       detector grouping, i.e. the channels that are
                         summed over along the
                         position-sensitive detector tubes. It can be an
                         integer, then the same number
                         is used for all detectors, where the integer
                         defines a region (middle of the
                         detector +/- detGroup). It can be a list of
                         integers, then each integers of
                         the list should corresponds to a detector.
                         Or it can be a string, defining
                         a path to an xml file as used in Mantid.
                         If set to `no`, no detector gouping is performed
                         and the data represents the signal for each
                         pixel on the detectors. In this case, the
                         observable become the momentum transfer q in
                         the vertical direction.
    :arg normalize:      whether the data should be normalized
                         to the monitor
    :arg observable:     the observable that might be changing over scans.
                         It can be `time`, `temperature`
    :arg offset:         If not None, only the data with energy offset
                         that equals the given value will be imported.

    """

    def __init__(
        self,
        scanList,
        observable="time",
    ):

        self.scanList = parseString(scanList)
        self.observable = observable

        self.sample = None

    def process(self):
        """Extract data from the provided files and reduce
        them using the given parameters.

        """
        self.dataList = []
        self.diffList = []
        self.diffQList = []
        self.errList = []
        self.monitor = []
        self.energyList = []
        self.qList = []
        self.qzList = []
        self.tempList = []
        self.pressureList = []
        self.startTimeList = []

        out = []
        for dataFile in self.scanList:
            print("Processing file %s...             " % dataFile, end="\r")
            dataset = h5py.File(dataFile, mode="r")
            self.name = dataset["entry0/subtitle"][(0)].astype(str)

            data = dataset["entry0/data/PSD_data"][()]

            maxDeltaE = dataset[
                "entry0/instrument/Doppler/maximum_delta_energy"
            ][(0)]
            self.energyList.append(maxDeltaE)

            wavelength = dataset["entry0/wavelength"][()]

            # Gets the monitor data
            monitor = (
                dataset["entry0/monitor/data"][()].squeeze().astype("float")
            )
            self.monitor.append(np.copy(monitor))

            # process the diffraction detectors
            if "dataDiffDet" in dataset["entry0"].keys():
                diffraction = dataset["entry0/dataDiffDet/DiffDet_data"][()]
                self.diffQList = getDiffDetXAxis(wavelength)
                self.diffList.append(diffraction.squeeze().mean(-1))

            tof = None
            monTof = None
            if "time_of_flight" in dataset["entry0/instrument/PSD"].keys():
                tof = dataset["entry0/instrument/PSD/time_of_flight"][()]
                tof = (np.arange(tof[1]) + 0.5) * tof[0] + tof[2]
                monTof = dataset["entry0/monitor/time_of_flight"][()]
                monTof = (np.arange(monTof[1]) + 0.5) * monTof[0] + monTof[2]

            # get q-values along vertical axis
            nbrDet = data.shape[0]
            nbrYPixels = int(data.shape[1])
            sampleToDec = (
                dataset["entry0/instrument/PSD/distance_to_sample"][()] * 10
            )
            tubeHeight = dataset["entry0/instrument/PSD/tubes_size"][()]
            qZ = np.arange(nbrYPixels) - nbrYPixels / 2
            qZ *= tubeHeight / nbrYPixels
            qZ *= 4 * np.pi / (sampleToDec * wavelength)
            self.qzList = qZ

            # process detector angles
            nbrDet = data.shape[0]
            angles = [
                dataset["entry0/instrument/PSD/PSD angle %s" % int(val + 1)][
                    ()
                ]
                for val in range(nbrDet)
            ]
            data, angles, nbrSD = self._getSingleD(dataset, data, angles)
            angles = (
                4
                * np.pi
                * np.sin(np.pi * np.array(angles).squeeze() / 360)
                / wavelength
            )
            self.nbrSD = nbrSD

            # extract sample data
            temp = dataset["entry0/sample/temperature"][()]
            pressure = dataset["entry0/sample/pressure"][()]
            time = parse(dataset["entry0/start_time"][0])

            self.dataList.append(np.copy(data))
            self.startTimeList.append(time)
            self.qList.append(np.copy(angles))
            self.tempList.append(np.copy(temp))
            self.pressureList.append(np.copy(pressure))

            dataset.close()

            out.append(
                Sample(
                    data,
                    errors=np.sqrt(data),
                    q=angles,
                    name=self.name,
                    time=time,
                    temperature=temp,
                    pressure=pressure,
                    energies=maxDeltaE if np.isfinite(maxDeltaE) else -1,
                    diffraction=self.diffList,
                    qdiff=self.diffQList,
                    observable=self.observable,
                    channels=np.arange(data.shape[1]),
                    wavelength=wavelength,
                    monitor=monitor,
                    qz=qZ,
                    axes=["q", "channels", "qz"],
                    nbr_single_dec=nbrSD,
                    tof=tof,
                    monTof=tof,
                )
            )

        return out

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

        tmpSD = np.zeros((len(dataSD), data.shape[1], data.shape[2]))
        tmpSD[:, int(data.shape[1] / 2)] += np.array(np.array(dataSD))
        data = np.row_stack((tmpSD, data)).transpose(0, 2, 1)

        angles = np.concatenate((anglesSD, angles))

        return data, angles, len(dataSD)
