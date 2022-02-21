"""This module is used for importation of raw data from IN16B instrument.

"""
import numpy as np

from nPDyn.dataParsers.in16b_nexus import IN16B_nexus
import nPDyn.dataParsers.process_functions as proc


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
    monitorCutoff:
        Cutoff with respect to monitor maximum to discard data.
    pulseChopper : {'C12', 'C34'}
        Chopper pair that is used to define the pulse.

    """

    def __init__(
        self,
        scanList,
        sumScans=True,
        detGroup=None,
        normalize=True,
        strip=0,
        observable="time",
        tElastic=None,
        monitorCutoff=0.80,
        pulseChopper="C34",
        slidingSum=None,
    ):
        self.scanList = scanList
        self.sumScans = sumScans
        self.detGroup = detGroup
        self.normalize = normalize
        self.observable = observable
        self.strip = strip
        self.tElastic = tElastic
        self.monitorCutoff = monitorCutoff
        self.pulseChopper = pulseChopper
        self._refDist = {"C12": 34.300, "C34": 33.388}
        self.slidingSum = slidingSum

    def process(self, center=None, peaks=None, monPeaks=None):
        """Extract data from the provided files and
        reduce them using the given parameters.

        Parameters
        ----------
        center : int
            Position of the elastic signal along channels.
        peaks : 2D array
            Reference position of the peaks in dataset.
            Column vector with integer position for each q value.
        monPeaks : int
            Reference position of monitor peak signal.

        """
        dataset = IN16B_nexus(self.scanList, self.observable).process()

        aligned = [proc.alignGroups(val, center) for val in dataset]
        dataset = [val[0] for val in aligned]
        if center is None:
            center = int(np.mean([val[1] for val in aligned]))

        dataset = [proc.detGrouping(val, self.detGroup) for val in dataset]
        dataset = [proc.alignTo(val, center, peaks) for val in dataset]
        dataset = proc.mergeDataset(dataset, self.observable)

        if self.sumScans:
            dataset = proc.avgAlongObservable(dataset)
        elif self.slidingSum is not None:
            dataset = dataset.sliding_average(self.slidingSum)
            dataset.monitor = np.mean(dataset.monitor, 0)
        else:
            dataset.monitor = np.mean(dataset.monitor, 0)

        dataset = proc.convertChannelsToEnergy(
            dataset,
            "bats",
            refDist=self._refDist[self.pulseChopper],
            tElastic=self.tElastic,
        )

        toKeep = np.argwhere(
            dataset.monitor >= self.monitorCutoff * dataset.monitor.max()
        ).flatten()
        dataset = dataset.take(toKeep, dataset.axes.index("energies"))
        dataset.monitor = dataset.monitor[toKeep]

        if self.normalize:
            dataset = proc.normalizeToMonitor(dataset)

        dataset = dataset.take(
            np.arange(self.strip, dataset.energies.size - self.strip),
            dataset.axes.index("energies"),
        )

        return dataset

    def getReference(self):
        """Process files to obtain reference values for elastic signal."""
        dataset = IN16B_nexus(self.scanList, self.observable).process()

        aligned = [proc.alignGroups(val) for val in dataset]
        dataset = [val[0] for val in aligned]
        center = int(np.mean([val[1] for val in aligned]))

        dataset = [proc.detGrouping(val, self.detGroup) for val in dataset]

        dataset = proc.mergeDataset(dataset, self.observable)

        dataset = proc.sumAlongObservable(dataset)

        peaks = proc.findPeaks(dataset)
        monPeaks = proc.findPeaks(dataset.monitor)

        return center, peaks, monPeaks
