"""

Classes
^^^^^^^

"""
import numpy as np

from nPDyn.dataParsers.in16b_nexus import IN16B_nexus
import nPDyn.dataParsers.process_functions as proc


class IN16B_FWS:
    """This class can handle raw E/IFWS data from IN16B at
    the ILL in the hdf5 format.

    :arg scanList:       a string or a list of files to be read
                         and parsed to extract the data.
                         It can be a path to a folder as well.
    :arg sumScans:       whether the scans should be summed or not
    :arg alignPeaks:     if True, will try to align peaks of the monitor
                         with the ones from the PSD data.
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
        sumScans=False,
        alignPeaks=True,
        detGroup=None,
        normalize=True,
        observable="time",
        offset=None,
    ):

        self.scanList = scanList
        self.sumScans = sumScans
        self.alignPeaks = alignPeaks
        self.detGroup = detGroup
        self.normalize = normalize
        self.observable = observable
        self.offset = offset

    def process(self):
        """Extract data from the provided files and reduce
        them using the given parameters.

        """
        dataset = IN16B_nexus(self.scanList, self.observable).process()

        dataset = [proc.detGrouping(data, self.detGroup) for data in dataset]

        if self.normalize:
            peaks = [None] * len(dataset)
            monPeaks = [None] * len(dataset)
            if self.alignPeaks:
                peaks = [proc.findPeaksFWS(data) for data in dataset]
                monPeaks = [
                    proc.findPeaksFWS(data.monitor) for data in dataset
                ]

            dataset = [
                proc.normalizeToMonitor(data, peaks[idx], monPeaks[idx], True)
                for idx, data in enumerate(dataset)
            ]

        dataset = proc.mergeDataset(dataset, self.observable)

        if self.sumScans:
            dataset = dataset.mean(0)[None]

        return dataset
