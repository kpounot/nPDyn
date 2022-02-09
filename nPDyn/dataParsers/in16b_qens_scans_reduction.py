"""This module is used for importation of raw data from IN16B instrument.

"""
import numpy as np

from nPDyn.dataParsers.in16b_nexus import IN16B_nexus
import nPDyn.dataParsers.process_functions as proc


class IN16B_QENS:
    """This class can handle raw QENS data from IN16B
    at the ILL in the hdf5 format.

    :arg scanList:    a string or a list of files to be read and
                      parsed to extract the data.
                      It can be a path to a folder as well.
    :arg sumScans:    whether the scans should be summed or not.
    :arg unmirroring: whether the data should be unmirrored or not.
    :arg vanadiumRef: if :arg unmirroring: is True, then the peaks
                      positions are identified
                      using the data provided with this argument.
                      If it is None, then the peaks positions are
                      identified using the data in scanList.
    :arg detGroup:    detector grouping, i.e. the channels that
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
    :arg normalize:   whether the data should be normalized to the
                      monitor
    :arg strip:       an integer defining the number of points that
                      are ignored at each extremity of the spectrum.
    :arg observable:  the observable that might be changing over scans.
                      It can be `time` or `temperature`

    """

    def __init__(
        self,
        scanList,
        sumScans=True,
        unmirroring=True,
        vanadiumRef=None,
        detGroup=None,
        normalize=True,
        strip=25,
        observable="time",
    ):
        self.scanList = scanList
        self.sumScans = sumScans
        self.unmirroring = unmirroring
        self.vanadiumRef = vanadiumRef
        self.detGroup = detGroup
        self.normalize = normalize
        self.observable = observable
        self.strip = strip

    def process(self):
        """Extract data from the provided files and
        reduce them using the given parameters.

        """
        dataset = IN16B_nexus(self.scanList, self.observable).process()

        dataset = [proc.detGrouping(data, self.detGroup) for data in dataset]

        if self.unmirroring:
            refPeaks = None
            if self.vanadiumRef:
                leftPeaks = None
                rightPeaks = None
                vana = IN16B_nexus(self.vanadiumRef, self.observable).process()
                vana = [proc.detGrouping(data, self.detGroup) for data in vana]
                vana = proc.mergeDataset(vana)
                vana = proc.sumAlongObservable(vana)
                leftPeaks = proc.findPeaks(
                    vana[:, :, : int(vana.shape[2] / 2)]
                )
                rightPeaks = proc.findPeaks(
                    vana[:, :, int(vana.shape[2] / 2) :]
                )
                refPeaks = np.column_stack([leftPeaks, rightPeaks])
            dataset = [proc.unmirror(val, refPeaks) for val in dataset]

        dataset = proc.mergeDataset(dataset, self.observable)

        dataset = proc.convertChannelsToEnergy(dataset, "qens")

        if self.sumScans:
            dataset = proc.sumAlongObservable(dataset)

        if self.normalize:
            dataset = proc.normalizeToMonitor(dataset)

        dataset = dataset.take(
            np.arange(self.strip, dataset.energies.size - self.strip),
            dataset.axes.index("energies"),
        )

        return dataset
