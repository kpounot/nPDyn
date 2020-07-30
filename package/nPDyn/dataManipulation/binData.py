import numpy as np


def binData(data, binS):
    """ This method takes a dataSet, and return the binned one.
        This cannot be used with FWS data processed from hdf5 files.

        :arg data: dataSet to be binned
        :arg binS: bin size to reduce the dataset

    """

    binS = int(binS)

    # Compute the loop number needed for binning
    loopSize = int(data.intensities.shape[1] / binS)

    # Get the arrays to be binned
    tempX           = data.X
    tempIntensities = data.intensities
    tempErrors      = data.errors

    # Iterate binning, basically computes values mean inside
    # a bin, then shift to the next bin
    for i in range(loopSize):
        tempX[i] = np.mean(tempX[i * binS:i * binS + binS])

        # For intensities and errors, discard values that
        # equal 0 or inf from bin averaging
        IntSlice = tempIntensities[:, i * binS:i * binS + binS]
        ErrSlice = tempErrors[:, i * binS:i * binS + binS]
        for row in range(IntSlice.shape[0]):
            if IntSlice[row][IntSlice[row] != 0.0].size != 0:
                tempIntensities[row, i] = np.mean(
                    IntSlice[row][IntSlice[row] != 0.0])
            else:
                tempIntensities[row, i] = 0


            if ErrSlice[row][ErrSlice[row] != np.inf].size != 0:
                tempErrors[row, i] = np.mean(
                    ErrSlice[row][ErrSlice[row] != np.inf])
            else:
                tempErrors[row, i] = np.inf




    # Remove unecessary data after binning
    data = data._replace(X           = tempX[:loopSize],
                         intensities = tempIntensities[:, :loopSize],
                         errors      = tempErrors[:, :loopSize])

    return data
