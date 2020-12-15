import numpy as np


def binData(data, binS, axis):
    """This method takes a dataset, and return the binned one.
    This cannot be used with FWS data processed from hdf5 files
    with axis being set to 'energies'.

    Parameters
    ----------
    data : :class:`BaseType` or any nPDyn dataType
        dataset to be binned
    binS : int
        bin size
    axis : {'energies', 'observable'}
        axis to be used for binning

    """
    binS = int(binS)

    # Get the axis to be binned
    if axis == "energies":
        tmpX = data.energies
    else:
        tmpX = data.observable

    # Computes the loop number needed for binning
    loopSize = int(tmpX.size / binS)

    tmpIntensities = data.intensities
    tmpErrors = data.errors

    # Iterates binning, basically computes mean value inside
    # a bin, then continue to the next bin
    for i in range(loopSize):
        tmpX[i] = np.mean(tmpX[i * binS : i * binS + binS])

        if axis == "energies":
            intSlice = tmpIntensities[:, :, i * binS : i * binS + binS]
            errSlice = tmpErrors[:, :, i * binS : i * binS + binS]

            # For intensities and errors, discard values that
            # equal 0 or inf from bin averaging
            for row in range(intSlice.shape[0]):
                for col in range(intSlice.shape[1]):
                    if intSlice[row, col][intSlice[row, col] != 0.0].size != 0:
                        tmpIntensities[row, col, i] = np.mean(
                            intSlice[row, col][intSlice[row, col] != 0.0]
                        )
                    else:
                        tmpIntensities[row, col, i] = 0

                    if (
                        errSlice[row, col][errSlice[row, col] != np.inf].size
                        != 0
                    ):
                        tmpErrors[row, col, i] = np.mean(
                            errSlice[row, col][errSlice[row, col] != np.inf]
                        )
                    else:
                        tmpErrors[row, col, i] = np.inf

        else:
            intSlice = tmpIntensities[i * binS : i * binS + binS]
            errSlice = tmpErrors[i * binS : i * binS + binS]

            tmpIntensities[i] = np.mean(intSlice, 0)
            tmpErrors[i] = np.mean(errSlice, 0)

    # Remove unnecessary data after binning
    if axis == "energies":
        data = data._replace(
            energies=tmpX[:loopSize],
            intensities=tmpIntensities[:, :, :loopSize],
            errors=tmpErrors[:, :, :loopSize],
        )
    elif axis == "observable":
        data = data._replace(
            observable=tmpX[:loopSize],
            intensities=tmpIntensities[:loopSize],
            errors=tmpErrors[:loopSize],
        )

    return data
