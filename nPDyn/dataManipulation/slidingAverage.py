import numpy as np


def slidingAverage(data, windowLength):
    """Sliding average of the dataset along the observable axis.

    Parameters
    ----------
    data : :class:`BaseType` or any nPDyn dataType
        dataset to be used
    windowLength : int
        size of the window for averaging

    """
    window = np.ones(windowLength)
    nbrIter = data.observable.size - windowLength + 1

    outIntensities = []
    outErrors = []
    for i in range(nbrIter):
        intensities = data.intensities[i : i + windowLength]
        errors = data.errors[i : i + windowLength]

        intensities = np.array(np.mean(intensities, 0))
        errors = np.array(np.sqrt(np.sum(errors ** 2, 0)))
        outIntensities.append(intensities)
        outErrors.append(errors)

    observable = np.convolve(data.observable, window, mode="valid")

    data = data._replace(
        intensities=np.array(outIntensities),
        errors=np.array(outErrors) / windowLength,
        observable=observable / windowLength,
    )

    return data
