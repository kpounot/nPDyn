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

    intensities = np.array(
        [
            np.mean(data.intensities[i : i + windowLength], 0)
            for i in range(nbrIter)
        ]
    )
    errors = np.array(
        [
            np.sqrt(np.sum(data.errors[i : i + windowLength] ** 2, 0))
            for i in range(nbrIter)
        ]
    )
    observable = np.convolve(data.observable, window, mode="valid")

    data = data._replace(
        intensities=intensities,
        errors=errors / windowLength,
        observable=observable / windowLength,
    )

    return data
