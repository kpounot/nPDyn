import numpy as np

from collections import namedtuple


def convert(datafile, FWS=None):
    """This method takes a single dataFile as argument and
    returns the corresponding dataSet.

    Then the result is stored as a namedtuple containing several
    members (all being numpy arrays).
        - intensities   - 3D array of counts values for each frame
                          (axis 0), q-value (axis 1) and energy channels
                          (axis 2)
        - errors        - 3D array of errors values for each frame
                          (axis 0), q-value (axis 0) and energy channels
                          (axis 2)
        - energies      - 1D array of energy offsets used
        - temps         - 2D array of temperatures, the first dimension
                          is of size 1 for QENS, and of the same size
                          as the number of energy offsets for FWS. The
                          second dimensions represents the frames
        - times         - same structure as for temps but representing
                          the time
        - name          - name that is stored in the 'subtitle' entry
        - qVals         - 1D array of q-values used
        - selQ          - same as qVals, used later to define a q-range
                          for analysis
        - qIdx          - same as selQ but storing the indices
        - observable    - data for the observable used for data series
                          ('time' or 'temperature')
        - observable_name - name of the observable used for data series
        - norm           - boolean, whether data were normalized or not

    """

    datafile = datafile

    dataTuple = namedtuple(
        "dataTuple",
        "intensities errors energies "
        "temps times name qVals "
        "qIdx observable "
        "observable_name norm",
    )

    with open(datafile) as fileinput:
        data = fileinput.read().splitlines()

    # Get the index of the data corresponding to each q-value
    indexList = []
    for i, value in enumerate(data):
        if value == data[0]:
            indexList.append(i)

    data = [val.split() for i, val in enumerate(data)]
    qVals = []
    intensities = []
    X = []  # can be energies or temperatures
    errors = []
    temps = None
    times = np.array([0.0])
    obs = None
    obs_name = ""
    energies = None

    for i, value in enumerate(indexList):
        if i < len(indexList) - 1:  # While we're not at the last angle entries
            qVals.append(data[value + 2][0])
            X.append(
                [
                    val[0]
                    for j, val in enumerate(data)
                    if indexList[i] + 3 < j < indexList[i + 1]
                ]
            )
            intensities.append(
                [
                    val[1]
                    for j, val in enumerate(data)
                    if indexList[i] + 3 < j < indexList[i + 1]
                ]
            )
            errors.append(
                [
                    val[2]
                    for j, val in enumerate(data)
                    if indexList[i] + 3 < j < indexList[i + 1]
                ]
            )
        else:  # Used for the last entries
            qVals.append(data[value + 2][0])
            X.append(
                [val[0] for j, val in enumerate(data) if j > indexList[i] + 3]
            )
            intensities.append(
                [val[1] for j, val in enumerate(data) if j > indexList[i] + 3]
            )
            errors.append(
                [val[2] for j, val in enumerate(data) if j > indexList[i] + 3]
            )

    # Converting to floats
    qVals = np.array(qVals).astype(float)
    X = np.array(X[0]).astype(float)
    intensities = np.array(intensities).astype(float)[np.newaxis, :, :]
    errors = np.array(errors).astype(float)[np.newaxis, :, :]

    if X[X < 0].size != 0:  # X probably refers to energies (QENS)
        energies = X
        obs = np.array([0])
        obs_name = ""
        temps = np.array([0.0])
    else:  # X probably refers to temperatures
        energies = np.array([0.0])
        temps = X
        obs = temps
        obs_name = "temperature"

    # Creating the named tuple (no temp with .inx)
    dataSet = dataTuple(
        intensities.T,
        errors.T,
        energies,
        temps,
        times,
        obs_name,
        qVals,
        np.arange(qVals.size),
        obs,
        obs_name,
        False,
    )

    return dataSet
