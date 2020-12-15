import numpy as np


def subplotsFormat(
    caller,
    sharex=False,
    sharey=False,
    projection=None,
    params=False,
    FWS=False,
):
    """This method is used to try to determine the best number of
    rows and columns for plotting. Depending on the size of the
    fileIdxList, the plot will have a maximum of subplots per row,
    typically around 4-5 and the required number of rows.
    :arg sharex:     matplotlib's parameter for x-axis sharing
    :arg sharey:     matplotlib's parameter for y-axis sharing
    :arg projection: projection type for subplots (None, '3d',...)
                     (optional, default None)
    :arg params:     if True, use size of paramsNames instead of
                     fileIdxList
    :arg FWS:        if True, use numbers of energy offsets in
                     fixed-window scans instead
    :returns: axis list from figure.subplots method of matplotlib
    """
    # Getting number of necessary subplots
    if params and not FWS:
        listSize = len(caller.dataset[0].params[0].keys())
    elif FWS and not params:
        listSize = caller.dataset[0].data.intensities.shape[2]
    else:
        listSize = len(caller.dataset)

    # Generating the subplots
    if listSize == 1:
        caller.figure.subplots(
            1,
            listSize,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
        )

    if listSize != 1 and listSize < 4:
        caller.figure.subplots(
            1,
            listSize,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
        )

    if listSize == 4:
        caller.figure.subplots(
            2,
            2,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
        )

    if listSize > 4 and listSize <= 9:
        caller.figure.subplots(
            int(np.ceil(listSize / 3)),
            3,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
        )

    if listSize > 9 and listSize <= 12:
        caller.figure.subplots(
            int(np.ceil(listSize / 4)),
            4,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
        )

    if listSize > 12:
        caller.figure.subplots(
            int(np.ceil(listSize / 5)),
            5,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
        )

    # Removing unnecessary axes
    for idx, subplot in enumerate(caller.figure.axes):
        if idx >= listSize:
            caller.figure.delaxes(subplot)

    ax = caller.figure.axes

    if listSize == 1:
        ax = np.array(ax)

    return ax


def subplotsFormatWithColorBar(
    caller, sharex=False, sharey=False, projection=None, params=False
):
    """This method is used to try to determine the best number of
    rows and columns for plotting. Depending on the size of the
    fileIdxList, the plot will have a maximum of subplots per row,
    typically around 4-5 and the required number of rows.
    Axes are added to plot colorbars as well, so that the number of
    columns will be twice the number required initially by the data.
    :arg sharex:     matplotlib's parameter for x-axis sharing
    :arg sharey:     matplotlib's parameter for y-axis sharing
    :arg projection: projection type for subplots (None, '3d',...)
                     (optional, default None)
    :arg params:     if True, use size of paramsNames instead of
                     fileIdxList
    :returns: axis list from figure.subplots method of matplotlib
    """

    # Getting number of necessary subplots
    if params:
        listSize = len(caller.dataset[0].params[0].keys())
    else:
        listSize = len(caller.dataset)

    # Generating the subplots
    if listSize == 1:
        caller.figure.subplots(
            1,
            2,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
            gridspec_kw={"width_ratios": [20, 1]},
        )

    if listSize != 1 and listSize < 4:
        caller.figure.subplots(
            1,
            2 * listSize,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
            gridspec_kw={"width_ratios": listSize * [20, 1]},
        )

    if listSize == 4:
        caller.figure.subplots(
            2,
            4,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
            gridspec_kw={"width_ratios": 2 * [20, 1]},
        )

    if listSize > 4 and listSize <= 9:
        nbrRows = int(np.ceil(listSize / 3))
        caller.figure.subplots(
            nbrRows,
            6,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
            gridspec_kw={"width_ratios": 3 * [20, 1]},
        )

    if listSize > 9 and listSize <= 12:
        nbrRows = int(np.ceil(listSize / 4))
        caller.figure.subplots(
            nbrRows,
            8,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
            gridspec_kw={"width_ratios": 4 * [20, 1]},
        )

    if listSize > 12:
        nbrRows = int(np.ceil(listSize / 5))
        caller.figure.subplots(
            nbrRows,
            10,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": projection},
            gridspec_kw={"width_ratios": 5 * [20, 1]},
        )

    # Removing unnecessary axes
    for idx, subplot in enumerate(caller.figure.axes):
        if idx >= 2 * listSize:
            caller.figure.delaxes(subplot)

    ax = caller.figure.axes

    if listSize > 1:
        ax0 = ax[::2]
        ax1 = ax[1::2]
    else:
        ax0 = [ax[0]]
        ax1 = [ax[1]]

    return ax0, ax1
