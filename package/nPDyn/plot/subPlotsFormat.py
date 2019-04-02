import matplotlib.pyplot as plt
import numpy as np


def subplotsFormat(caller, sharex=False, sharey=False, projection=None, params=False):
    """ This method is used to try to determine the best number of rows and columns for plotting.
        Depending on the size of the fileIdxList, the plot will have a maximum of subplots per row,
        typically around 4-5 and the required number of columns.

        Input: sharex       -> matplotlib's parameter for x-axis sharing
               sharey       -> matplotlib's parameter for y-axis sharing
               projection   -> projection type for subplots (None, '3d',...) (optional, default None)
               params       -> if True, use size of paramsNames instead of fileIdxList
        
        Output: axis list from figure.subplots method of matplotlib """

    #_Getting number of necessary subplots
    if params:
        listSize = len(caller.dataset[0].paramsNames)
    else:
        listSize = len(caller.dataset)


    #_Generating the subplots
    if listSize == 1:
        ax = np.array([caller.figure.subplots(1, listSize, sharex, sharey,
                                                subplot_kw={'projection':projection})])

    if listSize !=1 and listSize < 4:
        ax = caller.figure.subplots(1, listSize, sharex, sharey,
                                                subplot_kw={'projection':projection})

    if listSize == 4:
        ax = caller.figure.subplots(2, 2, sharex, sharey, subplot_kw={'projection':projection}).flatten()

    if listSize > 4 and listSize <= 9:
        ax = caller.figure.subplots(int(np.ceil(listSize / 3)), 3, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()


    if listSize > 9 and listSize <= 12:
        ax = caller.figure.subplots(int(np.ceil(listSize / 4)), 4, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()


    if listSize > 12:
        ax = caller.figure.subplots(int(np.ceil(listSize / 5)), 5, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()
    

    #_Removing unecessary axes
    for idx, subplot in enumerate(ax):
        if idx >= listSize:
            caller.figure.delaxes(subplot)

    return caller.figure.axes







def subplotsFormatWithColorBar(caller, sharex=False, sharey=False, projection=None, params=False):
    """ This method is used to try to determine the best number of rows and columns for plotting.
        Depending on the size of the fileIdxList, the plot will have a maximum of subplots per row,
        typically around 4-5 and the required number of columns.

        Input: sharex       -> matplotlib's parameter for x-axis sharing
               sharey       -> matplotlib's parameter for y-axis sharing
               projection   -> projection type for subplots (None, '3d',...) (optional, default None)
               params       -> if True, use size of paramsNames instead of fileIdxList
        
        Output: axis list from figure.subplots method of matplotlib """

    #_Getting number of necessary subplots
    if params:
        listSize = len(caller.dataset[0].paramsNames)
    else:
        listSize = len(caller.dataset)


    #_Generating the subplots
    if listSize == 1:
        ax = np.array([caller.figure.subplots(1, 2*listSize, sharex, sharey,
                                                subplot_kw={'projection':projection})])

    if listSize !=1 and listSize < 4:
        ax = caller.figure.subplots(1, 2*listSize, sharex, sharey,
                                                subplot_kw={'projection':projection})

    if listSize == 4:
        ax = caller.figure.subplots(2, 4, sharex, sharey, subplot_kw={'projection':projection}).flatten()

    if listSize > 4 and listSize <= 9:
        ax = caller.figure.subplots(int(np.ceil(listSize / 3)), 6, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()


    if listSize > 9 and listSize <= 12:
        ax = caller.figure.subplots(int(np.ceil(listSize / 4)), 8, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()


    if listSize > 12:
        ax = caller.figure.subplots(int(np.ceil(listSize / 5)), 10, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()
    

    #_Removing unecessary axes
    for idx, subplot in enumerate(ax):
        if idx >= 2*listSize:
            caller.figure.delaxes(subplot)

    return caller.figure.axes


