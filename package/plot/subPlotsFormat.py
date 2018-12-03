import matplotlib.pyplot as plt
import numpy as np


def subplotsFormat(caller, sharex=False, sharey=False, projection=None, params=False, res=False,
                        resParams=False):
    """ This method is used to try to determine the best number of rows and columns for plotting.
        Depending on the size of the fileIdxList, the plot will have a maximum of subplots per row,
        typically around 4-5 and the required number of columns.

        Input: sharex       -> matplotlib's parameter for x-axis sharing
               sharey       -> matplotlib's parameter for y-axis sharing
               projection   -> projection type for subplots (None, '3d',...) (optional, default None)
               params       -> if True, use size of paramsList instead of fileIdxList
               res          -> if True, use size of resFile list instead of fileIdxList
               resParams    -> if True, use size of resParams list instead of fileIdxList
        
        Output: axis list from figure.subplots method of matplotlib """

    #_Getting number of necessary subplots
    if params:
        listSize = caller.paramsList[0][0].x.size
    if resParams:
        listSize = len(caller.resParams[0][0][0])
    if res:
        listSize = len(caller.resFiles)
    if not params and not resParams and not res:
        listSize = len(caller.fileIdxList)


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

        #_Setting the shared axis, useful for listSize not multiple of 3
        if sharex and not resParams:
            for idx in range(listSize-1, listSize-4, -1):
                caller.figure.axes[idx].set_xticklabels(caller.dataFiles)


    if listSize > 9 and listSize <= 12:
        ax = caller.figure.subplots(int(np.ceil(listSize / 4)), 4, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()

        #_Setting the shared axis, useful for listSize not multiple of 3
        if sharex and not resParams:
            for idx in range(listSize-1, listSize-5, -1):
                caller.figure.axes[idx].set_xticklabels(caller.dataFiles)



    if listSize > 12:
        ax = caller.figure.subplots(int(np.ceil(listSize / 5)), 5, sharex, sharey, 
                                                            subplot_kw={'projection':projection}).flatten()
    
        #_Setting the shared axis, useful for listSize not multiple of 3
        if sharex and not resParams:
            for idx in range(listSize-1, listSize-6, -1):
                caller.figure.axes[idx].set_xticklabels(caller.dataFiles)


    #_Removing unecessary axes
    for idx, subplot in enumerate(ax):
        if idx >= listSize:
            caller.figure.delaxes(subplot)

    return caller.figure.axes


