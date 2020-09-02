"""

Classes
^^^^^^^

"""

import numpy as np

from nPDyn.dataTypes.baseType import BaseType
from nPDyn.fileFormatParser import guessFileFormat, readFile
from nPDyn.dataTypes import fECType

try:
    from ..lib.pyabsco import py_absco_slab, py_absco_tube
except ImportError:
    print('\nAbsorption correction libraries are not available. '
          'Paalman_Pings correction cannot be used.\n '
          'Verify that GSL libraries are available on this computer, '
          'and the path was correctly \n '
          'set in the setup.cfg file during package installation.')
    pass




class FWSType(BaseType):
    """ This class inherits from :class:`baseType` class.

    """

    def __init__(self, fileName, data=None, rawData=None,
                 resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


        self.Ylabel = 'Time [h]'

        self.globalFit = True


    def importData(self, fileFormat=None):
        """ Extract data from file and store them in *data*
            and *rawData* attributes.

            If no fileFormat is given, tries to guess it, try hdf5
            format if format cannot be guessed.

            This is redefined from :class:`baseType`, to take into
            account FWS data specificities.

        """

        if fileFormat:
            data = readFile(fileFormat, self.fileName, True)
        else:
            data = guessFileFormat(self.fileName, True)

        self.data       = data
        self.rawData = self.data._replace(
            qVals       = np.copy(self.data.qVals),
            selQ        = np.copy(self.data.selQ),
            times       = np.copy(self.data.times),
            intensities = np.copy(self.data.intensities),
            errors      = np.copy(self.data.errors),
            temps       = np.copy(self.data.temps),
            norm        = False,
            qIdx        = np.copy(self.data.qIdx),
            energies    = np.copy(self.data.energies),
            observable  = np.copy(self.data.observable),
            observable_name = np.copy(self.data.observable_name))



    def binData(self, binSize):
        """ Binning in energy dimension does nothing as
            it make no sense for FWS data type. """

        return



    def normalize(self):
        """ Normalizes data using a list of scaling factors
            from resolution function fit. It assumes that the
            scaling factor is in first position in the model parameters.
            There should be as many normalization factors as q-values in data.

        """

        normFList = np.array([params[0][0] for params in self.resData.params])

        # Applying normalization
        self.data = self.data._replace(
            intensities = (self.data.intensities
                           / normFList[np.newaxis, :, np.newaxis]),
            errors      = (self.data.errors
                           / normFList[np.newaxis, :, np.newaxis]),
            norm        = True)



    def normalize_usingLowTemp(self, nbrBins):
        """ Normalizes data using low temperature signal.
            An average is performed over the given
            number of bins for each q value and data
            are divided by the result.

        """

        normFList = np.mean(
            self.data.intensities[:nbrBins, :, :], axis=0)[np.newaxis, :, :]

        self.data = self.data._replace(
            intensities = self.data.intensities / normFList,
            errors      = self.data.errors / normFList,
            norm        = True)

        self._normFList = normFList




    def subtractEC(self, scaleFactor=0.95, useModel=True):
        """ Use the assigned empty cell data for substraction to loaded data.

            Empty cell data are scaled using the given scaleFactor
            prior to substraction.

        """

        if isinstance(self.ECData, fECType.fECType):
            ECFunc = self.ECData.data.intensities


        else:  # Assumes full QENS with fitted model
            # Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append(self.ECData.model(self.data.X,
                                                *self.ECData.params[qIdx][0]))

            ECFunc = np.array(ECFunc)


        # If data are normalized, uses the same normalization
        # factor for empty cell data
        if self.data.norm:
            normFList = np.array([params[0][0] for params
                                  in self.resData.params])[:, np.newaxis]
            ECFunc /= normFList

        self.data = self.data._replace(intensities = self.data.intensities
                                       - scaleFactor
                                       * ECFunc[np.newaxis, :, :])


        # Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(S, S < 0, 0)
        np.place(errors, S <= 0, np.inf)
        self.data = self.data._replace(intensities=S)
        self.data = self.data._replace(errors=errors)




    def absorptionCorrection(self, canType='tube', canScaling=0.9,
                             neutron_wavelength=6.27, absco_kwargs={}):
        """ Computes absorption coefficients for sample in a flat or
            tubular can and apply corrections to data, for each q-value
            in *data.qVals* attribute.

            :arg canType:            type of can used, either 'tube' or 'slab'
            :arg canScaling:         scaling factor for empty can contribution
                                     term, set it to 0 to use
                                     only correction of sample
                                     self-attenuation
            :arg neutron_wavelength: incident neutrons wavelength
            :arg absco_kwargs:       geometry arguments for absco library
                                     from Joachim Wuttke [#]_ .

            References:

            .. [#] http://apps.jcns.fz-juelich.de/doku/sc/absco

        """

        # Defining some defaults arguments
        kwargs = {'mu_i_S': 0.660,
                  'mu_f_S': 0.660,
                  'mu_i_C': 0.147,
                  'mu_f_C': 0.147}



        if canType == 'slab':
            kwargs['slab_angle']        = 45
            kwargs['thickness_C_front'] = 0.5
            kwargs['thickness_C_rear']  = 0.5

        if canType == 'tube':
            kwargs['radius']            = 2.15
            kwargs['thickness_C_inner'] = 0.1
            kwargs['thickness_C_outer'] = 0.1



        # Modifies default arguments with given ones, if any
        for key, value in absco_kwargs.items():
            kwargs[key] = value

        sampleSignal = self.data.intensities

        # Empty cell data
        if isinstance(self.ECData, fECType.fECType):
            ECFunc = self.ECData.data.intensities

        else:  # Assumes full QENS with fitted model
            # Tries to extract empty cell intensities, use an array
            # of zeros if no data are found
            try:
                # Compute the fitted Empty Cell function
                ECFunc = []
                for qIdx, qVal in enumerate(self.data.qVals):
                    ECFunc.append(self.ECData.model(self.data.X,
                                  *self.ECData.params[qIdx][0]))

                ECFunc = np.array(ECFunc)


            except AttributeError:
                ECFunc = np.zeros_like(sampleSignal)


        # If data are normalized, uses the same normalization
        # factor for empty cell data
        normFList = np.array([params[0][0] for params
                              in self.resData.params])[:, np.newaxis]
        if self.data.norm:
            normFList = np.array([params[0][0] for params
                                  in self.resData.params])[:, np.newaxis]
            ECFunc /= normFList


        for qIdx, angle in enumerate(self.data.qVals):
            angle = np.arcsin(neutron_wavelength * angle / (4 * np.pi))

            if canType == 'slab':
                A_S_SC, A_C_SC, A_C_C = py_absco_slab(angle, **kwargs)
            if canType == 'tube':
                A_S_SC, A_C_SC, A_C_C = py_absco_tube(angle, **kwargs)


            # Applies correction
            if sampleSignal.ndim == 3:
                sampleSignal[:, qIdx] = ((1 / A_S_SC) * sampleSignal[:, qIdx]
                                         - A_C_SC / (A_S_SC * A_C_C)
                                         * canScaling * ECFunc[qIdx])
            else:
                sampleSignal[qIdx] = ((1 / A_S_SC) * sampleSignal[qIdx]
                                      - A_C_SC / (A_S_SC * A_C_C)
                                      * canScaling * ECFunc[qIdx])



        # Clean useless values from intensities and errors arrays
        errors  = self.data.errors
        np.place(sampleSignal, sampleSignal < 0, 0)
        np.place(errors, sampleSignal <= 0, np.inf)
        self.data = self.data._replace(intensities = sampleSignal)
        self.data = self.data._replace(errors = errors)


        self.data = self.data._replace(intensities = sampleSignal,
                                       errors      = errors)






    def getD2OSignal(self, qIdx=None):
        """ Computes D2O line shape for each q values.

            If a qIdx is given, returns D2O signal only for the
            corresponding q value.

        """

        D2OSignal = self.D2OData.getD2OSignal()[self.data.qIdx]


        # Check for difference in normalization state
        normF = np.array([self.resData.params[qIdx][0][0]
                          for qIdx in self.data.qIdx])
        if self.data.norm and not self.D2OData.data.norm:
            D2OSignal /= normF[:, np.newaxis]
        if not self.data.norm and self.D2OData.data.norm:
            D2OSignal *= normF[:, np.newaxis]




        xIdx = []
        for xVal in self.data.X:
            xIdx.append(np.where(
                self.D2OData.data.X - xVal == min(self.D2OData.data.X
                                                  - xVal))[0][0])

        D2OSignal = D2OSignal[:, xIdx]



        if qIdx is not None:
            D2OSignal = D2OSignal[qIdx]


        return D2OSignal





class DataTypeDecorator(FWSType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data,
                         dataType.rawData, dataType.resData,
                         dataType.D2OData, dataType.ECData)
