"""

Classes
^^^^^^^

"""

import numpy as np

from scipy.interpolate import interp1d

from nPDyn.dataTypes.baseType import BaseType, ensure_attr
from nPDyn.fileFormatParser import guessFileFormat, readFile

try:
    from ..lib.pyabsco import py_absco_slab, py_absco_tube
except ImportError:
    print('\nAbsorption correction libraries are not available. '
          'Paalman_Pings correction cannot be used.\n '
          'Verify that GSL libraries are available on this computer, '
          'and the path was correctly \n '
          'set in the setup.cfg file during package installation.\n')
    pass



class TempRampType(BaseType):
    """ This class inherits from :class:`baseType` class.

    """

    def __init__(self, fileName, data=None, rawData=None,
                 resData=None, D2OData=None, ECData=None,
                 model=None):
        super().__init__(fileName, data, rawData, resData, 
                         D2OData, ECData, model)


        self._normFList = None

    def importData(self, fileFormat=None):
        """ Extract data from file and store them in *data*
            and *rawData* attributes.

            If no fileFormat is given, tries to guess it, try
            hdf5 format if format cannot be guessed.

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

    def importRawData(self, dataList, instrument, dataType, kwargs):
        """ This method uses instrument-specific algorithm to process raw data.

            :arg dataList:      a list of data files to be imported
            :arg instrument:    the instrument used to record data
                                (only 'IN16B' possible for now)
            :arg dataType:      type of data recorded (can be 'QENS' or 'FWS')
            :arg kwargs:        keyword arguments to be passed to the algorithm
                                (see algorithm in dataParsers for details)

        """

        data = self.FWS_redAlgo[instrument](dataList, **kwargs)
        data.process()

        self.data    = data.outTuple

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

    def normalize_usingLowTemp(self, nbrBins):
        """ Normalizes data using low temperature signal.
            An average is performed over the given
            number of bins for each q value and data
            are divided by the result.

        """

        normFList = np.mean(
            self.data.intensities[:, :nbrBins], axis=1)[:, np.newaxis]

        self.data = self.data._replace(
            intensities = self.data.intensities / normFList,
            errors      = self.data.errors / normFList,
            norm        = True)

        self._normFList = normFList


    def getNormF(self):
        """ Returns normalization factors from average at
            low temperature for each q-values.
            These are computed on raw data.

        """

        return self._normFList

    @ensure_attr('ECData')
    def substractEC(self, scaleFactor=0.95):
        """ This method can be used to subtract empty cell data.
            Empty cell signal is rescaled using the given
            *subFactor* prior to substraction.

        """
        # Determine empty cell data type, and extract signal
        if isinstance(self.ECData, ECType):
            # Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                if qVal in self.ECData.data.qVals:
                    ECqIdx = int(
                        np.argwhere(qVal == self.ECData.data.qVals)[:, 0])
                    ECFunc.append(
                        self.ECData.model(0.0, *self.ECData.params[ECqIdx][0]))

                else:
                    ECFunc.append(0.0)

            ECFunc = np.array(ECFunc)[:, np.newaxis]

            # If data are normalized, uses the same normalization
            # factor for empty cell data
            if self.data.norm and not self.ECData.data.norm:
                normFList = np.array([params[0][0] for params
                                      in self.resData.params])[:, np.newaxis]
                ECFunc /= normFList


        elif isinstance(self.ECData, TempRampType):
            ECFunc = interp1d(self.ECData.data.X,
                              self.ECData.data.intensities,
                              fill_value="extrapolate")
            ECFunc = ECFunc(self.data.X)

            # If data are normalized, uses the same normalization
            # factor for empty cell data
            if self.data.norm and not self.ECData.data.norm:
                ECFunc /= self.getNormF()


        # Substracting the empty cell intensities from experimental data
        self.data = self.data._replace(
            intensities = self.data.intensities - scaleFactor * ECFunc)


        # Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities=S)
        self.data = self.data._replace(errors=errors)

    @ensure_attr('ECData')
    def absorptionCorrection(self, canType='tube', canScaling=0.9,
                             neutron_wavelength=6.27, absco_kwargs=None):
        """Computes absorption Paalman-Pings coefficients 
        
        Can be used for sample in a flat or tubular can and apply corrections 
        to data, for each q-value in *data.qVals* attribute.

        Parameters
        ----------
        canType : {'tube', 'slab'}
            type of can used, either 'tube' or 'slab'
            (default 'tube')
        canScaling : float         
            scaling factor for empty can contribution term, set it to 0 
            to use only correction of sample self-attenuation
        neutron_wavelength : float 
            incident neutrons wavelength
        absco_kwargs : dict  
            geometry arguments for absco library
            from Joachim Wuttke [#]_.

        References
        ----------
        .. [#] http://apps.jcns.fz-juelich.de/doku/sc/absco

        """
        # Defining some defaults arguments
        kwargs = {'mu_i_S': 0.660,
                  'mu_f_S': 0.660,
                  'mu_i_C': 0.147,
                  'mu_f_C': 0.147}

        if canType == 'slab':
            kwargs['slab_angle']        = 45
            kwargs['thickness_S']       = 0.03
            kwargs['thickness_C_front'] = 0.5
            kwargs['thickness_C_rear']  = 0.5

        if canType == 'tube':
            kwargs['radius']            = 2.15
            kwargs['thickness_S']       = 0.03
            kwargs['thickness_C_inner'] = 0.1
            kwargs['thickness_C_outer'] = 0.1

        # Modifies default arguments with given ones, if any
        if absco_kwargs is not None:
            kwargs.update(absco_kwargs)

        sampleSignal = self.data.intensities

        # Determine empty cell data type, and extract signal
        if isinstance(self.ECData, ECType):
            # Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append(
                    self.ECData.model(0.0, *self.ECData.params[qIdx][0]))

            ECFunc = np.array(ECFunc)[:, np.newaxis]

            # If data are normalized, uses the same normalization
            # factor for empty cell data
            if self.data.norm and not self.ECData.data.norm:
                normFList = np.array([params[0][0] for params
                                      in self.resData.params])[:, np.newaxis]
                ECFunc /= normFList

        elif isinstance(self.ECData, TempRampType):
            ECFunc = interp1d(self.ECData.data.X,
                              self.ECData.data.intensities,
                              fill_value="extrapolate")
            ECFunc = ECFunc(self.data.X)

            # If data are normalized, uses the same normalization
            # factor for empty cell data
            if self.data.norm and not self.ECData.data.norm:
                ECFunc /= self.getNormF(5)

        else:
            ECFunc = np.zeros_like(sampleSignal)

        for qIdx, angle in enumerate(self.data.qVals):
            angle = np.arcsin(neutron_wavelength * angle / (4 * np.pi))

            if canType == 'slab':
                A_S_SC, A_C_SC, A_C_C = py_absco_slab(angle, **kwargs)
            if canType == 'tube':
                A_S_SC, A_C_SC, A_C_C = py_absco_tube(angle, **kwargs)

            # Applies correction
            sampleSignal[qIdx] = ((1 / A_S_SC) * sampleSignal[qIdx]
                                  - A_C_SC / (A_S_SC * A_C_C)
                                  * canScaling * ECFunc[qIdx])

        # Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities=S)
        self.data = self.data._replace(errors=errors)
