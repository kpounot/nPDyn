"""

Classes
^^^^^^^

"""

import numpy as np

from nPDyn.dataManipulation.binData import binData
from nPDyn.fileFormatParser import guessFileFormat, readFile
from nPDyn.dataParsers import *

try:
    from ..lib.pyabsco import py_absco_slab, py_absco_tube
except ImportError:
    print('\nAbsorption correction libraries are not available. \
            Paalman_Pings correction cannot be used.\n \
            Verify that GSL libraries are available on this computer, \
            and the path was correctly \n \
            set in the setup.cfg file during package installation.\n')
    pass


class BaseType:
    """ Initialize a base type that will be inherited by all other
        specialized types through its decorator.

        :arg fileName: name of the file being read
        :arg data:     resulting namedtuple from data parsers
        :arg rawData:  used by the decorator
        :arg resData:  data for resolution function
        :type resData: :class:`resType`
        :arg D2OData:  D2O data if needed
        :type D2OData: :class:`D2OType`
        :arg ECData:   empty cell measurement data
        :type ECData:  :class:`ECType`

    """


    def __init__(self, fileName=None, data=None, rawData=None, resData=None,
                 D2OData=None, ECData=None):
        self.fileName   = fileName
        self.data       = data
        self.rawData    = rawData  # Used to reset data to its initial state

        self.resData    = resData  # For use with sample data types
        self.D2OData    = D2OData  # For use with sample data types
        self.ECData     = ECData

        self.QENS_redAlgo = {'IN16B': IN16B_QENS_scans_reduction.IN16B_QENS}
        self.FWS_redAlgo  = {'IN16B': IN16B_FWS_scans_reduction.IN16B_FWS}



    def importData(self, fileFormat=None):
        """ Extract data from file and store them in *data* and *rawData*
            attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format
            if format cannot be guessed.

        """

        if fileFormat:
            data = readFile(fileFormat, self.fileName)
        else:
            data = guessFileFormat(self.fileName)


        self.data    = data
        self.rawData = self.data._replace(
            qVals       = np.copy(self.data.qVals),
            X           = np.copy(self.data.X),
            intensities = np.copy(self.data.intensities),
            errors      = np.copy(self.data.errors),
            temp        = np.copy(self.data.temp),
            norm        = False,
            qIdx        = np.copy(self.data.qIdx))




    def importRawData(self, dataList, instrument, dataType, kwargs):
        """ This method uses instrument-specific algorithm to process raw data.

            :arg dataList:      a list of data files to be imported
            :arg instrument:    the instrument used to record data
                                (only 'IN16B' possible for now)
            :arg dataType:      type of data recorded (can be 'QENS' or 'FWS')
            :arg kwargs:        keyword arguments to be passed to the algorithm
                                (see algorithm in dataParsers for details)

        """

        if dataType in ['QENS', 'res', 'ec', 'D2O']:
            data = self.QENS_redAlgo[instrument](dataList, **kwargs)
            data.process()

            self.data = data.outTuple

            self.rawData = []
            for data in self.data:
                self.rawData.append(data._replace(
                    qVals       = np.copy(data.qVals),
                    X           = np.copy(data.X),
                    intensities = np.copy(data.intensities),
                    errors      = np.copy(data.errors),
                    temp        = np.copy(data.temp),
                    norm        = False,
                    qIdx        = np.copy(data.qIdx)))


        elif dataType in ['FWS', 'fec', 'fD2O']:
            data = self.FWS_redAlgo[instrument](dataList, **kwargs)
            data.process()

            self.data    = data.outTuple
            self.rawData = self.data._replace(
                qVals       = np.copy(self.data.qVals),
                X           = np.copy(self.data.X),
                Y           = np.copy(self.data.Y),
                intensities = np.copy(self.data.intensities),
                errors      = np.copy(self.data.errors),
                temp        = np.copy(self.data.temp),
                norm        = False,
                qIdx        = np.copy(self.data.qIdx))



    def binData(self, binSize):
        """ Bin *data* attribute using the given *binSize*. """

        self.data = binData(self.data, binSize)



    def normalize(self):
        """ Normalizes data using a list of scaling factors from
            resolution function fit.

            It assumes that the scaling factor is in first position
            in the model parameters.
            There should be as many normalization factors as q-values in data.

        """

        normFList = np.array([params[0][0] for params in self.resData.params])

        # Applying normalization
        self.data = self.data._replace(
            intensities = self.data.intensities / normFList[:, np.newaxis],
            errors      = self.data.errors / normFList[:, np.newaxis],
            norm        = True)


        # Normalizes also D2O data if needed
        try:
            if not self.D2OData.data.norm:
                self.D2OData.normalize()
        except AttributeError:
            pass




    def subtractEC(self, scaleFactor=0.95, useModel=True):
        """ Use the assigned empty cell data for substraction to loaded data.

            :arg scaleFactor:   Empty cell data are scaled using the given
                                factor prior to subtraction.
            :arg useModel:      For QENS data, use the fitted model instead
                                of experimental points to perform the
                                subtraction if True.

        """

        if useModel:  # Uses a fitted model
            # Gets the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append(self.ECData.model(
                    self.data.X, *self.ECData.params[qIdx][0]))

            ECFunc = np.array(ECFunc)

        else:
            ECFunc = self.ECData.data.intensities


        # If data are normalized, uses the same normalization
        # factor for empty cell data
        if self.data.norm:
            normFList = np.array([params[0][0] for params
                                 in self.resData.params])[:, np.newaxis]
            ECFunc /= normFList

        self.data = self.data._replace(
            intensities = self.data.intensities - scaleFactor * ECFunc)


        # Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(S, S < 0, 0)
        np.place(errors, S <= 0, np.inf)
        self.data = self.data._replace(intensities=S)
        self.data = self.data._replace(errors=errors)




    def resetData(self):
        """ Reset *data* attrbute to its initial state by
            copying *rawData* attribute.

        """

        self.data = self.data._replace(
            qVals       = np.copy(self.rawData.qVals),
            X           = np.copy(self.rawData.X),
            intensities = np.copy(self.rawData.intensities),
            errors      = np.copy(self.rawData.errors),
            temp        = np.copy(self.rawData.temp),
            norm        = False,
            qIdx        = np.copy(self.rawData.qIdx))



    def discardDetectors(self, *qIdx):
        """ Remove detectors indices.

            The process modifies *data.qIdx* attribute that is
            used for fitting and plotting.

        """

        self.data = self.data._replace(qIdx=np.array(
            [val for val in self.data.qIdx if val not in qIdx]))




    def resetDetectors(self):
        """ Reset *data.qIdx* entry to its original state, with all
            q values taken into account. """

        self.data = self.data._replace(qIdx=np.array([idx for idx, val
                                       in enumerate(self.data.qVals)]))




    def getResFunc(self, withBkgd=False):
        """ Quick way to obtain the fitted resolution
            function for this dataset. """

        if withBkgd:
            params = [self.resData.params[i][0] for i
                      in range(self.data.qVals.size)]
        else:
            params = [np.append(self.resData.params[i][0][:-1], 0)
                      for i in range(self.data.qVals.size)]

        # Use normalized resolution function if data were normalized
        if self.data.norm:
            f_res = np.array(
                [self.resData.model(self.data.X, 1, *params[i][1:])
                 for i in self.data.qIdx])
        else:
            f_res = np.array([self.resData.model(self.data.X, *params[i])
                              for i in self.data.qIdx])


        return f_res




    def getResBkgd(self):
        """ Returns the background fitted from resolution function """


        bkgd = np.array([self.resData.params[i][0][-1]
                         for i in self.data.qIdx])

        # Use normalized resolution function if data were normalized
        if not self.data.norm:
            bkgd = np.array(
                [self.resData.params[i][0][-1] * self.resData.params[i][0][0]
                 for i in self.data.qIdx])


        return bkgd






    def assignResData(self, resData):
        """ Sets *resData* attribute to the given one, a ResType instance
            that can be used by fitting functions in QENS or FWS types.

        """

        self.resData = resData




    def getD2OSignal(self, qIdx=None):
        """ Computes D2O line shape for each q values.

            If a *data.qIdx* is given, returns D2O signal only for the
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

        if qIdx is not None:
            D2OSignal = D2OSignal[qIdx]


        return D2OSignal




    def assignD2OData(self, D2OData):
        """ Sets *D2OData* attribute to the given one, a :class:`D2OType`
            instance that can be used by fitting functions in QENS or
            FWS types.

        """

        self.D2OData = D2OData


    def assignECData(self, ECData):
        """ Sets *ECData* attribute to the given one, a :class:`ECType`
            instance that can be used by fitting functions in
            QENS or FWS types.

        """

        self.ECData = ECData



    def absorptionCorrection(self, canType='tube', canScaling=0.9,
                             neutron_wavelength=6.27, absco_kwargs={}):
        """ Computes absorption coefficients for sample in a flat or tubular
            can and apply corrections to data, for each q-value
            in *data.qVals* attribute.

            :arg canType:           type of can used, either 'tube' or 'slab'
            :arg canScaling:        scaling factor for empty can contribution
                                    term, set it to 0 to use only correction
                                    of sample self-attenuation
            :arg neutron_wavelength: incident neutrons wavelength
            :arg absco_kwargs:       geometry arguments for absco library
                                     from Joachim Wuttke. See:
                                http://apps.jcns.fz-juelich.de/doku/sc/absco

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
        for key, value in absco_kwargs.items():
            kwargs[key] = value

        sampleSignal = self.data.intensities
        # Tries to extract empty cell intensities,
        # use an array of zeros if no data are found
        # Compute the fitted Empty Cell function
        try:
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append(self.ECData.model(self.data.X,
                              * self.ECData.params[qIdx][0]))

            ECFunc = np.array(ECFunc)


        except AttributeError:
            ECFunc = np.zeros_like(sampleSignal)

        # If data are normalized, uses the same normalization
        # factor for empty cell data
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
            sampleSignal[qIdx] = ((1 / A_S_SC) * sampleSignal[qIdx]
                                  - A_C_SC / (A_S_SC * A_C_C)
                                  * canScaling * ECFunc[qIdx])



        # Clean useless values from intensities and errors arrays
        errors  = self.data.errors
        np.place(sampleSignal, sampleSignal < 0, 0)
        np.place(errors, sampleSignal <= 0, np.inf)
        self.data = self.data._replace(intensities=sampleSignal)
        self.data = self.data._replace(errors=errors)


        self.data = self.data._replace(intensities=sampleSignal,
                                       errors=errors)



    def discardOutliers(self, meanScale):
        """ Discards outliers in experimental based on signal / noise ratio.

            :arg meanScale: factor by which mean of signal over noise ratio
                            will be multiplied. Then, this scaled mean is used
                            as a threshold under which data errors will be
                            set to infinite so that they won't weigh in the
                            fitting procedure.

        """


        sigNoiseR = self.data.intensities / self.data.errors
        threshold = meanScale * np.mean(sigNoiseR)

        errors = self.data.errors
        np.place(errors, sigNoiseR < threshold, np.inf)

        self.data = self.data._replace(errors=errors)


    def setQRange(self, minQ, maxQ):
        """ Discard detectors that do not lie inside required q-range

            :arg minQ: minimum wanted q-value
            :arg maxQ: maximum wanted q-value

        """

        # In case some detectors were already discarded, put qIdx
        # list back to its original state
        self.resetDetectors()

        ids = np.argwhere(np.logical_and(self.data.qVals > minQ,
                                         self.data.qVals < maxQ)).flatten()

        self.data = self.data._replace(qIdx=self.data.qIdx[ids])







class DataTypeDecorator(BaseType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data,
                         dataType.rawData, dataType.resData,
                         dataType.D2OData, dataType.ECData)
