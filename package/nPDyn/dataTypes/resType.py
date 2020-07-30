"""

Classes
^^^^^^^

"""

import numpy as np

from nPDyn.dataTypes.baseType import BaseType

try:
    from ..lib.pyabsco import py_absco_slab, py_absco_tube
except ImportError:
    print('\nAbsorption correction libraries are not available. \
            Paalman_Pings correction cannot be used.\n \
            Verify that GSL libraries are available on this computer, \
            and the path was correctly \n \
            set in the setup.cfg file during package installation.\n')
    pass



class ResType(BaseType):
    """ This class inherits from :class:`baseType` class.

    """

    def __init__(self, fileName=None, data=None, rawData=None,
                 resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)



    def substractEC(self, scaleFactor=0.95):
        """ Use the assigned empty cell data for substraction to loaded data.

            Empty cell data are scaled using the given
            scaleFactor prior to substraction.

        """

        # Compute the fitted Empty Cell function
        ECFunc = []
        for qIdx, qVal in enumerate(self.data.qVals):
            ECFunc.append(self.ECData.model(
                self.data.X, 0, *self.ECData.params[qIdx][0][1:]))

        ECFunc = np.array(ECFunc)

        # If data are normalized, uses the same normalization
        # factor for empty cell data
        normFList = np.array([params[0][0] for params
                              in self.params])[:, np.newaxis]
        if self.data.norm:
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



    def absorptionCorrection(self, canType='tube', canScaling=0.9,
                             neutron_wavelength=6.27, absco_kwargs={}):
        """ Computes absorption coefficients for sample in a flat or
            tubular can and apply corrections to data, for each q-value
            in *data.qVals* attribute.

            :arg canType:            type of can used, either 'tube' or 'slab'
            :arg canScaling:         scaling factor for empty can
                                     contribution term, set it to 0 to use
                                     only correction of sample self-attenuation
            :arg neutron_wavelength: incident neutrons wavelength
            :arg absco_kwargs:       geometry arguments for absco library
                                     from Joachim Wuttke. See:
                                http://apps.jcns.fz-juelich.de/doku/sc/absco

        """

        # Defining some defaults arguments
        kwargs = {'mu_i_S': 0.65,
                  'mu_f_S': 0.65,
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
        # Tries to extract empty cell intensities, use an array
        # of zeros if no data are found
        try:
            # Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append(self.ECData.model(
                    self.data.X, *self.ECData.params[qIdx][0]))

            ECFunc = np.array(ECFunc)


        except AttributeError:
            ECFunc = np.zeros_like(sampleSignal)

        # If data are normalized, uses the same normalization
        # factor for empty cell data
        normFList = np.array([params[0][0] for params
                              in self.params])[:, np.newaxis]
        if self.data.norm:
            normFList = np.array([params[0][0] for params
                                  in self.params])[:, np.newaxis]
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


        self.data = self.data._replace(intensities = sampleSignal,
                                       errors      = errors)





class DataTypeDecorator(ResType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data,
                         dataType.rawData, dataType.resData,
                         dataType.D2OData, dataType.ECData)
