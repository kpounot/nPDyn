""" 

Classes
^^^^^^^

"""

import numpy as np

from collections import namedtuple

from scipy.interpolate import interp1d

from nPDyn.dataTypes.baseType import BaseType
from nPDyn.dataTypes.ECType import ECType
from nPDyn.fileFormatParser import guessFileFormat, readFile, fileImporters


class TempRampType(BaseType):
    """ This class inherits from :class:`baseType` class.

    """

    def __init__(self, fileName, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


    def importData(self, fileFormat=None):
        """ Extract data from file and store them in *data* and *rawData* attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. 

        """

        if fileFormat:
            data = readFile(fileFormat, self.fileName, True)
        else:
            data = guessFileFormat(self.fileName, True)

        self.data       = data
        self.rawData    = self.data._replace(   qVals       = np.copy(self.data.qVals),
                                                X           = np.copy(self.data.X),
                                                intensities = np.copy(self.data.intensities),
                                                errors      = np.copy(self.data.errors),
                                                temp        = np.copy(self.data.temp),
                                                norm        = False,
                                                qIdx        = np.copy(self.data.qIdx) )



    def normalize_usingLowTemp(self, nbrBins):
        """ Normalizes data using low temperature signal. An average is performed over the given
            number of bins for each q value and data are divided by the result. 

        """

        normFList = np.mean(self.data.intensities[:,:nbrBins], axis=1)[:,np.newaxis]

        self.data = self.data._replace( intensities = self.data.intensities / normFList,
                                        errors      = self.data.errors / normFList,
                                        norm=True )

    def getNormF(self, nbrBins):
        """ Returns normalization factors from average at low temperature for each q-values.
            These are computed on raw data. 

        """

        normFList = np.mean(self.rawData.intensities[:,:nbrBins], axis=1)[:,np.newaxis]




    def substractEC(self, scaleFactor=0.95):
        """ This method can be used to subtract empty cell data.
            Empty cell signal is rescaled using the given *subFactor* prior to substraction. 

        """

        #_Determine empty cell data type, and extract signal
        if isinstance(self.ECData, ECType):
            #_Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append( self.ECData.model( 0.0, *self.ECData.params[qIdx][0] ) )

            ECFunc = np.array( ECFunc )[:,np.newaxis]

            #_If data are normalized, uses the same normalization factor for empty cell data
            if self.data.norm and not self.ECData.data.norm:
                normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
                ECFunc /= normFList


        elif isinstance(self.ECData, TempRampType):
            ECFunc = interp1d(self.ECData.data.X, self.ECData.data.intensities, fill_value="extrapolate")
            ECFunc = ECFunc(self.data.X)

            #_If data are normalized, uses the same normalization factor for empty cell data
            if self.data.norm and not self.ECData.data.norm:
                ECFunc /= self.getNormF(5)



        #_Substracting the empty cell intensities from experimental data
        self.data = self.data._replace(intensities = self.data.intensities - scaleFactor * ECFunc )


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)






    def absorptionCorrection(self, canType='tube', canScaling=0.9, neutron_wavelength=6.27, 
                                                                                    absco_kwargs={}):
        """ Computes absorption coefficients for sample in a flat or tubular can and apply corrections 
            to data, for each q-value in *data.qVals* attribute. 
            
            :arg canType:            type of can used, either 'tube' or 'slab'
            :arg canScaling:         scaling factor for empty can contribution term, set it to 0 to use
                                        only correction of sample self-attenuation
            :arg neutron_wavelength: incident neutrons wavelength
            :arg absco_kwargs:       geometry arguments for absco library from Joachim Wuttke
                                        see http://apps.jcns.fz-juelich.de/doku/sc/absco """

        #_Defining some defaults arguments
        kwargs = {  'mu_i_S'            : 0.660, 
                    'mu_f_S'            : 0.660, 
                    'mu_i_C'            : 0.147,
                    'mu_f_C'            : 0.147 }


        if canType=='slab':
            kwargs['slab_angle']        = 45
            kwargs['thickness_S']       = 0.03 
            kwargs['thickness_C_front'] = 0.5 
            kwargs['thickness_C_rear']  = 0.5 

        if canType=='tube':
            kwargs['radius']            = 2.15
            kwargs['thickness_S']       = 0.03 
            kwargs['thickness_C_inner'] = 0.1 
            kwargs['thickness_C_outer'] = 0.1 



        #_Modifies default arguments with given ones, if any
        for key, value in absco_kwargs.items():
            kwargs[key] = value

        sampleSignal = self.data.intensities



        #_Determine empty cell data type, and extract signal
        if isinstance(self.ECData, ECType):
            #_Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append( self.ECData.model( 0.0, *self.ECData.params[qIdx][0] ) )

            ECFunc = np.array( ECFunc )[:,np.newaxis]

            #_If data are normalized, uses the same normalization factor for empty cell data
            if self.data.norm and not self.ECData.data.norm:
                normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
                ECFunc /= normFList


        elif isinstance(self.ECData, TempRampType):
            ECFunc = interp1d(self.ECData.data.X, self.ECData.data.intensities, fill_value="extrapolate")
            ECFunc = ECFunc(self.data.X)

            #_If data are normalized, uses the same normalization factor for empty cell data
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



            #_Applies correction
            sampleSignal[qIdx] = ( (1 / A_S_SC) * sampleSignal[qIdx] 
                                            - A_C_SC / (A_S_SC*A_C_C) * canScaling * ECFunc[qIdx] )



        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(errors, S < 0, np.inf)
        np.place(S, S < 0, 0)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)





class DataTypeDecorator(TempRampType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)




 
