import numpy as np

from collections import namedtuple

from .baseType import BaseType
from ..fileFormatParser import guessFileFormat, readFile, fileImporters
from ..dataTypes import ECType, fECType


class FWSType(BaseType):

    def __init__(self, fileName, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        super().__init__(fileName, data, rawData, resData, D2OData, ECData)


        self.timestep = None #_Should be given in hours


    def importData(self, fileFormat=None):
        """ Extract data from file and store them in self.data and self.rawData attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. """

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




    def binData(self, binSize):
        """ Binning in energy dimension does nothing as it make no sense for FWS data type. """

        return



    def normalize(self):
        """ Normalizes data using a list of scaling factors from resolution function fit.
            It assumes that the scaling factor is in first position in the model parameters.
            There should be as many normalization factors as q-values in data. """

        normFList = np.array( [params[0][0] for params in self.resData.params] )

        #_Applying normalization
        self.data = self.data._replace( 
                                    intensities = self.data.intensities / normFList[np.newaxis,:,np.newaxis],
                                    errors      = self.data.errors / normFList[np.newaxis,:,np.newaxis],
                                    norm        = True )





    def substractEC(self, scaleFactor=0.8):
        """ Use the assigned empty cell data for substraction to loaded data.
            
            Empty cell data are scaled using the given scaleFactor prior to substraction. """

        if isinstance(self.ECData, fECType.fECType):
            ECFunc = self.ECData.data.intensities


        else: #_Assumes full QENS with fitted model
            #_Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append( self.ECData.model( self.data.X, *self.ECData.params[qIdx][0] ) )

            ECFunc = np.array( ECFunc )


        #_If data are normalized, uses the same normalization factor for empty cell data
        if self.data.norm:
            normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
            ECFunc /= normFList

        self.data = self.data._replace( intensities = self.data.intensities 
                                                                    - scaleFactor * ECFunc[np.newaxis,:,:] )


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(S, S < 0, 0)
        np.place(errors, S <= 0, np.inf)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)




    def absorptionCorrection(self, canType='tube', canScaling=0.9, neutron_wavelength=6.27, 
                                                                                    absco_kwargs={}):
        """ Computes absorption coefficients for sample in a flat can and apply corrections to data,
            for each q-value in self.data.qVals. 
            
            Input:  canType             -> type of can used, either 'tube' or 'slab'
                    canScaling          -> scaling factor for empty can contribution term, set it to 0 to use
                                            only correction of sample self-attenuation
                    neutron_wavelength  -> incident neutrons wavelength
                    absco_kwargs        -> geometry arguments for absco library from Joachim Wuttke
                                            see http://apps.jcns.fz-juelich.de/doku/sc/absco """

        #_Defining some defaults arguments
        kwargs = {  'mu_i_S'            : 0.660, 
                    'mu_f_S'            : 0.660, 
                    'mu_i_C'            : 0.147,
                    'mu_f_C'            : 0.147 }



        if canType=='slab':
            kwargs['slab_angle']        = 45
            kwargs['thickness_C_front'] = 0.5 
            kwargs['thickness_C_rear']  = 0.5 

        if canType=='tube':
            kwargs['radius']            = 2.15
            kwargs['thickness_C_inner'] = 0.1 
            kwargs['thickness_C_outer'] = 0.1 



        #_Modifies default arguments with given ones, if any
        for key, value in absco_kwargs.items():
            kwargs[key] = value

        sampleSignal = self.data.intensities

        #_Empty cell data
        if isinstance(self.ECData, fECType.fECType):
            ECFunc = self.ECData.data.intensities

        else: #_Assumes full QENS with fitted model
            try: #_Tries to extract empty cell intensities, use an array of zeros if no data are found
                #_Compute the fitted Empty Cell function
                ECFunc = []
                for qIdx, qVal in enumerate(self.data.qVals):
                    ECFunc.append( self.ECData.model( self.data.X, *self.ECData.params[qIdx][0] ) )

                ECFunc = np.array( ECFunc )

            
            except AttributeError:
                ECFunc = np.zeros_like(sampleSignal)


        #_If data are normalized, uses the same normalization factor for empty cell data
        normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
        if self.data.norm:
            normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
            ECFunc /= normFList


        for qIdx, angle in enumerate(self.data.qVals):
            angle = np.arcsin(neutron_wavelength * angle / (4 * np.pi))

            if canType == 'slab':
                A_S_SC, A_C_SC, A_C_C = py_absco_slab(angle, **kwargs)
            if canType == 'tube':
                A_S_SC, A_C_SC, A_C_C = py_absco_tube(angle, **kwargs)


            #_Applies correction
            sampleSignal[:,qIdx] = ( (1 / A_S_SC) * sampleSignal[:,qIdx] 
                            - A_C_SC / (A_S_SC*A_C_C) * canScaling * ECFunc[qIdx] )



        #_Clean useless values from intensities and errors arrays
        errors  = self.data.errors
        np.place(sampleSignal, sampleSignal < 0, 0)
        np.place(errors, sampleSignal <= 0, np.inf)
        self.data = self.data._replace(intensities = sampleSignal)
        self.data = self.data._replace(errors = errors)


        self.data = self.data._replace( intensities = sampleSignal,
                                        errors      = errors )






    def getD2OSignal(self, qIdx=None):
        """ Computes D2O line shape for each q values.
            
            If a qIdx is given, returns D2O signal only for the corresponding q value. """

        D2OSignal = self.D2OData.getD2OSignal()[self.data.qIdx]


        #_Check for difference in normalization state
        normF = np.array( [ self.resData.params[qIdx][0][0] for qIdx in self.data.qIdx ] )
        if self.data.norm and not self.D2OData.data.norm:
            D2OSignal /= normF[:,np.newaxis]
        if not self.data.norm and self.D2OData.data.norm:
            D2OSignal *= normF[:,np.newaxis]


    
    
        xIdx = []
        for xVal in self.data.X:
            xIdx.append( np.where( self.D2OData.data.X - xVal == min(self.D2OData.data.X - xVal) )[0][0] )
    
        D2OSignal = D2OSignal[:,xIdx]



        if qIdx is not None:
            D2OSignal = D2OSignal[qIdx]


        return D2OSignal





class DataTypeDecorator(FWSType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)



