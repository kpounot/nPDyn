import numpy as np

from collections import namedtuple

from itertools import count

from ..dataManipulation.binData import binData
from ..fileFormatParser import guessFileFormat, readFile, fileImporters
from ..lib.pyabsco import py_absco_slab, py_absco_tube


class BaseType:

    def __init__(self, fileName=None, data=None, rawData=None, resData=None, D2OData=None, ECData=None):
        """ Initialize a base type that will be inherited by all other specialized types through its decorator. 
        
            Input:  fileName    -> name of the file being read
                    data        -> resulting namedtuple from data parsers 
                    rawData     -> used by the decorator """

        self.fileName   = fileName
        self.data       = data 
        self.rawData    = rawData #_Used to reset data to its initial state

        self.resData    = resData #_For use with sample data types
        self.D2OData    = D2OData #_For use with sample data types
        self.ECData     = ECData 



    def importData(self, fileFormat=None):
        """ Extract data from file and store them in self.data and self.rawData attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format if format cannot be guessed. """

        if fileFormat:
            data = readFile(fileFormat, self.fileName)
        else:
            data = guessFileFormat(self.fileName)

        self.data       = data
        self.rawData    = self.data._replace(   qVals       = np.copy(self.data.qVals),
                                                X           = np.copy(self.data.X),
                                                intensities = np.copy(self.data.intensities),
                                                errors      = np.copy(self.data.errors),
                                                temp        = np.copy(self.data.temp),
                                                norm        = False,
                                                qIdx        = np.copy(self.data.qIdx) )



    def binData(self, binSize):
        """ Bin self.data attribute using the given binSize. """

        self.data = binData(self.data, binSize)



    def normalize(self):
        """ Normalizes data using a list of scaling factors from resolution function fit.
            It assumes that the scaling factor is in first position in the model parameters.
            There should be as many normalization factors as q-values in data. """

        normFList = np.array( [params[0][0] for params in self.resData.params] )

        #_Applying normalization
        self.data = self.data._replace( intensities = self.data.intensities / normFList[:,np.newaxis],
                                        errors      = self.data.errors / normFList[:,np.newaxis],
                                        norm        = True )


        #_Normalizes also D2O data if needed
        try:
            if not self.D2OData.data.norm:
                self.D2OData.normalize()
        except AttributeError:
            pass




    def substractEC(self, scaleFactor=0.95):
        """ Use the assigned empty cell data for substraction to loaded data.
            
            Empty cell data are scaled using the given scaleFactor prior to substraction. """

        #_Compute the fitted Empty Cell function
        ECFunc = []
        for qIdx, qVal in enumerate(self.data.qVals):
            ECFunc.append( self.ECData.model( self.data.X, *self.ECData.params[qIdx][0] ) )

        ECFunc = np.array( ECFunc )

        #_If data are normalized, uses the same normalization factor for empty cell data
        normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
        if self.data.norm:
            normFList = np.array([params[0][0] for params in self.resData.params])[:,np.newaxis]
            ECFunc /= normFList

        self.data = self.data._replace( intensities = self.data.intensities - scaleFactor * ECFunc )


        #_Clean useless values from intensities and errors arrays
        S       = self.data.intensities
        errors  = self.data.errors
        np.place(S, S < 0, 0)
        np.place(errors, S <= 0, np.inf)
        self.data = self.data._replace(intensities = S)
        self.data = self.data._replace(errors = errors)




    def resetData(self):
        """ Reset self.data to its initial state by copying rawData attribute. """

        self.data = self.data._replace( qVals       = np.copy(self.rawData.qVals),
                                        X           = np.copy(self.rawData.X),
                                        intensities = np.copy(self.rawData.intensities),
                                        errors      = np.copy(self.rawData.errors),
                                        temp        = np.copy(self.rawData.temp),
                                        norm        = False,
                                        qIdx        = np.copy(self.rawData.qIdx) )



    def discardDetectors(self, *qIdx):
        """ Remove detectors indices.
            The process modifies self.data.qIdx attribute that is used for fitting and plotting. """

        self.data = self.data._replace(qIdx = np.array( [val for val in self.data.qIdx if val not in qIdx] ))



    def resetDetectors(self):
        """ Reset qIdx entry to its original state, with all q values taken into account. """

        self.data = self.data._replace(qIdx = np.array([idx for idx, val in enumerate(self.data.qVals)]))





    def getResFunc(self, withBkgd=False):
        """ Quick way to obtain the fitted resolution function for this dataset. """

        if withBkgd:
            params = [ self.resData.params[i][0] for i in range(self.data.qVals.size) ]
        else:
            params = [ np.append(self.resData.params[i][0][:-1], 0) for i in range(self.data.qVals.size) ]


        if self.data.norm: #_Use normalized resolution function if data were normalized
            f_res = np.array( [self.resData.model(self.data.X, 1, *params[i][1:]) for i in self.data.qIdx] )
        else:
            f_res = np.array( [self.resData.model(self.data.X, *params[i]) for i in self.data.qIdx] )


        return f_res




    def getResBkgd(self):
        """ Returns the background fitted from resolution function """


        bkgd = np.array( [self.resData.params[i][0][-1] for i in self.data.qIdx] )

        #_Use normalized resolution function if data were normalized
        if not self.data.norm: 
            bkgd = np.array( [self.resData.params[i][0][-1] * 
                                        self.resData.params[i][0][0] for i in self.data.qIdx] )


        return bkgd






    def assignResData(self, resData):
        """ Sets self.resData attribute to the given one, a ResType instance that can be used by fitting 
            functions in QENS or FWS types. """

        self.resData = resData




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

        if qIdx is not None:
            D2OSignal = D2OSignal[qIdx]


        return D2OSignal




    def assignD2OData(self, D2OData):
        """ Sets self.D2OData attribute to the given one, a D2OType instance that can be used by fitting
            functions in QENS or FWS types. """

        self.D2OData = D2OData
 

    def assignECData(self, ECData):
        """ Sets self.ECData attribute to the given one, a ECType instance that can be used by fitting
            functions in QENS or FWS types. """

        self.ECData = ECData





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
        try: #_Tries to extract empty cell intensities, use an array of zeros if no data are found
            #_Compute the fitted Empty Cell function
            ECFunc = []
            for qIdx, qVal in enumerate(self.data.qVals):
                ECFunc.append( self.ECData.model( self.data.X, *self.ECData.params[qIdx][0] ) )

            ECFunc = np.array( ECFunc )

        
        except AttributeError:
            ECFunc = np.zeros_like(sampleSignal)

        #_If data are normalized, uses the same normalization factor for empty cell data
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
            sampleSignal[qIdx] = ( (1 / A_S_SC) * sampleSignal[qIdx] 
                                            - A_C_SC / (A_S_SC*A_C_C) * canScaling * ECFunc[qIdx] )



        #_Clean useless values from intensities and errors arrays
        errors  = self.data.errors
        np.place(sampleSignal, sampleSignal < 0, 0)
        np.place(errors, sampleSignal <= 0, np.inf)
        self.data = self.data._replace(intensities = sampleSignal)
        self.data = self.data._replace(errors = errors)


        self.data = self.data._replace( intensities = sampleSignal,
                                        errors      = errors )



    def discardOutliers(self, meanScale):
        """ Discards outliers in experimental based on signal / noise ratio.
            
            Input: meanScale -> factor by which mean of signal over noise ratio will be 
                                multiplied. Then, this scaled mean is used as a threshold under which
                                data errors will be set to infinite so that they won't weigh in the 
                                fitting procedure. """


        sigNoiseR = self.data.intensities / self.data.errors
        threshold = meanScale * np.mean( sigNoiseR )

        errors = self.data.errors
        np.place(errors, sigNoiseR < threshold, np.inf)

        self.data = self.data._replace( errors = errors )





class DataTypeDecorator(BaseType):

    def __init__(self, dataType):
        super().__init__(dataType.fileName, dataType.data, dataType.rawData, dataType.resData, 
                                                                        dataType.D2OData, dataType.ECData)

