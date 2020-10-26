"""Base type for all data imported in nPDyn.

This module contains a :class:`BaseType` class definition which is used
for all data imported in nPDyn.

Its role is to handle files, importation of data, and common data processing
routines. Moreover, each dataset created with this class can have associated
data for corrections and fitting (see :class:`BaseType` documentation)

"""

from functools import wraps

import numpy as np

from scipy.interpolate import interp1d

from lmfit import Model

from nPDyn.dataManipulation.binData import binData
from nPDyn.fileFormatParser import guessFileFormat, readFile
from nPDyn.dataParsers import *
from nPDyn.models.convolutions import getGlobals
from nPDyn.models.convolvedModel import ConvolvedModel

try:
    from nPDyn.lib.pyabsco import py_absco_slab, py_absco_tube
except ImportError:
    print('\nAbsorption correction libraries are not available. '
          'Paalman_Pings correction cannot be used.\n'
          'Verify that GSL libraries are available on this computer, '
          'and the path was correctly \n'
          'set in the setup.cfg file during package installation.\n')
    pass


# -------------------------------------------------------
# Useful decorators for BaseType class
# -------------------------------------------------------
def ensure_attr(attr):
    def dec(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if getattr(args[0], attr) is None:
                raise AttributeError(
                    "Attribute '%s' is None, please set it "
                    "before using this method." % attr)
            else:
                return func(*args, **kwargs)
        return wrapper
    return dec

def ensure_fit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args[0]._fit is []:
            raise ValueError(
                "Dataset (%s) has no fitted model associated with "
                "it, please fit a model before using it." % args[0].__repr__())
        else:
            return func(*args, **kwargs)
    return wrapper

# -------------------------------------------------------
# BaseType class
# -------------------------------------------------------
class BaseType:
    """Initialize a base type that can handle files, their parsing
    and importation as well as common data processing routines.

    Note
    ----
    This class is usually not used directly, but rather decorated by
    more specialized class depending on the type of data that is 
    imported (see :class:`QENSType`, :class:`FWSType`, 
    :class:`TempRampType`)

    Parameters
    ----------
    fileName : str or list(str), optional
        name of the file(s) to be read, can also be a directory for 
        raw data (in this case, all files in the directory are imported)
    data : data namedtuple, optional    
        resulting namedtuple from data parsers
    rawData : data namedtuple, optional
        named tuple containing the imported data without any further 
        processing. Used by the decorator for specialized classes 
    resData : :class:`resType`, optional 
        data for resolution function
    D2OData : :class:`D2OType` or :class:`fD2OType`, optional
        D2O (or buffer) data if needed
    ECData : :class:`ECType` or :class:`fECType`, optional
        empty cell measurement data

    """
    def __init__(self, fileName=None, data=None, rawData=None, resData=None,
                 D2OData=None, ECData=None, model=None):
        self.fileName = fileName
        self.data     = data
        self._rawData = rawData  # Used to reset data to its initial state

        self.resData  = resData  
        self.D2OData  = D2OData 
        self.ECData   = ECData

        self._QENS_redAlgo = {'IN16B': in16b_qens_scans_reduction.IN16B_QENS}
        self._FWS_redAlgo  = {'IN16B': in16b_fws_scans_reduction.IN16B_FWS}

        self.model = model
        self._fit = []

    def importData(self, fileFormat=None):
        """ Extract data from file and store them in *data* and *rawData*
            attributes.

            If no fileFormat is given, tries to guess it, try hdf5 format
            if format cannot be guessed.

            Parameters
            ----------
            fileFormat : str, optional
                file format to be used, can be 'inx' or 'mantid'

        """
        if fileFormat:
            data = readFile(fileFormat, self.fileName)
        else:
            data = guessFileFormat(self.fileName)

        self.data = data
        self._rawData = self.data._replace(
            qVals       = np.copy(self.data.qVals),
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
        if dataType in ['QENS', 'res', 'ec', 'D2O']:
            data = self._QENS_redAlgo[instrument](dataList, **kwargs)
            data.process()
            self.data = data.outTuple

        elif dataType in ['FWS', 'fec', 'fD2O']:
            data = self._FWS_redAlgo[instrument](dataList, **kwargs)
            data.process()
            self.data = data.outTuple

        self._rawData = self.data._replace(
            qVals       = np.copy(self.data.qVals),
            times       = np.copy(self.data.times),
            intensities = np.copy(self.data.intensities),
            errors      = np.copy(self.data.errors),
            temps       = np.copy(self.data.temps),
            norm        = False,
            qIdx        = np.copy(self.data.qIdx),
            energies    = np.copy(self.data.energies),
            observable  = np.copy(self.data.observable),
            observable_name = np.copy(self.data.observable_name))

    def resetData(self):
        """Reset *data* attrbute to its initial state by
        copying *rawData* attribute.

        """
        self.data = self.data._replace(
            qVals       = np.copy(self._rawData.qVals),
            selQ        = np.copy(self._rawData.selQ),
            times       = np.copy(self._rawData.times),
            intensities = np.copy(self._rawData.intensities),
            errors      = np.copy(self._rawData.errors),
            temps       = np.copy(self._rawData.temps),
            norm        = False,
            qIdx        = np.copy(self._rawData.qIdx),
            energies    = np.copy(self._rawData.energies),
            observable  = np.copy(self._rawData.observable),
            observable_name = np.copy(self._rawData.observable_name))

    @property
    def resData(self):
        """Return the data for resolution function."""
        return self._resData

    @resData.setter
    def resData(self, data):
        """Setter for the resData attribute."""
        if data is not None:
            if not isinstance(data, BaseType):
                raise ValueError(
                    "This attribute should be an instance of class "
                    "'BaseType' or inherits from it.")
            self._resData = data
        else:
            self._resData = None

    @property
    def ECData(self):
        """Return the data for resolution function."""
        return self._ECData

    @ECData.setter
    def ECData(self, data):
        """Setter for the resData attribute."""
        if data is not None:
            if not isinstance(data, BaseType):
                raise ValueError(
                    "This attribute should be an instance of class "
                    "'BaseType' or inherits from it.")
            self._ECData = data
        else:
            self._ECData = None

    @property
    def D2OData(self):
        """Return the data for resolution function."""
        return self._D2OData

    @D2OData.setter
    def D2OData(self, data):
        """Setter for the resData attribute."""
        if data is not None:
            if not isinstance(data, BaseType):
                raise ValueError(
                    "This attribute should be an instance of class "
                    "'BaseType' or inherits from it.")
            self._D2OData = data
        else:
            self._D2OData = None

    def binData(self, binSize, axis):
        """Bin *data* attribute using the given *binSize*."""
        self.data = binData(self.data, binSize, axis)

    def scaleData(self, scale):
        """Scale intensities and errors using the provided `scale`."""
        self.data = self.data._replace(
            intensities=scale * self.data.intensities,
            errors=scale * self.data.errors)

    @ensure_attr('resData')
    def normalize_usingResFunc(self):
        """Normalizes data using integral of `resData.fit_best()`
        or experimental measurement of `resData`.
        
        """
        if not self.data.norm:
            intensities = self.data.intensities
            errors = self.data.errors

            norm = self._getNormRes()

            # Applying normalization
            self.data = self.data._replace(
                intensities=intensities / norm,
                errors=errors / norm,
                norm=True)

    def normalize_usingSelf(self):
        """Normalized data using the integral of the model or experimental."""
        if not self.data.norm:
            intensities = self.data.intensities
            energies = self.data.energies
            errors = self.data.errors

            if len(self._fit) > 0:
                x = self._fit[0].userkws['x']
                norm = self.fit_best()
            else:
                x = energies
                norm = np.copy(intensities)

            norm = (norm.sum(2) * (x[1] - x[0]))[:, :, np.newaxis]

            # Applying normalization
            self.data = self.data._replace(
                intensities=intensities / norm,
                errors=errors / norm,
                norm=True)

    @ensure_attr('ECData')
    def subtractEC(self, scaleFactor=0.95, useModel=True):
        """Use the assigned empty cell data for substraction to loaded data.

        Parameters
        ----------
        scaleFactor : float
            Empty cell data are scaled using the given
            factor prior to subtraction.
        useModel : bool
            For QENS data, use the fitted model instead of experimental 
            points to perform the subtraction if True.

        """
        if useModel:  # Use a fitted model
            ECFunc = self.ECData.fit_best()
        else:
            ECFunc = self.ECData.data.intensities

        normEC = np.zeros_like(self.data.intensities) + ECFunc

        intensities = self.data.intensities

        # If data are normalized, uses the same normalization
        # factor for empty cell data
        if self.data.norm and not self.ECData.data.norm:
            norm = self._getNormRes()
            normEC /= norm

        self.data = self.data._replace(
            intensities = intensities - scaleFactor * normEC)

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

        if useModel:  # Use a fitted model
            ECFunc = self.ECData.fit_best()
        else:
            ECFunc = self.ECData.data.intensities

        normEC = np.zeros_like(self.data.intensities) + ECFunc

        # If data are normalized, uses the same normalization
        # factor for empty cell data
        if self.data.norm:
            # assume a q-wise fit of normalized resolution function
            normFList = []
            norm = self.resData.params
            if len(norm) == normEC.shape[0]:
                for idx, val in enumerate(intensities):
                    normFList.append(norm[idx]['values'][normFactorName])
            elif len(norm) == 1:
                normFList.append(norm[0]['values'][normFactorName])
            else:
                raise ValueError("The shape of 'resData' parameters does not "
                                 "match current dataset.")
            normFList = np.array(normFList)
            normEC /= normFList

        for qIdx, angle in enumerate(self.data.selQ):
            angle = np.arcsin(neutron_wavelength * angle / (4 * np.pi))
            if canType == 'slab':
                A_S_SC, A_C_SC, A_C_C = py_absco_slab(angle, **kwargs)
            if canType == 'tube':
                A_S_SC, A_C_SC, A_C_C = py_absco_tube(angle, **kwargs)

            # Applies correction
            sampleSignal[qIdx] = ((1 / A_S_SC) * sampleSignal[:, qIdx]
                                  - A_C_SC / (A_S_SC * A_C_C)
                                  * canScaling * ECFunc[:, qIdx])

        self.data = self.data._replace(intensities=sampleSignal)

    def discardDetectors(self, *qIdx):
        """Remove detectors (q-values)."""
        ids = np.array([idx for idx, val in enumerate(self.data.qIdx) 
                        if val not in qIdx])

        self.data = self.data._replace(
            intensities=self.data.intensities[:, ids],
            errors=self.data.errors[:, ids],
            qVals=self.data.qVals[ids],
            qIdx=self.data.qIdx[ids])

    def setQRange(self, minQ, maxQ):
        """Discard detectors that do not lie inside required q-range

        Parameters
        ----------
        minQ : float 
            Minimum value of the range.
        maxQ : float 
            Maximum value of the range.

        """
        ids = np.argwhere(np.logical_and(self.data.qVals > minQ,
                                         self.data.qVals < maxQ)).flatten()
        self.data = self.data._replace(
            intensities=self.data.intensities[:, ids],
            errors=self.data.errors[:, ids],
            qVals=self.data.qVals[ids],
            qIdx=ids)

    @ensure_attr('resData')
    def _getNormRes(self):
        """Return the normalization factors from `resData`."""
        if len(self.resData._fit) > 0:
            res = self.resData.fit_best()
            x = self.resData._fit[0].userkws['x']
        else:
            x = self.resData.data.energies
            res = self.resData.data.intensities

        norm = res.sum(2) * (x[1] - x[0])

        return norm[:, :, np.newaxis]

# -------------------------------------------------------
# Methods related to data fitting
# -------------------------------------------------------
    @property
    def model(self):
        """Return the model instance."""
        return self._model

    @model.setter
    def model(self, model):
        """Setter for the model attribute."""
        if model is not None:
            if not isinstance(model, Model):
                raise ValueError("The model should be an instance of the "
                                 "'lmfit.Model' class or a class instance "
                                 "that inherits from it.")
            else:
                self._model = model
        else:
            self._model = None

    @ensure_fit
    def getFixedParams(self, obsIdx):
        """Return the fixed parameters

        The parameters are return for the given observable
        value at index `obsIdx`.

        """
        params = self._fit[obsIdx].params
        for key, par in params.items():
            par.set(vary=False)

        return params

    def fit(self, model=None, cleanData=True, convolveRes=False,
            addEC=False, addD2O=False, **kwargs):
        """Fit the dataset using the `model` attribute.

        Parameters
        ----------
        model : `lmfit.Model`, `lmfit.CompositeModel`, :class:`ConvolvedModel`
            The model to be used for fitting.
            If None, will look for a model instance in 'model' attribute of
            the class instance.
            If not None, will override the model attribute of the class
            instance.
        cleanData : bool, optional
            If True, the null or inf values in data and weights are 
            removed from the input arrays prior to fitting.
        convolveRes : bool, optional
            If True, will use the attribute `resData`, fix the parameters,
            and convolve it with the data using: 
            ``model = ConvolvedModel(self, resModel)`` 
        addEC : bool, optional
            If True, will use the attribute `ECData`, fix the parameters,
            model by calling:
            ``ECModel = self.ECData.fixedModel``
            and generate a new model by calling:
            ``model = self.model + ECModel``
        addD2O : bool, optional
            If True, will use the attribute `D2OData` to obtain the fixed
            model by calling:
            ``D2OModel = self.D2OData.fixedModel``
            and generate a new model by calling:
            ``model = self.model + D2OModel``
        kwargs : dict, optional
            Additional keyword arguments to pass to `lmfit.Model.fit` method.
            It can override any parameters obtained from the dataset, which are
            passed to the fit function ('data', 'weights', 'x',...).

        """
        print("Fitting dataset: %s" % self.fileName)

        if model is None:
            if self.model is None:
                raise ValueError("The dataset has no model associated "
                                 "with it.\n"
                                 "Please assign one before using this method "
                                 "without specifying a model.")
            else:
                model = self.model
        self.model = model

        # reset the state of '_fit'
        self._fit = []

        for idx, data in enumerate(self.data.intensities):
            q = self.data.qVals
            data = np.copy(data)
            errors = np.copy(self.data.errors[idx])
            x = np.copy(self.data.energies)

            if cleanData:
                data, errors, x = self._cleanData(data, errors, x)

            fit_kwargs = {
                'data': data,
                'weights': errors,
                'x': x,
                'q': q,
                'params': self.model.make_params()}

            if kwargs is not None:
                fit_kwargs.update(kwargs)

            # get initial guess if any available
            try:
                fit_kwargs['params'].update(model.guess(**fit_kwargs))
            except NotImplementedError:
                if idx == 0:
                    print("No guess method available with this model, "
                          "using provided default and keyword values.\n")

            if convolveRes:
                fit_kwargs['params'].update(self.resData.getFixedParams(idx))
                model = ConvolvedModel(model, self.resData.model)

            if addEC:
                fit_kwargs['params'].update(self.ECData.getFixedParams(idx))
                model = model + self.ECData.model

            if addD2O:
                fit_kwargs['params'].update(self.D2OData.getFixedParams(idx))
                model = model + self.D2OData.model

            print("\tFit of observable %i of %i (%s=%s)\r" %
                  (idx + 1, 
                   self.data.intensities.shape[0],
                   self.data.observable_name,
                   self.data.observable[idx]))

            self._fit.append(model.fit(**fit_kwargs))

        print("Done.\n")

    @property
    @ensure_fit
    def params(self):
        """Return the best values and errors from the fit result.
        
        The 'params' attribute from *lmfit* is formatted as 
        two dictionaries that are obtained by 
        ``p = self.params[idx]['values']`` and 
        ``p = self.params[idx]['errors']``.
        The 'idx' correspond to the index of the `observable` array, and
        the keys `'values'` and `'errors'` contain a dictionary of 
        parameter values and parameters errors, respectively.

        Returns
        -------
        A list of dictionaries of parameters, the first containing the fitted 
        values and the second the corresponding standard errors. Each entry 
        of the list correspond to a value of the 'observable'
        (use for example 'data.datasetList[0].data.observable').
        
        """
        out = []
        for idx, fitRes in enumerate(self._fit):
            q = fitRes.userkws['q']
            best_values = {key: val.value for key, val 
                           in fitRes.params.items()}
            stderr = {key: val.stderr for key, val in fitRes.params.items()}

            pGlob = getGlobals(best_values)

            # obtain the parameter root names
            param_root_names = []
            for p in best_values.keys():
                if p in pGlob:
                    param_root_names.append(p)
                else:
                    if p[:p.rfind('_')] not in param_root_names:
                        param_root_names.append(p[:p.rfind('_')])

            # construct the dictionary
            val = {}
            err = {}
            for p in param_root_names:
                if p in pGlob:
                    val[p] = np.zeros_like(q) + best_values[p]
                    if np.any(stderr[p] == None):
                        err[p] = np.zeros_like(q)
                    else:
                        err[p] = np.zeros_like(q) + stderr[p]
                else:
                    tmp = []
                    tmpErr = []
                    for qId, qVal in enumerate(q):
                        tmp.append(best_values[p + '_%i' % qId])
                        if np.any(stderr[p + '_%i' % qId] == None):
                            tmpErr.append(np.zeros_like(q))
                        else:
                            tmpErr.append(stderr[p + '_%i' % qId])
                    val[p] = np.array(tmp)
                    err[p] = np.array(tmpErr)
                    np.place(err[p], err[p] == None, 0.0)

            out.append({'values': val, 'errors': val})

        return out

    @ensure_fit
    def fit_best(self, **kwargs):
        """Return the fitted model.
               
        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to 
            `ModelResult.eval`.

        """
        if 'x' not in kwargs.keys():
            kwargs['x'] = self.data.energies

        return np.array([fit.eval(**kwargs) for fit in self._fit])

    @ensure_fit
    def fit_components(self, **kwargs):
        """Return the fitted components.
        
        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to 
            `ModelResult.eval_components`.

        """
        if 'x' not in kwargs.keys():
            kwargs['x'] = self.data.energies

        return np.array([fit.eval_components(**kwargs) for fit in self._fit])

    @ensure_fit
    def fit_report(self):
        """Return the fit report."""
        return [fit.fit_report() for fit in self._fit]

    @ensure_fit
    def fit_bic(self):
        """Return the bayesian information criterion from the fit."""
        return np.array([fit.bic for fit in self._fit])
    
    @ensure_fit
    def fit_aic(self):
        """Return the akaike information criterion from the fit."""
        return np.array([fit.aic for fit in self._fit])

    def _cleanData(self, data, errors, x):
        """Remove inf and null values from the input arrays."""
        mask = ~(data <= 0.) & ~(data == np.inf)

        mask1d = mask[0]
        for idx, val in enumerate(mask):
            mask1d = mask1d & val

        data = data[:, mask1d] 
        errors = errors[:, mask1d]

        return data, errors, x[mask1d]
