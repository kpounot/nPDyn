"""
Handle data associated with a sample.

"""
from copy import copy
from functools import wraps

import re

from collections import OrderedDict

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib

try:
    from lmfit import Model as lmModel
except ImportError:

    class lmModel:
        pass


from nPDyn.models import Model, Component, Parameters
from nPDyn.lmfit.convolvedModel import ConvolvedModel

try:
    from nPDyn.lib.pyabsco import py_absco_slab, py_absco_tube
except ImportError:
    print(
        "\nAbsorption correction libraries are not available. "
        "Paalman_Pings correction cannot be used.\n"
        "Verify that GSL libraries are available on this computer.\n"
        "On Windows, it might be necessary to use the following before any"
        "import of nPDyn:\n"
        "\t`os.add_dll_directory(<path_to_gsl_dlls>)`"
    )


HANDLED_FUNCTIONS = {}


# -------------------------------------------------------
# Useful decorators for Sample class
# -------------------------------------------------------
def ensure_fit(func):
    """Ensures the class has a fitted model."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args[0]._fit == []:
            raise ValueError(
                "Dataset (%s) has no fitted model associated with "
                "it, please fit a model before using it." % args[0].__repr__()
            )
        return func(*args, **kwargs)

    return wrapper


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


# -------------------------------------------------------
# Implementation of NumPy functions
# -------------------------------------------------------
@implements(np.sum)
def sum(arr, axis=None, **kwargs):
    return arr.__array_ufunc__(np.add, "reduce", arr, axis=axis, **kwargs)


@implements(np.cumsum)
def cumsum(arr, axis=None, **kwargs):
    return arr.__array_ufunc__(np.add, "accumulate", arr, axis=axis, **kwargs)


@implements(np.mean)
def mean(arr, axis=None, **kwargs):
    out = arr.__array_ufunc__(np.add, "reduce", arr, axis=axis, **kwargs)
    out /= arr.shape[axis] if axis is not None else arr.size
    return out


@implements(np.ravel)
def ravel(arr, order="C"):
    return arr.ravel(order)


@implements(np.transpose)
def transpose(arr, *axes):
    return arr.transpose(*axes)


@implements(np.swapaxes)
def swapaxes(arr, axis1, axis2):
    return arr.swapaxes(axis1, axis2)


@implements(np.squeeze)
def squeeze(arr):
    return arr.squeeze()


@implements(np.take)
def take(arr, indices, axis=None):
    return arr.take(indices, axis)


@implements(np.roll)
def roll(arr, shift, axis=None):
    return arr.roll(shift, axis=axis)


@implements(np.concatenate)
def concatenate(arrays, axis=0, **kwargs):
    new_arr = np.concatenate([arr._arr for arr in arrays], axis=axis, **kwargs)
    new_err = np.concatenate(
        [arr.errors for arr in arrays], axis=axis, **kwargs
    )
    new_ax = np.concatenate(
        [getattr(arr, arr.axes[axis]) for arr in arrays],
    )
    arr = Sample(
        new_arr,
        errors=new_err,
        **{
            key: val
            for key, val in arrays[0].__dict__.items()
            if key not in ("_arr", "errors")
        },
    )
    setattr(arr, arr.axes[axis], new_ax)

    return arr


@implements(np.std)
def std(arr, **kwargs):
    return np.std(arr._arr, **kwargs)


@implements(np.convolve)
def convolve(arr, v, mode="full"):
    new_arr = np.convolve(np.asarray(arr), v, mode) / v.size
    err_v = v.errors if isinstance(v, Sample) else np.zeros_like(v)
    new_err = np.sqrt(
        np.convolve(arr.errors ** 2, v ** 2, mode) / v.size
        + np.convolve(err_v ** 2, np.asarray(arr) ** 2) / v.size
    )
    out = Sample(
        new_arr,
        errors=new_err,
        **{
            key: val
            for key, val in arr.__dict__.items()
            if key not in ("_arr", "errors")
        },
    )

    return out


@implements(np.zeros_like)
def zeros_like(arr, **kwargs):
    return np.zeros_like(arr._arr, **kwargs)


@implements(np.ones_like)
def ones_like(arr, **kwargs):
    return np.ones_like(arr._arr, **kwargs)


@implements(np.max)
def max(arr, axis=None, **kwargs):
    return np.max(arr._arr, axis=axis, **kwargs)


@implements(np.min)
def min(arr, axis=None, **kwargs):
    return np.min(arr._arr, axis=axis, **kwargs)


@implements(np.argmax)
def argmax(arr, axis=None, **kwargs):
    return np.argmax(arr._arr, axis=axis, **kwargs)


@implements(np.argmin)
def argmin(arr, axis=None, **kwargs):
    return np.argmin(arr._arr, axis=axis, **kwargs)


@implements(np.stack)
def stack(arrays, axis=0):
    ax_name = "_dummy"
    if isinstance(axis, str):
        ax_name = axis
        axis = 0

    new_arr = np.stack(
        [arr._arr for arr in arrays],
        axis=axis,
    )
    new_err = np.stack(
        [arr.errors for arr in arrays],
        axis=axis,
    )
    arr = Sample(
        new_arr,
        errors=new_err,
        **{
            key: val
            for key, val in arrays[0].__dict__.items()
            if key not in ("_arr", "errors", "axes")
        },
    )
    arr.axes = copy(arrays[0].axes)
    arr.axes.insert(0, ax_name)
    if ax_name is not None:
        new_ax = np.array([getattr(a, ax_name) for a in arrays])
        new_ax = new_ax if new_ax.ndim == 1 else np.concatenate(new_ax)
        setattr(arr, arr.axes[axis], new_ax)

    return arr


@implements(np.place)
def place(arr, mask, vals):
    new_arr = np.asarray(arr)
    new_err = arr.errors
    mask = np.asarray(mask)
    np.place(new_arr, mask, vals)
    np.place(new_err, mask, np.inf)
    arr._arr = new_arr
    arr.errors = new_err


class Sample(NDArrayOperatorsMixin):
    """Handle the measured data along with metadata.

    This class is a subclass of the numpy.ndarray class with additional
    methods and attributes that are specific to neutron backscattering
    experiments.

    It can handle various operations such as addition and subtraction
    of sample data or numpy array, scaling by a scalar or an array,
    indexing, broadcasting, reshaping, binning, sliding average or
    data cleaning.

    Parameters
    ----------
    input_arr : np.ndarray, list, tuple or scalar
        Input array corresponding to sample scattering data.
    kwargs : dict (optional)
        Additional keyword arguments either for :py:meth:`np.asarray`
        or for sample metadata. The metadata are:
            - **filename**, the name of the file used to extract the data.
            - **errors**, the errors associated with scattering data.
            - **energies**, the energy transfers associated with the data.
            - **time**, the experimental time.
            - **wavelength**, the wavelength of the incoming neutrons.
            - **name**, the name for the sample.
            - **temperature**, the temperature(s) used experimentally.
            - **concentration**, the concentration of the sample.
            - **pressure**, the pressure used experimentally.
            - **buffer**, a description of the buffer used experimentally.
            - **q**, the values for the momentum transfer q.
            - **beamline**, the name of the beamline used.
            - **observable_name**, the name of the observable variable.

    Note
    ----
    The **errors** metadata is special as it is updated for various operations
    that are performed on the data array such as indexing or for the use
    of universal functions.
    For instance, indexing of the data will be performed on **errors** as
    well if its shape is the same as for the data. Also, addition,
    subtraction and other universal functions will lead to automatic error
    propagation.
    Some other metadata might change as well, like **q**, but only for
    the use of methods specific of the :py:class:`Sample` class and
    not for methods inherited from numpy.

    Examples
    --------
    A sample can be created using the following:

    >>> s1 = Sample(
    ...     np.arange(5),
    ...     dtype='float32',
    ...     errors=np.array([0.1, 0.2, 0.12, 0.14, 0.15])
    ... )

    >>> buffer = Sample(
    ...     [0., 0.2, 0.4, 0.3, 0.1],
    ...     dtype='float32',
    ...     errors=np.array([0.1, 0.2, 0.05, 0.1, 0.2])
    ... )

    where *my_data*, *my_errors* and *q_values* are numpy arrays.
    A buffer subtraction can be performed using:

    >>> s1 = s1 - buffer
    Sample([0. , 0.80000001, 1.60000002, 2.70000005, 3.9000001], dtype=float32)

    where *buffer1* is another instance of :py:class:`Sample`. The error
    propagation is automatically performed and the other attributes are taken
    from the first operand (here s1).
    Other operations such as scaling can be performed using:

    >>> s1 = 0.8 * s1
    Sample([0. , 0.80000001, 1.60000002, 2.4000001, 3.20000005], dtype=float32)

    You can transform another :py:class:`Sample` instance into a column
    vector and look how broadcasting and error propagation work:

    >>> s2 = Sample(
    ...     np.arange(5, 10),
    ...     dtype='float32',
    ...     errors=np.array([0.1, 0.3, 0.05, 0.1, 0.2])
    ... )
    >>> s2 = s2[:, np.newaxis]
    >>> res = s1 * s2
    >>> res.errors
    array([[0.5       , 1.00498756, 0.63245553, 0.76157731, 0.85      ],
           [0.6       , 1.23693169, 0.93722996, 1.23109707, 1.5       ],
           [0.7       , 1.40089257, 0.84593144, 0.99141313, 1.06887792],
           [0.8       , 1.60312195, 0.98061205, 1.15948264, 1.26491106],
           [0.9       , 1.81107703, 1.1516944 , 1.3955644 , 1.56923548]])

    """

    def __init__(self, arr, errors=None, **kwargs):
        self._arr = np.asarray(arr)

        if errors is None:
            errors = np.zeros_like(self._arr)
        self.errors = errors

        self._model = None
        self._fit = []
        for key, val in kwargs.items():
            setattr(self, key, val)

        if "name" not in kwargs:
            self.name = "New sample"

        if "observable" not in kwargs:
            self.observable = "time"

        if "axes" not in kwargs:
            self.axes = []

        self._dummy = np.array([0])

    def __array__(self, dtype=None):
        return self._arr

    def __len__(self):
        return self._arr.size

    def __repr__(self):
        out = ("Sample\n------\n" "Name:\n\t%s\n" "Axes:\n") % self.name
        for idx, val in enumerate(self.axes):
            if self._arr.ndim == len(self.axes):
                out += "\t%i - %-15s: %i %s\n" % (
                    idx,
                    val,
                    self._arr.shape[idx],
                    "entry" if self._arr.shape[idx] == 1 else "entries",
                )
            else:
                out += "\tSample array has been modified such that "
                out += "axes do not correspond to it anymore.\n"
                out += "\tPlease make sure this was an intended result.\n"
                break
        out += "Data:\n"
        out += self._arr.__repr__()

        return out

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out_cast = []
        inp_cast = [
            np.asarray(val) if isinstance(val, Sample) else val
            for val in inputs
        ]

        if "out" in kwargs.keys():
            for out in kwargs["out"]:
                if isinstance(out, Sample):
                    out_cast.append(np.asarray(out))
                else:
                    out_cast.append(out)
            kwargs["out"] = tuple(out_cast)

        obj = np.asarray(self).__array_ufunc__(
            ufunc, method, *inp_cast, **kwargs
        )
        if obj is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            obj = [obj]

        obj[0] = self._process_attributes(
            obj[0], ufunc, method, *inputs, **kwargs
        )

        return obj[0] if ufunc.nout == 1 else tuple(obj)

    def _process_attributes(self, obj, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs.keys():
            kwargs["out"] = (None,) * ufunc.nout

        obj = Sample(
            obj,
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )

        errors = [
            Sample(inp).errors if not isinstance(inp, Sample) else inp.errors
            for inp in inputs
        ]
        inputs = [np.asarray(inp) for inp in inputs]

        if ufunc == np.add or ufunc == np.subtract:
            if method == "__call__":
                obj.errors = np.sqrt(
                    np.add(
                        np.power(errors[0], 2),
                        np.power(errors[1], 2),
                        **kwargs,
                    ),
                )

            elif method == "reduce":
                obj.errors = np.sqrt(
                    np.add.reduce(np.power(errors[0], 2), **kwargs),
                )
                if kwargs["axis"] is None:
                    obj.axes = []
                else:
                    ax = kwargs["axis"]
                    ax_vals = getattr(obj, obj.axes[ax])
                    ax_vals = np.asarray([np.sum(ax_vals)])
                    setattr(obj, obj.axes[ax], ax_vals)
                    obj.axes = [
                        val for i, val in enumerate(obj.axes) if ax != i
                    ]

            elif method == "accumulate":
                obj.errors = np.sqrt(
                    np.add.accumulate(np.power(errors[0], 2), **kwargs),
                )

        elif ufunc == np.multiply:
            if method == "__call__":
                obj.errors = np.sqrt(
                    np.add(
                        np.power(inputs[1] * errors[0], 2),
                        np.power(inputs[0] * errors[1], 2),
                        **kwargs,
                    ),
                )
            elif method == "reduce":
                obj.errors = np.sqrt(
                    np.multiply.reduce(
                        np.power(inputs[0] * errors[0], 2), **kwargs
                    ),
                )
                if kwargs["axis"] is None:
                    obj.axes = []
                else:
                    ax = kwargs["axis"]
                    ax_vals = getattr(obj, obj.axes[ax])
                    ax_vals = np.asarray([np.multiply.reduce(ax_vals)])
                    setattr(obj, obj.axes[ax], ax_vals)
                    obj.axes = [
                        val for i, val in enumerate(obj.axes) if ax != i
                    ]

        elif ufunc in (
            np.divide,
            np.true_divide,
            np.floor_divide,
            np.remainder,
            np.mod,
            np.fmod,
            np.divmod,
        ):
            obj.errors = np.sqrt(
                np.add(
                    np.power(errors[0] / inputs[1], 2),
                    np.power(
                        inputs[0] / np.power(inputs[1], 2) * errors[1], 2
                    ),
                    **kwargs,
                ),
            )

        elif ufunc in (np.power, np.float_power):
            obj.errors = np.sqrt(
                np.add(
                    np.power(
                        inputs[1] * inputs[0] ** (inputs[1] - 1) * errors[0],
                        2,
                    ),
                    np.power(
                        np.exp(inputs[1]) * np.log(inputs[0]) * errors[1],
                        2,
                    ),
                    **kwargs,
                ),
            )
        elif ufunc == np.exp:
            obj.errors = np.sqrt(
                np.power(np.exp(inputs[0]) * errors[0], 2, **kwargs),
            )
        elif ufunc == np.log:
            obj.errors = np.sqrt(
                np.power(1 / inputs[0] * errors[0], 2, **kwargs)
            )
        elif ufunc == np.sqrt:
            obj.errors = np.sqrt(
                np.power(1 / (2 * np.sqrt(inputs[0])) * errors[0], 2, **kwargs)
            )
        elif ufunc == np.square:
            obj.errors = np.sqrt(
                np.power(2 * inputs[0] * errors[0], 2, **kwargs)
            )
        elif ufunc == np.cbrt:
            obj.errors = np.sqrt(
                np.power(
                    1 / (3 * inputs[0] ** (2 / 3)) * errors[0], 2, **kwargs
                )
            )
        else:
            obj.errors = np.array(errors)

        return obj

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def sum(self, axis=None, **kwargs):
        return np.sum(self, axis=axis, **kwargs)

    def cumsum(self, axis=None, **kwargs):
        return np.cumsum(self, axis=axis, **kwargs)

    def mean(self, axis=None, **kwargs):
        return np.mean(self, axis=axis, **kwargs)

    def max(self, axis=None, **kwargs):
        return np.max(self, axis=axis, **kwargs)

    def min(self, axis=None, **kwargs):
        return np.min(self, axis=axis, **kwargs)

    def argmax(self, axis=None, **kwargs):
        return np.argmax(self, axis=axis, **kwargs)

    def argmin(self, axis=None, **kwargs):
        return np.argmin(self, axis=axis, **kwargs)

    def ravel(self, order="C"):
        arr = Sample(
            self._arr.ravel(order),
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        arr.errors = arr.errors.ravel(order)
        arr.axes = []

        return arr

    def squeeze(self, axis=None):
        """Override the corresponding NumPy function to process axes too."""
        arr = Sample(
            self._arr.squeeze(axis),
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        arr.errors = self.errors.squeeze(axis)

        axes = self.axes.copy()
        if axis is None:
            ids = np.argwhere(np.array(self.shape) == 1)[:, 0]
        else:
            ids = [axis]
        axes = [val for i, val in enumerate(axes) if i not in ids]
        setattr(arr, "axes", axes)

        return arr

    def transpose(self, *axes):
        """Override the corresponding NumPy function to process axes too."""
        arr = Sample(
            self._arr.transpose(*axes),
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        arr.errors = self.errors.transpose(*axes)

        if len(axes) > 0:
            setattr(arr, "axes", list(np.array(arr.axes)[list(axes)]))
        else:
            setattr(arr, "axes", list(np.array(arr.axes)[::-1]))

        return arr

    @property
    def T(self):
        """Override the corresponding NumPy function to process axes too."""
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        """Override the corresponding NumPy function to process axes too."""
        arr = Sample(
            self._arr.swapaxes(axis1, axis2),
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        arr.errors = self.errors.swapaxes(axis1, axis2)

        tmpAxes = arr.axes.copy()
        tmpAx1 = arr.axes[axis1]
        tmpAxes[axis1] = arr.axes[axis2]
        tmpAxes[axis2] = tmpAx1
        setattr(arr, "axes", tmpAxes)

        return arr

    def take(self, indices, axis=None):
        """Override the corresponding NumPy function to process axes too."""
        arr = Sample(
            self._arr.take(indices, axis),
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        arr.errors = arr.errors.take(indices, axis)
        if axis is not None:
            attr = getattr(self, self.axes[axis])
            attr = attr.take(indices)
            setattr(arr, self.axes[axis], attr)
            if isinstance(indices, int):
                axes = self.axes.copy()
                axes.pop(axis)
                setattr(arr, "axes", axes)
        else:
            setattr(arr, "axes", [])

        return arr

    def roll(self, shift, axis=None, roll_attr=False):
        arr = Sample(
            np.roll(self._arr, shift, axis),
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        arr.errors = np.roll(arr.errors, shift, axis)
        if axis is not None and roll_attr:
            attr = getattr(self, self.axes[axis])
            attr = np.roll(attr, shift)
            setattr(arr, self.axes[axis], attr)

        return arr

    @property
    def shape(self):
        return np.asarray(self).shape

    @property
    def size(self):
        return np.asarray(self).size

    @property
    def ndim(self):
        return np.asarray(self).ndim

    def copy(self):
        out = Sample(
            np.asarray(self),
            np.copy(self.errors),
            **{
                key: val
                for key, val in self.__dict__.items()
                if key not in ("_arr", "errors")
            },
        )
        return out

    def __getitem__(self, key):
        arr = Sample(
            self._arr[key],
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        if np.asarray(self.errors).shape == self.shape:
            arr.errors = np.asarray(self.errors)[key]

        obs = getattr(self, self.observable)
        obs = np.asarray(obs)

        def process_meta(idx, key):
            try:
                ax = getattr(arr, arr.axes[idx])[key]
                setattr(arr, arr.axes[idx], ax)
            except IndexError:
                pass

        if key is None or isinstance(
            key, (int, slice, list, np.ndarray, Sample)
        ):
            process_meta(0, key)
            if isinstance(key, int):
                new_ax = arr.axes.copy()
                new_ax.pop(0)
                setattr(arr, "axes", new_ax)
        else:
            new_ax = []
            noneCount = 0
            toRemove = []
            for idx, val in enumerate(key):
                if val is None:
                    noneCount += 1
                corrId = idx - noneCount
                if any([val is Ellipsis for val in key]):
                    ellPos = key.index(Ellipsis)
                    if idx > ellPos:
                        corrId = idx - len(key)
                    if val is Ellipsis:
                        ellRangeMax = len(arr.axes) - (len(key) - ellPos)
                        ellRangeMax += noneCount
                        for i in range(corrId, ellRangeMax):
                            new_ax.append(arr.axes[i])
                            toRemove.append(arr.axes[i])
                        continue

                if not isinstance(val, int):
                    new_ax.append(
                        arr.axes[corrId] if val is not None else "_dummy"
                    )

                if val is not None:
                    toRemove.append(arr.axes[corrId])
                    process_meta(corrId, val)

            if len(new_ax) < arr.ndim:
                new_ax += [val for val in arr.axes if val not in toRemove]

            setattr(arr, "axes", new_ax)

        return arr

    def __setitem__(self, key, value):
        self._arr[key] = np.asarray(value)
        if isinstance(value, Sample):
            self.errors[key] = value.errors

    def discardData(self, indices, axis=0):
        """Discard data at given indices along the given axis.

        Parameters
        ----------
        indices : int, list
            The indices of the data to be discarded.
        axis : int
            The index of the axis along which the data are discarded.

        """
        mask = np.zeros(self.shape[axis]).astype(bool)
        mask[indices] = 1

        return self.take(np.argwhere(~mask)[:, 0], axis)

    def bin(self, bin_size, axis=-1):
        """Bin data with the given bin size along specified axis.

        Parameters
        ----------
        bin_size : int
            The size of the bin (in number of data points).
        axis : int, optional
            The axis over which the binning is to be performed.
            (default, -1 for energies)

        Returns
        -------
        out_arr : :py:class:`Sample`
            A binned instance of :py:class:`Sample` with the same
            metadata except for **errors** and the corresponding axis
            values, which are binned as well.

        """
        ax_vals = getattr(self, self.axes[axis])
        max_idx = self.shape[axis] - self.shape[axis] % bin_size

        new_arr = np.stack(
            [
                np.asarray(self).take(np.arange(idx, max_idx, bin_size), axis)
                for idx in range(bin_size)
            ]
        ).mean(0)

        new_err = (
            1
            / bin_size
            * np.sqrt(
                np.stack(
                    [
                        self.errors.take(
                            np.arange(idx, max_idx, bin_size), axis
                        )
                        ** 2
                        for idx in range(bin_size)
                    ]
                ).sum(0)
            )
        )

        new_ax = np.stack(
            [
                ax_vals.take(np.arange(idx, max_idx, bin_size))
                for idx in range(bin_size)
            ]
        ).mean(0)

        out_arr = Sample(
            new_arr,
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        out_arr.errors = np.array(new_err)
        setattr(out_arr, self.axes[axis], new_ax)

        return out_arr

    def sliding_average(self, win_size, axis=0):
        """Performs a sliding average of data and errors along given axis.

        Parameters
        ----------
        win_size : int

        axis : int, optional
            The axis over which the average is to be performed.
            (default, 0)

        Returns
        -------
        out_arr : :py:class:`Sample`
            An averaged instance of :py:class:`Sample` with the same
            metadata except for **errors** and the corresponding axis values,
            which are processed as well.

        """
        ax_vals = getattr(self, self.axes[axis])

        def kernel(array, w, axis):
            arr = np.swapaxes(array, 0, axis)
            new_arr = []
            for start, _ in enumerate(arr[:-w]):
                new_arr.append(arr[start : start + w].mean(0))
            new_arr = np.array(new_arr)

            return np.swapaxes(new_arr, 0, axis)

        new_arr = kernel(np.asarray(self), win_size, axis)
        new_err = np.sqrt(kernel(self.errors ** 2, win_size, axis))
        new_ax = kernel(ax_vals, win_size, 0)

        out_arr = Sample(
            new_arr,
            **{
                key: val for key, val in self.__dict__.items() if key != "_arr"
            },
        )
        out_arr.errors = np.array(new_err)
        setattr(out_arr, self.axes[axis], new_ax)

        return out_arr

    def normalize(self, ref=None):
        """Normalize the data using sample intensities or reference sample.

        The integration to get the normalization factor is performed
        along the energy axis.

        Parameters
        ----------
        ref : :py:class:`Sample`
            A reference sample that is used for as resolution function.

        """
        if ref is None:
            energies_idx = self.axes.index("energies")
            out = self.swapaxes(energies_idx, -1)
            energies = self.energies
            delta_E = energies[1] - energies[0]
            norm = out.sum(-1) * delta_E
            out = out / norm[..., None]
            return out.swapaxes(-1, energies_idx)
        else:
            energies_idx = self.axes.index("energies")
            ref_energies_idx = ref.axes.index("energies")
            out = self.swapaxes(energies_idx, -1)
            try:
                ref_dat = ref.fit_best(x=self.energies)
            except ValueError:
                ref_dat = ref
            ref_dat = ref_dat.swapaxes(ref_energies_idx, -1)
            energies = ref.energies
            delta_E = energies[1] - energies[0]
            norm = ref_dat.sum(-1) * delta_E
            out = out / norm[..., None]
            return out.swapaxes(energies_idx, -1)

    def absorptionCorrection(
        self,
        ec,
        canType="tube",
        canScaling=0.9,
        neutron_wavelength=6.27,
        absco_kwargs=None,
        useModel=True,
    ):
        """Computes absorption Paalman-Pings coefficients

        Can be used for sample in a flat or tubular can and apply corrections
        to data, for each q-value in *data.qVals* attribute.

        Parameters
        ----------
        ec : :py:class:`Sample`
            The data corresponding to the empty can.
        canType : {'tube', 'slab'}
            Type of can used, either 'tube' or 'slab'.
            (default, 'tube')
        canScaling : float
            Scaling factor for empty can contribution term, set it to 0
            to use only correction of sample self-attenuation.
        neutron_wavelength : float
            Incident neutrons wavelength.
        absco_kwargs : dict
            Geometry arguments for absco library.
            from Joachim Wuttke [#]_.

        References
        ----------

        .. [#] http://apps.jcns.fz-juelich.de/doku/sc/absco

        """
        # Defining some defaults arguments
        kwargs = {
            "mu_i_S": 0.660,
            "mu_f_S": 0.660,
            "mu_i_C": 0.147,
            "mu_f_C": 0.147,
        }

        if canType == "slab":
            kwargs["slab_angle"] = 45
            kwargs["thickness_S"] = 0.03
            kwargs["thickness_C_front"] = 0.5
            kwargs["thickness_C_rear"] = 0.5

        if canType == "tube":
            kwargs["radius"] = 2.15
            kwargs["thickness_S"] = 0.03
            kwargs["thickness_C_inner"] = 0.1
            kwargs["thickness_C_outer"] = 0.1

        out = self.copy()

        # process axes
        energies_ax = self.axes.index("energies")
        q_ax = self.axes.index("q")

        # Modifies default arguments with given ones, if any
        if absco_kwargs is not None:
            kwargs.update(absco_kwargs)

        if useModel and ec.model is not None:  # Use a fitted model
            ECFunc = ec.fit_best(x=self.energies)
        else:
            if ec.shape == self.shape:
                ECFunc = ec
            else:
                ids = np.argmin(
                    [(ec.energies - val) ** 2 for val in self.energies],
                    axis=1,
                )
                ECFunc = np.take(ec, ids, energies_ax)

        # Applies correction
        out = out.swapaxes(q_ax, 0)
        ECFunc = ECFunc.swapaxes(q_ax, 0)
        for qIdx, angle in enumerate(self.q):
            angle = np.arcsin(neutron_wavelength * angle / (4 * np.pi))
            if canType == "slab":
                A_S_SC, A_C_SC, A_C_C = py_absco_slab(angle, **kwargs)
            if canType == "tube":
                A_S_SC, A_C_SC, A_C_C = py_absco_tube(angle, **kwargs)

            out[qIdx] = (1 / A_S_SC) * out[qIdx] - A_C_SC / (
                A_S_SC * A_C_C
            ) * canScaling * ECFunc[qIdx]

        out = out.swapaxes(q_ax, 0)

        return out

    def _get_range(self, attr_name, min_val, max_val):
        """General accessor to get a range of an attribute values."""
        if attr_name not in self.axes:
            print(
                "Attribute %s is not available for this sample or "
                "was reduced to a scalar.\n" % attr_name
            )
            return

        attr_vals = getattr(self, attr_name)

        min_idx = np.argmin((attr_vals - min_val) ** 2)
        max_idx = np.argmin((attr_vals - max_val) ** 2)

        axis = self.axes.index(attr_name)

        out = self.take(np.arange(min_idx, max_idx), axis)
        setattr(out, attr_name, attr_vals[min_idx:max_idx])

        return out

    def get_q_range(self, min, max):
        """Helper function to select a specific momentum transfer range.

        The function assumes that q values correspond to the last
        dimension of the data set.

        Parameters
        ----------
        min : int
            The minimum value for the momentum transfer q range.
        max : int
            The maximum value for the momentum transfer q range.

        Returns
        -------
        out : :py:class:`Sample`
            A new instance of the class with the selected q range.

        """
        return self._get_range("q", min, max)

    def get_energy_range(self, min, max):
        """Helper function to select a specific energy range.

        The function assumes that time values correspond to the first
        dimension of the data set.

        Parameters
        ----------
        min : int
            The minimum value for time.
        max : int
            The maximum value for time.

        Returns
        -------
        out : :py:class:`Sample`
            A new instance of the class with the selected energy range.

        """
        return self._get_range("energies", min, max)

    def get_observable_range(self, min, max):
        """Helper function to select a specific observable range.

        The function assumes that time values correspond to the first
        dimension of the data set.

        Parameters
        ----------
        min : int
            The minimum value for the observable.
        max : int
            The maximum value for the observable.

        Returns
        -------
        out : :py:class:`Sample`
            A new instance of the class with the selected observable range.

        """
        return self._get_range(self.observable, min, max)

    # -------------------------------------------------------
    # Methods related to plotting
    # -------------------------------------------------------
    def plot(
        self,
        fig_ax=None,
        cb_ax=None,
        axis=-1,
        xlabel=None,
        ylabel="$\\rm S(q, \\hbar \\omega)$",
        label=None,
        yscale="log",
        plot_errors=True,
        plot_legend=True,
        max_lines=15,
        colormap="jet",
    ):
        """Helper function for quick plotting.

        Parameters
        ----------
        fig_ax : matplotlib Axis, optional
            An instance of Axis from *matplotlib* to be used for plotting.
            (default, None)
        cb_ax : matplotlib Axis, optional
            An instance of Axis from *matplotlib* to be used for
            the colorbar if needed.
            (default, None, for 1D arrays)
        axis : int
            The axis corresponding abscissa.
            (default, -1)
        xlabel : str
            The label for the x-axis.
            (default None, will be guessed for **axes** attribute)
        ylabel : str
            The label for the y-axis.
            (default '$\\rm S(q, \\hbar \\omega)$')
        label : str
            The label for curve.
            (default, the *name* attribute of the sample)
        yscale : str
            The scale of the y-axis.
            (default, 'log')
        plot_errors : bool
            If True, plot the error bars for each data point.
        plot_legend : bool
            If True, add the legend to the plot.
        max_lines : int
            For 2D data, maximum number of lines to be plotted.
        colormap : str
            The colormap to be used for 2D data.

        """
        if fig_ax is None:
            fig = plt.figure(figsize=(9, 6))
            if self.ndim == 1:
                ax = [fig.subplots(1, 1)]
            else:
                ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": (15, 1)})
        else:
            fig = fig_ax.figure
            ax = [fig_ax, cb_ax]

        x = getattr(self, self.axes[axis])
        xlabels = {
            "energies": "$\\rm \\hbar \\omega ~ [\\mu eV]$",
            "q": "q [$\\rm \\AA^{-1}$]",
            "qz": "$q_z ~ [\\rm \\AA^{-1}$]",
            "time": "time [h]",
            "temperature": "temperature [K]",
            "pressure": "pressure [Pa]",
        }

        if xlabel is None:
            ax_name = self.axes[axis]
            xlabel = xlabels[ax_name] if ax_name in xlabels else ax_name

        if label is None:
            label = self.name

        y = np.asarray(self)
        err = self.errors
        if plot_errors is False:
            err *= 0

        if self.ndim == 1:
            ax[0].errorbar(x, y, err, label=label)
        elif self.ndim == 2:
            cmap = get_cmap(colormap)
            y = y.T if axis == 0 else y
            err = err.T if axis == 0 else err
            step = int(np.ceil(y.shape[0] / max_lines))
            y = y[::step]
            err = err[::step]
            cb_x = getattr(self, self.axes[axis - 1])
            cb_x = cb_x[::step]
            norm = Normalize(cb_x[0], cb_x[-1])
            for idx, line in enumerate(y):
                ax[0].errorbar(
                    x,
                    line,
                    err[idx],
                    color=cmap(norm(cb_x[idx])),
                    label=label if idx == 0 else None,
                )
            cb_label = xlabels[self.axes[axis - 1]]
            ColorbarBase(ax[1], cmap=cmap, norm=norm, label=cb_label)

        ax[0].set_yscale(yscale)
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)

        if plot_legend:
            leg = ax[0].legend()
            leg.set_draggable(True)

    def plot_3D(
        self,
        fig_ax=None,
        axis="observable",
        index=0,
        xlabel=None,
        ylabel=None,
        zlabel="$\\rm S(q, \\hbar \\omega)$",
        zscale="log",
        colormap="winter",
    ):
        """Helper function for quick plotting.

        Parameters
        ----------
        fig_ax : matplotlib axis
            An instance of Axis from *matplotlib* to be used for plotting.
            (default, None)
        axis : {'observable', 'q', 'energies', 'time', 'temperature'}
            The axis along which the data are plotted.
            Valid for 3D arrays, has no effect for 2D arrays.
            (default, 'observable')
        index : int
            The index on the axis given for plotting.
            Valid for 3D arrays. For 2D, the whole dataset is plotted.
        xlabel : str
            The label for the x-axis.
            (default None, will be guessed for **axes** attribute)
        ylabel : str
            The label for the y-axis.
            (default None, will be guessed for **axes** attribute)
        zlabel : str
            The label for the z-axis.
            (default '$\\rm S(q, \\hbar \\omega)$')
        zscale : str
            The scale of the z-axis.
            (default, 'linear')
        new_fig : bool
            If true, create a new figure instead of plotting on the existing
            one.
        colormap : str
            The colormap to be used.
            (default, 'winter')

        """
        if fig_ax is None:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.subplots(subplot_kw={"projection": "3d"})
        else:
            fig = fig_ax.figure
            ax = fig_ax

        labels = {
            "energies": "$\\rm \\hbar \\omega ~ [\\mu eV]$",
            "q": "q [$\\rm \\AA^{-1}$]",
            "qz": "$q_z ~ [\\rm \\AA^{-1}$]",
            "time": "time [h]",
            "temperature": "temperature [K]",
            "pressure": "pressure [Pa]",
        }

        if axis == "observable":
            axis = self.observable
        ax_idx = self.axes.index(axis)
        z = self.swapaxes(0, ax_idx)[index]
        x = getattr(z, z.axes[0])
        xlabel = labels[z.axes[1]] if xlabel is None else xlabel
        y = getattr(z, z.axes[1])
        ylabel = labels[z.axes[0]] if ylabel is None else ylabel
        z = np.asarray(z)

        def log_tick_formatter(val, pos=None):
            return f"$10^{{{int(val)}}}$"

        if zscale == "log":
            z = np.log10(z)
            ax.zaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
            ax.zaxis.set_major_locator(MaxNLocator(integer=True))

        cmap = get_cmap(colormap)
        filt_z = z[np.isfinite(z)]
        norm = plt.Normalize(filt_z.min(), filt_z.max())
        colors = cmap(norm(z))
        xx, yy = np.meshgrid(y, x)

        surf = ax.plot_surface(
            xx,
            yy,
            z,
            rcount=np.min((20, x.size)),
            ccount=np.min((20, y.size)),
            facecolors=colors,
            shade=False,
        )

        surf.set_facecolor((0, 0, 0, 0))

        labelpad = matplotlib.rcParams["font.size"] + 1

        ax.set_xlabel(xlabel, labelpad=labelpad)
        ax.set_ylabel(ylabel, labelpad=labelpad)
        ax.set_zlabel(zlabel, labelpad=labelpad)

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
            if not isinstance(model, (lmModel, Model)):
                raise ValueError(
                    "The model should be an instance of the "
                    "Model class or of the "
                    "'lmfit.Model' class or a class instance "
                    "that inherits from it."
                )
            self._model = model
        else:
            self._model = None

    @ensure_fit
    def getFixedOptParams(self, obsIdx):
        """Return the fixed optimal parameters

        The parameters are return for the given observable
        value at index `obsIdx` or the first entry if there is only
        one observable.

        """
        fitList = self._fit

        # check if the obsIdx is within the size, use the first index if not.
        if not obsIdx <= len(fitList):
            obsIdx = 0

        bestModel = fitList[obsIdx]

        if isinstance(self.model, Model):
            params = bestModel.optParams
            for key, par in params.items():
                params.set(key, fixed=True)
        else:
            params = {}
            opt = bestModel.params
            for key, par in opt.items():
                params[key] = OrderedDict(
                    {
                        "value": par.value,
                        "min": par.min,
                        "max": par.max,
                        "vary": False,
                        "expr": par.expr,
                    }
                )

        return params

    def fit(
        self,
        model=None,
        cleanData="replace",
        res=None,
        ec=None,
        bkgd=None,
        volume_fraction_bkgd=0.95,
        **kwargs,
    ):
        """Fit the dataset using the `model` attribute.

        Parameters
        ----------
        model : :class:`Model` instance
            The model to be used for fitting.
            If None, will look for a model instance in 'model' attribute of
            the class instance.
            If not None, will override the model attribute of the class
            instance.
        cleanData : {'replace', 'omit'} or anything else for no, optional
            If set to 'replace' the locations of null or inf values in data
            are set to *np.inf* in weights prior to fitting.
            If set to 'omit' the locations of null or inf values in data
            are removed from data, weights and x prior to fitting.
            Else, nothing is done.
        res : bool, optional
            If True, will use the attribute `resData`, fix the parameters,
            and convolve it with the data using:
            ``model = ConvolvedModel(self, resModel)``
        ec : bool, optional
            If True, will use the attribute `ECData`, fix the parameters,
            model by calling:
            ``ECModel = self.ECData.fixedModel``
            and generate a new model by calling:
            ``model = self.model + ECModel``
        bkgd : bool, optional
            If True, will use the attribute `D2OData` to obtain the fixed
            model by calling:
            ``D2OModel = self.D2OData.fixedModel``
            and generate a new model by calling:
            ``model = self.model + D2OModel``
        volume_fraction_bkgd : float [0, 1]
            Volume fraction for the D2O in the sample.
            (default 0.95)
        kwargs : dict, optional
            Additional keyword arguments to pass to `Model.fit` method.
            It can override any parameters obtained from the dataset, which are
            passed to the fit function ('data', 'errors', 'x',...).

        """
        print("Fitting dataset: %s" % self.name)

        if model is None:
            if self.model is None:
                raise ValueError(
                    "The dataset has no model associated "
                    "with it.\n"
                    "Please assign one before using this method "
                    "without specifying a model."
                )
            model = self.model
        self.model = model

        # reset the state of '_fit'
        self._fit = []

        q = self.q[:, None]

        if "data" not in kwargs.keys():
            data = np.asarray(self)
        else:
            data = kwargs["data"]

        if "errors" not in kwargs.keys():
            errors = self.errors
        else:
            errors = kwargs["errors"]

        x = np.copy(self.energies)

        if cleanData in ["replace", "omit"]:
            data, errors, x = self._cleanData(data, errors, x, cleanData)

        if isinstance(self.model, lmModel):
            errors = 1 / errors

        if isinstance(self.model, Model):
            kwParams = self.model.params._paramsToList()
        else:
            kwParams = None

        fit_kwargs = {"x": x, "q": q, "params": kwParams}

        if kwargs is not None:
            fit_kwargs.update(kwargs)

        if res is not None:
            if isinstance(self.model, Model):
                resModel = res.model.copy()
                fit_kwargs["convolve"] = resModel
            else:
                resModel = res.model
                model = ConvolvedModel(model, resModel)
                model.param_hints.update(res.getFixedOptParams(0))

        if ec is not None:
            if isinstance(self.model, Model):
                ecModel = ec.model.copy()
                for key, item in ecModel.components.items():
                    item.skip_convolve = True
                model = model + ecModel
            else:
                ecModel = ec.model
                model.param_hints.update(ec.getFixedOptParams(0))

        if bkgd is not None:
            if isinstance(self.model, Model):
                model.addComponent(
                    Component(
                        "$D_2O$",
                        lambda x: volume_fraction_bkgd
                        * bkgd.fit_best(x=x, q=q)[0],
                        skip_convolve=True,
                    )
                )
            else:
                model += bkgd.model
                model *= lmModel(lambda x: volume_fraction_bkgd, prefix="D2O_")
                model.param_hints.update(bkgd.getFixedOptParams(0))

        for idx, obs in enumerate(getattr(self, self.observable)):
            # get the right observable index for data and errors
            fit_kwargs["data"] = data[idx]
            fit_kwargs["weights"] = errors[idx]

            print(
                "\tFit of observable %i of %i (%s=%s)"
                % (
                    idx + 1,
                    getattr(self, self.observable).size,
                    self.observable,
                    obs,
                ),
                end="\r",
            )

            fitRes = model.fit(**fit_kwargs)

            if isinstance(self.model, Model):
                tmp = model.copy()
                self._fit.append(tmp)
            else:
                self._fit.append(fitRes)

        if isinstance(self.model, Model):
            self.model = model.copy()
        else:
            self.model = model

        print("\nDone.\n")

    @property
    @ensure_fit
    def params(self):
        """Return the best values and errors from the fit result."""
        out = []
        for val in self._fit:
            if isinstance(self.model, Model):
                out.append(val.optParams)
            else:
                out.append(self._extract_lmfit_params(val))

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
        kws = self._fit[0].userkws.copy()
        kws.update(**kwargs)

        out = []
        for fit in self._fit:
            if isinstance(self.model, Model):
                kws["params"] = fit.optParams
            else:
                kws["params"] = fit.params

            out.append(self.model.eval(**kws))

        return np.array(out)

    @ensure_fit
    def fit_components(self, **kwargs):
        """Return the fitted components.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to
            `ModelResult.eval_components`.

        """
        kws = self._fit[0].userkws.copy()
        kws.update(**kwargs)

        comps = {}
        for fit in self._fit:
            if isinstance(self.model, Model):
                kws["params"] = fit.optParams
            else:
                kws["params"] = fit.params

            for key, val in self.model.eval_components(**kws).items():
                if key not in comps.keys():
                    comps[key] = []
                comps[key].append(val)

        for key, val in comps.items():
            comps[key] = np.array(val)

        return comps

    @property
    @ensure_fit
    def model_best(self):
        """Return the model with the fitted parameters."""
        return self._fit

    @property
    @ensure_fit
    def fit_result(self):
        """Return the full result of the fit, if available."""
        res = [fit.fitResult for fit in self._fit]
        return res

    def _cleanData(self, data, errors, x, processType="replace"):
        """Remove inf and null values from the input arrays.

        Parameters
        ----------
        processType : {'omit', 'replace'}
            Type of processing to be performed.
                - 'omit': discard the bad entries from the input arrays.
                - 'replace': replace the bad entries by inf.

        """
        if processType == "omit":
            mask = (
                ~(data <= 0.0) & ~(np.isfinite(data)) & ~(np.isfinite(errors))
            )
            for obs in mask:
                for q in obs:
                    mask[0, 0] = mask[0, 0] & q

            data = data[:, :, mask[0, 0]]
            errors = errors[:, :, mask[0, 0]]
            x = x[mask[0, 0]]

        if processType == "replace":
            np.place(errors, data <= 0.0, np.inf)
            np.place(errors, ~(np.isfinite(data)), np.inf)
            np.place(errors, ~(np.isfinite(errors)) & (errors == 0.0), np.inf)

        return data, errors, x

    def _extract_lmfit_params(self, modelRes):
        """Convert lmfit parameters into the format used in nPDyn."""
        out = Parameters()

        tmp = {}
        for key, val in modelRes.params.items():
            if re.search(r"_\d+$", key):
                key = key[: key.rfind("_")]
            if key not in tmp.keys():
                tmp[key] = {
                    "value": [],
                    "error": [],
                    "fixed": False,
                    "bounds": (-np.inf, np.inf),
                }
            tmp[key]["value"].append(val.value)
            tmp[key]["error"].append(
                val.stderr if val.stderr is not None else 0
            )
            tmp[key]["fixed"] = not val.vary
            tmp[key]["bounds"] = (val.min, val.max)

        out.update(**tmp)

        return out
