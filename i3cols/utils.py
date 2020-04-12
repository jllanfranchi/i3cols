#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order


"""
Miscellaneous utility functions
"""


from __future__ import absolute_import, division, print_function

__author__ = "Justin L. Lanfranchi"
__license__ = """Copyright 2020 Justin L. Lanfranchi

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

__all__ = [
    "NSORT_RE",
    "nsort_key_func",
    "expand",
    "mkdir",
    "set_explicit_dtype",
    "dict2struct",
    "maptype2np",
    "get_widest_float_dtype",
]


try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from collections import OrderedDict
from copy import deepcopy
import errno
from numbers import Integral, Number
import os
import re

import numpy as np
from six import string_types


NSORT_RE = re.compile(r"(\d+)")


def nsort_key_func(s):
    """Use as the `key` argument to the `sorted` function or `sort` method.

    Code adapted from nedbatchelder.com/blog/200712/human_sorting.html#comments

    Examples
    --------
    >>> l = ['f1.10.0.txt', 'f1.01.2.txt', 'f1.1.1.txt', 'f9.txt', 'f10.txt']
    >>> sorted(l, key=nsort_key_func)
    ['f1.1.1.txt', 'f1.01.2.txt', 'f1.10.0.txt', 'f9.txt', 'f10.txt']

    """
    spl = NSORT_RE.split(s)
    key = []
    for non_number, number in zip(spl[::2], spl[1::2]):
        key.append(non_number)
        key.append(int(number))
    return key


def expand(p):
    """Fully expand a path.

    Parameters
    ----------
    p : string
        Path to expand

    Returns
    -------
    e : string
        Expanded path

    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))


def mkdir(d, mode=0o0770):
    """Simple wrapper around os.makedirs to create a directory but not raise an
    exception if the dir already exists

    Parameters
    ----------
    d : string
        Directory path
    mode : integer
        Permissions on created directory; see os.makedirs for details.
    warn : bool
        Whether to warn if directory already exists.

    Returns
    -------
    first_created_dir : str or None

    """
    d = expand(d)

    # Work up in the full path to find first dir that needs to be created
    first_created_dir = None
    d_copy = deepcopy(d)
    while d_copy:
        if os.path.isdir(d_copy):
            break
        first_created_dir = d_copy
        d_copy, _ = os.path.split(d_copy)

    try:
        os.makedirs(d, mode=mode)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    return first_created_dir


def set_explicit_dtype(x):
    """Force `x` to have a numpy type if it doesn't already have one.

    Parameters
    ----------
    x : numpy-typed object, bool, integer, float
        If not numpy-typed, type is attempted to be inferred. Currently only
        bool, int, and float are supported, where bool is converted to
        np.bool8, integer is converted to np.int64, and float is converted to
        np.float64. This ensures that full precision for all but the most
        extreme cases is maintained for inferred types.

    Returns
    -------
    x : numpy-typed object

    Raises
    ------
    TypeError
        In case the type of `x` is not already set or is not a valid inferred
        type. As type inference can yield different results for different
        inputs, rather than deal with everything, explicitly failing helps to
        avoid inferring the different instances of the same object differently
        (which will cause a failure later on when trying to concatenate the
        types in a larger array).

    """
    if hasattr(x, "dtype"):
        return x

    # "value" attribute is found in basic icecube.{dataclasses,icetray} dtypes
    # such as I3Bool, I3Double, I3Int, and I3String
    if hasattr(x, "value"):
        x = x.value

    # bools are numbers.Integral, so test for bool first
    if isinstance(x, bool):
        return np.bool8(x)

    if isinstance(x, Integral):
        x_new = np.int64(x)
        assert x_new == x
        return x_new

    if isinstance(x, Number):
        x_new = np.float64(x)
        assert x_new == x
        return x_new

    if isinstance(x, string_types):
        x_new = np.string0(x)
        assert x_new == x
        return x_new

    raise TypeError("Type of argument ({}) is invalid: {}".format(x, type(x)))


def dict2struct(
    mapping, set_explicit_dtype_func=set_explicit_dtype, only_keys=None, to_numpy=True,
):
    """Convert a dict with string keys and numpy-typed values into a numpy
    array with struct dtype.


    Parameters
    ----------
    mapping : Mapping
        The dict's keys are the names of the fields (strings) and the dict's
        values are numpy-typed objects. If `mapping` is an OrderedMapping,
        produce struct with fields in that order; otherwise, sort the keys for
        producing the dict.

    set_explicit_dtype_func : callable with one positional argument, optional
        Provide a function for setting the numpy dtype of the value. Useful,
        e.g., for icecube/icetray usage where special software must be present
        (not required by this module) to do the work. If no specified,
        the `set_explicit_dtype` function defined in this module is used.

    only_keys : str, sequence thereof, or None; optional
        Only extract one or more keys; pass None to extract all keys (default)

    to_numpy : bool, optional


    Returns
    -------
    array : numpy.array of struct dtype

    """
    if only_keys and isinstance(only_keys, str):
        only_keys = [only_keys]

    out_vals = []
    out_dtype = []

    keys = mapping.keys()
    if not isinstance(mapping, OrderedDict):
        keys.sort()

    for key in keys:
        if only_keys and key not in only_keys:
            continue
        val = set_explicit_dtype_func(mapping[key])
        out_vals.append(val)
        out_dtype.append((key, val.dtype))

    out_vals = tuple(out_vals)

    if to_numpy:
        return np.array([out_vals], dtype=out_dtype)[0]

    return out_vals, out_dtype


def maptype2np(mapping, dtype, to_numpy=True):
    """Convert a mapping (containing string keys and scalar-typed values) to a
    single-element Numpy array from the values of `mapping`, using keys
    defined by `dtype.names`.

    Use this function if you already know the `dtype` you want to end up with.
    Use `retro.utils.misc.dict2struct` directly if you do not know the dtype(s)
    of the mapping's values ahead of time.


    Parameters
    ----------
    mapping : mapping from strings to scalars

    dtype : numpy.dtype
        If scalar dtype, convert via `utils.dict2struct`. If structured dtype,
        convert keys specified by the struct field names and values are
        converted according to the corresponding type.


    Returns
    -------
    array : shape-(1,) numpy.ndarray of dtype `dtype`


    See Also
    --------
    dict2struct
        Convert from a mapping to a numpy.ndarray, dynamically building `dtype`
        as you go (i.e., this is not known a priori)

    mapscalarattrs2np

    """
    out_vals = []
    for name in dtype.names:
        val = mapping[name]
        if np.isscalar(val):
            out_vals.append(val)
        else:
            out_vals.append(tuple(val))
    out_vals = tuple(out_vals)
    if to_numpy:
        return np.array([out_vals], dtype=dtype)[0]
    return out_vals, dtype


def get_widest_float_dtype(dtypes):
    """Among `dtypes` select the widest floating point type; if no floating
    point types in `dtypes`, default to numpy.float64.

    Parameters
    ----------
    dtypes : numpy dtype or iterable thereof

    Returns
    -------
    widest_float_dtype : numpy dtype

    """
    float_dtypes = [np.float128, np.float64, np.float32, np.float16]
    if isinstance(dtypes, type):
        return dtypes

    if isinstance(dtypes, Iterable):
        dtypes = set(dtypes)

    if len(dtypes) == 1:
        return next(iter(dtypes))

    for dtype in float_dtypes:
        if dtype in dtypes:
            return dtype

    return np.float64


def fuse_arrays(arrays):
    """Horizontal join of arrays: Combine into one structured array whose
    fields are taken from each component array.

    Code from user Sven Marnach, https://stackoverflow.com/a/5355974

    Parameters
    ----------
    arrays : iterable of numpy ndarrays with struct dtypes

    Returns
    -------
    array : numpy ndarray with struct dtype

    """
    arrays = list(arrays)
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset : offset + size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


# TODO
# def create_new_columns(
#     func, srcpath, srckeys=None, outdir=None, outkeys=None, overwrite=False, **kwargs
# ):
#     if outdir is not None:
#         outdir = expand(outdir)
#         assert os.path.isdir(outdir)
#         assert outkeys is not None
#
#     if not overwrite and outdir is not None and outkeys:
#         outarrays, _ = find_array_paths(outdir, keys=outkeys)
#         existing_keys = sorted(set(outkeys).intersection(outarrays.keys()))
#         if existing_keys:
#             raise IOError(
#                 'keys {} already exist in outdir "{}"'.format(existing_keys, outdir)
#             )
#
#     if isinstance(srcobj, string_types):
#         srcobj = expand(srcobj)
#         arrays, scalar_ci = load(srcobj, keys=srckeys, mmap=True)
#     elif isinstance(srcobj, Mapping):
#         arrays = srcobj
#         scalar_ci = None
