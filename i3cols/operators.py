#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Operators on scalar and vector columnar arrays
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

__all__ = ["split", "apply"]


try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
from collections import OrderedDict
import numpy as np
import numba
from six import string_types

from i3cols.cols import load


def split(
    path,
    on,
    categories=None,
    inkeys=None,
    outkeys=None,
    convert_indexes_to_columns=False,
):
    """Split arrays using a function, values in a column, or index"""
    if convert_indexes_to_columns:
        raise NotImplementedError("can't convert scalar indexes to columns")

    if callable(on):  # function to operate on `path`
        # Memory map only if no keys specified
        arrays, _ = load(path, keys=inkeys, mmap=inkeys is None)
        # apply(on, arrays
    elif isinstance(on, string_types):  # index or scalar column path (not ambiguous)
        pass
    elif isinstance(on, Sequence):  # key path
        pass


def apply(func, data, out_dtype=None, valid=None, index=None, **kwargs):
    """Apply a function to a scalar or vector column on a row-by-row basis,
    returning an array with one element per row.

    Parameters
    ----------
    func : callable
        If numba-compiled (numba CPUDispatcher) and no kwargs, call in
        numba-compiled loop
    out_dtype : numpy dtype or None, optional
        dtype of output numpy ndarray; if None, `out_dtype` is set to be same
        as dtype of `data`
    data : numpy ndarray
        If `data` is scalar (one entry per row), then the input length is
        num_rows; if the data is vecotr (any number of entries per row),
        then `data` can have any length
    valid : None or shape-(num_rows,) numpy ndarray, optional
    index : None or shape-(num_rows,) numpy ndarray of dtype retro_types.START_STOP_T
        Required for chunking up vector `data` by row
    **kwargs
        Passed to `func` via ``func(x, **kwargs)``

    Returns
    -------
    out : shape-(num_rows,) numpy ndarray of dtype `out_dtype`

    Notes
    -----
    If `valid` is provided, the output for rows where ``bool(valid)`` is
    False will be present but is undefined (the `out` array is initialized via
    `np.empty()` and is not filled for these cases).

    Also, if `func` is a numba-compiled callable, it will be run from a
    numba-compiled loop to minimize looping in Python.

    """
    # pylint: disable=no-else-return

    # TODO: allow behavior for dynamically figuring out `out_type` (populate a
    #   list or a numba.typed.List, and convert the returned list to a ndarray)

    if out_dtype is None:
        out_dtype = data.dtype

    if isinstance(func, numba.targets.registry.CPUDispatcher):
        if not kwargs:
            return apply_numba(
                func=func, out_dtype=out_dtype, data=data, valid=valid, index=index,
            )
        else:
            print(
                "WARNING: cannot run numba functions within a numba loop"
                " since non-empty `**kwargs` were passed; will call in a"
                " Python loop instead."
            )

    # No `valid` array
    if valid is None:
        if index is None:
            out = np.empty(shape=len(data), dtype=out_dtype)
            for i, data_ in enumerate(data):
                out[i] = func(data_, **kwargs)
            return out

        else:
            out = np.empty(shape=len(index), dtype=out_dtype)
            for i, index_ in enumerate(index):
                out[i] = func(data[index_["start"] : index_["stop"]], **kwargs)
            return out

    # Has `valid` array
    else:

        if index is None:
            out = np.empty(shape=len(data), dtype=out_dtype)
            out_valid = out[valid]
            for i, data_ in enumerate(data[valid]):
                out_valid[i] = func(data_, **kwargs)
            return out

        else:
            out = np.empty(shape=len(index), dtype=out_dtype)
            out_valid = out[valid]
            for i, index_ in enumerate(index[valid]):
                out_valid[i] = func(data[index_["start"] : index_["stop"]], **kwargs)
            return out


@numba.generated_jit(nopython=True, error_model="numpy")
def apply_numba(func, out_dtype, data, valid, index):
    """Apply a numba-compiled function to a scalar or vector data column on a
    row-by-row basis, returning an array with one element per row.

    See docs for `apply` for full documentation; but note that `apply_numba`
    does not support **kwargs.

    """
    # pylint: disable=function-redefined, unused-argument, no-else-return

    # No `valid` array
    if isinstance(valid, numba.types.NoneType):

        if isinstance(index, numba.types.NoneType):

            def apply_impl(func, out_dtype, data, valid, index):
                out = np.empty(shape=len(data), dtype=out_dtype)
                for i, data_ in enumerate(data):
                    out[i] = func(data_)
                return out

            return apply_impl

        else:

            def apply_impl(func, out_dtype, data, valid, index):
                out = np.empty(shape=len(index), dtype=out_dtype)
                for i, index_ in enumerate(index):
                    out[i] = func(data[index_["start"] : index_["stop"]])
                return out

            return apply_impl

    # Has `valid` array
    else:

        if isinstance(index, numba.types.NoneType):

            def apply_impl(func, out_dtype, data, valid, index):
                out = np.empty(shape=len(data), dtype=out_dtype)
                for i, (valid_, data_) in enumerate(zip(valid, data)):
                    if valid_:
                        out[i] = func(data_)
                return out

            return apply_impl

        else:

            def apply_impl(func, out_dtype, data, valid, index):
                out = np.empty(shape=len(index), dtype=out_dtype)
                for i, (valid_, index_) in enumerate(zip(valid, index)):
                    if valid_:
                        out[i] = func(data[index_["start"] : index_["stop"]])
                return out

            return apply_impl


@numba.generated_jit(nopython=True, error_model="numpy")
def iter_col(data, valid, index):
    """Consistent return values for a given column whether `data` is scalar or
    vector, and whether or not it contains a `valid` array.

    This unfortunately can NOT be used within a Numba njit-ed function.

    Yields
    ------
    valid : bool
        Corresponding value in `valid` array; always True if `valid` is None

    values : numpy ndarray
        Length-1 ndarray (_not_ a Numpy scalar) corresponding to value in
        `data` array if `data` is scalar, or length-N array containing the
        corresponding values if `data` is vector

    """
    # pylint: disable=function-redefined, unused-argument, no-else-return

    # No `valid` array
    if isinstance(valid, numba.types.NoneType):

        if isinstance(index, numba.types.NoneType):

            def gen_impl(data, valid, index):
                # out = np.empty(shape=len(data), dtype=data.dtype)
                for i in range(len(data)):
                    yield True, data[i : i + 1]

            return gen_impl

        else:

            def gen_impl(data, valid, index):
                # out = np.empty(shape=len(index), dtype=data.dtype)
                for index_ in index:
                    yield True, data[index_["start"] : index_["stop"]]

            return gen_impl

    # Has `valid` array
    else:

        if isinstance(index, numba.types.NoneType):

            def gen_impl(data, valid, index):
                # out = np.empty(shape=len(data), dtype=data.dtype)
                for i, valid_ in enumerate(valid):
                    yield valid_, data[i : i + 1]

            return gen_impl

        else:

            def gen_impl(data, valid, index):
                # out = np.empty(shape=len(index), dtype=data.dtype)
                for valid_, index_ in zip(valid, index):
                    yield valid_, data[index_["start"] : index_["stop"]]

            return gen_impl


# def iter_ncol(arrays):
#    kwargs = OrderedDict()
#    for key, array_d in arrays.items():
