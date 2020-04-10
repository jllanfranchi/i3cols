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

__all__ = ["apply", "apply_numba"]


import numpy as np
import numba


def apply(func, data, out_dtype=None, valid=None, index=None, **kwargs):
    """Apply a function to scalar or vector data on an event-by-event basis,
    returning an array with one element per event.

    Parameters
    ----------
    func : callable
        If numba-compiled (numba CPUDispatcher) and no kwargs, call in
        numba-compiled loop
    out_dtype : numpy dtype or None, optional
        dtype of output numpy ndarray; if None, `out_dtype` is set to be same
        as dtype of `data`
    data : numpy ndarray
        If `data` is scalar (one entry per event), then the input length is
        num_events; if the data is vecotr (any number of entries per event),
        then `data` can have any length
    valid : None or shape-(num_events,) numpy ndarray, optional
    index : None or shape-(num_events,) numpy ndarray of dtype retro_types.START_STOP_T
        Required for chunking up vector `data` by event
    **kwargs
        Passed to `func` via ``func(x, **kwargs)``

    Returns
    -------
    out : shape-(num_events,) numpy ndarray of dtype `out_dtype`

    Notes
    -----
    If `valid` is provided, the output for events where ``bool(valid)`` is
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
    """Apply a numba-compiled function to scalar or vector data on an
    event-by-event basis, returning an array with one element per event.

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
