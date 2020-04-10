#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
    "get_widest_float_dtype",
]


try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from copy import deepcopy
import errno
from os import makedirs
from os import path
from os.path import abspath, expanduser, expandvars, isdir
import re

import numpy as np


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
    return abspath(expanduser(expandvars(p)))


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
        if isdir(d_copy):
            break
        first_created_dir = d_copy
        d_copy, _ = path.split(d_copy)

    try:
        makedirs(d, mode=mode)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    return first_created_dir


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


# TODO
# def create_new_columns(
#     func, srcpath, srckeys=None, outdir=None, outkeys=None, overwrite=False, **kwargs
# ):
#     if outdir is not None:
#         outdir = expand(outdir)
#         assert isdir(outdir)
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
