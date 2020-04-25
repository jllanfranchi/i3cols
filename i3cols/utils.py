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
    "simplify_paths",
    "get_i3_data_fname_info",
    "test_get_i3_data_fname_info",
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
import sys
import tempfile

import numpy as np
from six import string_types

from i3cols import dtypes, regexes


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
    return tuple(
        val if i % 2 == 0 else int(val) for i, val in enumerate(NSORT_RE.split(s))
    )


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
    Use `dict2struct` directly if you do not know the dtype(s)
    of the mapping's values ahead of time.


    Parameters
    ----------
    mapping : mapping from strings to scalars

    dtype : numpy.dtype
        If scalar dtype, convert via `dict2struct`. If structured dtype,
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


def simplify_paths(paths):
    """Formulate enough of the path to disambiguate all files from one another,
    and strip .i3 and any subsequent compression extension(s).

    Parameters
    ----------
    paths : str or iterable thereof

    Returns
    -------
    simplify_paths : list of str

    """
    simplified_paths = []
    for path in paths:
        head, tail = os.path.split(expand(path))
        match = regexes.I3_FNAME_RE.match(tail)
        if match:
            groupdict = match.groupdict()
            tail = groupdict["basename"]
        simplified_paths.append(os.path.join(head, tail))

    # Remove the common part, and make sure root dir isn't referenced (if not
    # index_and_concatenate, we will join as a subfolder of `outdir`; doing
    # join(outdir, "/absolute/path") yields "/absolute/path" which will write
    # to the source dir
    fewest_path_elements = None
    split_sps = []
    for simplified_path in simplified_paths:
        split_path = simplified_path.split(os.path.sep)
        split_sps.append(split_path)
        if fewest_path_elements is None:
            fewest_path_elements = len(split_path)
        else:
            fewest_path_elements = min(fewest_path_elements, len(split_path))

    # TODO: when we drop py2, py3 has os.path.commonpath!
    for n_parts_common in range(fewest_path_elements):
        st = set(tuple(p[: n_parts_common + 1]) for p in split_sps)
        if len(st) != 1:
            break

    return [os.path.sep.join(p[n_parts_common:]).lstrip("/") for p in split_sps]


def get_i3_data_fname_info(path):
    """Extract information about an IceCube data file from its filename, as
    much as that is possible (and as much as the information varies across
    convetions used with the collaboration).

    Attempt to retrieve the following information:

        * detector: (i.e., IC79 or IC86)
        * season: (e.g., 11, 12, 2011, 2012, ...)
        * level (processing level): (e.g. "2" for level 2)
        * levelver (processing level version): (e.g., oscNext v01.04 is "01.04")
        * pass: (e.g., "2" or "2a")
        * part: (only seen in old files...?)
        * run: (number prefixed by word "run"; seen in data but might be in MC)
        * subrun: (number prfixed by word "subrun")

    Parameters
    ----------
    path : str

    Returns
    -------
    info : OrderedDict
        Only keys present are those for which information was found.

    """
    path = os.path.basename(expand(path))

    info = OrderedDict()
    for regex in [
        regexes.I3_FNAME_RE,
        regexes.I3_DETECTOR_SEASON_RE,
        regexes.I3_LEVEL_LEVELVER_RE,
        regexes.I3_PASS_RE,
        regexes.I3_RUN_RE,
        regexes.I3_SUBRUN_RE,
        regexes.I3_PART_RE,
        regexes.I3_SUBRUN_RE,
    ]:
        match = regex.search(path)
        if match:
            info.update(match.groupdict())

    # Remove items with None, all-whitespace, or empty-string values
    for key, val in list(info.items()):
        if val is None or not val.strip():
            info.pop(key)

    return info


def test_get_i3_data_fname_info():
    """Unit tests for I3_OSCNEXT_FNAME_RE."""
    # pylint: disable=line-too-long

    test_cases = [
        (
            "/tmp/i3/data/level7_v01.04/IC86.11/Run00118552/oscNext_data_IC86.11_level7_v01.04_pass2_Run00118552_Subrun00000009.i3.zst",
            {
                "basename": "",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "7",
                "pass": "2",
                "levelver": "01.04",
                "run": "",
                "part": "",
                "season": "2011",
                "subrun": "00000009",
            },
        ),
        (
            "/data/exp/IceCube/2009/filtered/level1/0510/Level1_Run00113675_Part00000000.i3.gz",
            {
                "basename": "Level1_Run00113675_Part00000000",
                "compr_exts": ".gz",
                "detector": "",
                "level": "1",
                "pass": "",
                "levelver": "",
                "run": "00113675",
                "part": "00000000",
                "season": "",
                "subrun": "",
            },
        ),
        (
            "/data/exp/IceCube/2011/filtered/level2pass2/0703/Run00118401_1/Level2pass2_IC86.2011_data_Run00118401_Subrun00000000_00000184.i3.zst",
            {
                "basename": "Level2pass2_IC86.2011_data_Run00118401_Subrun00000000_00000184",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "2",
                "pass": "2",
                "levelver": "",
                "run": "00118401",
                "part": "",
                "season": "2011",
                "subrun": "00000000_00000184",  # TODO: ???
            },
        ),
        (
            "/data/exp/IceCube/2012/filtered/level2/0101/Level2_IC86.2011_data_Run00119221_Part00000001_SLOP.i3.bz2",
            {
                "basename": "Level2_IC86.2011_data_Run00119221_Part00000001_SLOP",
                "compr_exts": ".bz2",
                "detector": "IC86",
                "level": "2",
                "pass": "",
                "levelver": "",
                "run": "00119221",
                "part": "00000001",
                "season": "2011",
                "subrun": "",
            },
        ),
        (
            "/data/exp/IceCube/2013/filtered/level2pass2a/0417/Run00122201/Level2pass2_IC86.2012_data_Run00122201_Subrun00000000_00000138.i3.zst",
            {
                "basename": "Level2pass2_IC86.2012_data_Run00122201_Subrun00000000_00000138",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "2",
                "pass": "2",
                "levelver": "",
                "run": "00122201",
                "part": "",
                "season": "2012",
                "subrun": "00000000_00000138",  # ???
            },
        ),
        (
            "/data/exp/IceCube/2014/filtered/PFFilt/0316/PFFilt_PhysicsFiltering_Run00124369_Subrun00000000_00000134.tar.bz2",
            {
                "basename": "PFFilt_PhysicsFiltering_Run00124369_Subrun00000000_00000134",
                "compr_exts": ".bz2",
                "detector": "",
                "level": "",
                "pass": "",
                "levelver": "",
                "run": "00124369",
                "part": "",
                "season": "",
                "subrun": "00000000_00000134",
            },
        ),
        (
            "/data/exp/IceCube/2015/filtered/level2pass2/0903/Run00126809_3/Level2pass2_IC86.2015_data_Run00126809_Subrun00000000_00000114.i3.zst",
            {
                "basename": "Level2pass2_IC86.2015_data_Run00126809_Subrun00000000_00000114",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "2",
                "pass": "2",
                "levelver": "",
                "run": "00126809",
                "part": "",
                "season": "2015",
                "subrun": "00000000_00000114",
            },
        ),
        (
            "/data/exp/IceCube/2016/filtered/NewWavedeform/L2/data.round3/Level2pass3/Run00127996/Level2pass3_PhysicsFiltering_Run00127996_Subrun00000000_00000139.i3.zst",
            {
                "basename": "Level2pass3_PhysicsFiltering_Run00127996_Subrun00000000_00000139",
                "compr_exts": ".zst",
                "detector": "",
                "level": "2",
                "pass": "3",
                "levelver": "",
                "run": "00127996",
                "part": "",
                "season": "",
                "subrun": "00000000_00000139",
            },
        ),
        (
            "/data/exp/IceCube/2017/filtered/level2/1231/Run00130473/Level2_IC86.2017_data_Run00130473_Subrun00000000_00000095_IT.i3.zst",
            {
                "basename": "Level2_IC86.2017_data_Run00130473_Subrun00000000_00000095_IT",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "2",
                "pass": "",
                "levelver": "",
                "run": "00130473",
                "part": "",
                "season": "2017",
                "subrun": "00000000_00000095",
            },
        ),
        (
            "/data/exp/IceCube/2018/filtered/level2/0121/Run00130574_70/Level2_IC86.2017_data_Run00130574_Subrun00000000_00000018.i3.zst",
            {
                "basename": "Level2_IC86.2017_data_Run00130574_Subrun00000000_00000018",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "2",
                "pass": "",
                "levelver": "",
                "run": "00130574",
                "part": "",
                "season": "2017",
                "subrun": "00000000_00000018",
            },
        ),
        (
            "/data/exp/IceCube/2019/filtered/level2.season2019_RHEL_6_py2-v3.1.1/0602/Run00132643/Level2_IC86.2019RHEL_6_py2-v3.1.1_data_Run00132643_Subrun00000000_00000052.i3.zst",
            {
                "basename": "Level2_IC86.2019RHEL_6_py2-v3.1.1_data_Run00132643_Subrun00000000_00000052",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "2",
                "pass": "",
                "levelver": "",
                "run": "00132643",
                "part": "",
                "season": "2019",
                "subrun": "00000000_00000052",
            },
        ),
        (
            "/data/exp/IceCube/2020/filtered/level2/0306/Run00133807_78/Level2_IC86.2019_data_Run00133807_Subrun00000000_00000029.i3.zst",
            {
                "basename": "Level2_IC86.2019_data_Run00133807_Subrun00000000_00000029",
                "compr_exts": ".zst",
                "detector": "IC86",
                "level": "2",
                "pass": "",
                "levelver": "",
                "run": "00133807",
                "part": "",
                "season": "2019",
                "subrun": "00000000_00000029",
            },
        ),
    ]

    for test_input, expected_output in test_cases:
        try:
            info = get_i3_data_fname_info(test_input)

            expected_output = deepcopy(expected_output)
            for k, v in list(expected_output.items()):
                if not v:
                    expected_output.pop(k)

            ref_keys = set(expected_output.keys())
            actual_keys = set(info.keys())
            if actual_keys != ref_keys:
                excess = actual_keys.difference(ref_keys)
                missing = ref_keys.difference(actual_keys)
                err_msg = []
                if excess:
                    err_msg.append("excess keys: " + str(sorted(excess)))
                if missing:
                    err_msg.append("missing keys: " + str(sorted(missing)))
                if err_msg:
                    raise ValueError("; ".join(err_msg))

            err_msg = []
            for key, ref_val in expected_output.items():
                actual_val = info[key]
                if actual_val != ref_val:
                    err_msg.append(
                        '"{key}": actual_val = "{actual_val}"'
                        ' but ref_val = "{ref_val}"'.format(
                            key=key, actual_val=actual_val, ref_val=ref_val
                        )
                    )
            if err_msg:
                raise ValueError("; ".join(err_msg))
        except Exception:
            sys.stderr.write('Failure on test input = "{}"\n'.format(test_input))
            raise


def normalize_oscnext_I3MCWeightDict(path, tempdir, recurse=True):
    """Remove fields that appear on only a subset of the processed oscNext
    events (from level4 (or 5?) processing on)

    Parameters
    ----------
    path : str
        Path to I3MCWeightDict column directory, path to a subrun (i.e.,
        containing the I3MCWeightDict column directory), or path to a directory
        containing all subruns (each of which contains an I3MCWeightDict column
        directory)

    tempdir : str
        Directory in which to store temporary intermediate arrays as the are
        being constructed (note a new unique directory will be created within
        this to avoid collisions with other processes)

    """
    path = expand(path)
    tempdir = expand(tempdir)
    mkdir(tempdir)

    ref_dtype = dtypes.MIN_OSCNEXT_GENIE_I3MCWEIGHTDICT_T
    ref_names = ref_dtype.names

    my_tempdir = tempfile.mkdtemp(dir=tempdir)
    try:
        tmp_path_old = os.path.join(my_tempdir, "I3MCWeightDict.data.old.npy")
        tmp_path_new = os.path.join(my_tempdir, "I3MCWeightDict.data.new.npy")

        def fix_and_replace(i3mcwd_coldir_path):
            src_path = os.path.join(i3mcwd_coldir_path, "data.npy")
            src_array = np.load(src_path, mmap_mode="r")
            if src_array.dtype.names == ref_names:
                return
            print(src_path)
            new_array = np.lib.format.open_memmap(
                tmp_path_new, mode="w+", shape=(len(src_array),), dtype=ref_dtype
            )
            for name in ref_names:
                new_array[name][:] = src_array[name][:]
            del src_array, new_array
            shutil.move(src_path, tmp_path_old)
            shutil.move(tmp_path_new, src_path)

        if os.path.basename(path) == "I3MCWeightDict":
            fix_and_replace(i3mcwd_coldir_path=path)

        for dirpath, dirs, _ in os.walk(path, followlinks=True):
            if recurse:
                dirs.sort(key=nsort_key_func)
            else:
                del dirs[:]

            if "I3MCWeightDict" not in dirs:
                continue

            fix_and_replace(i3mcwd_coldir_path=os.path.join(dirpath, "I3MCWeightDict"))

    except:
        print('ERROR; Temporary files can be found in "{}"'.format(my_tempdir))
        raise

    else:
        shutil.rmtree(my_tempdir)
