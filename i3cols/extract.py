#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order, import-outside-toplevel


"""
Extract information from IceCube i3 file(s) into a set of columnar arrays
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
    "RUN_DIR_RE",
    "IC_SEASON_DIR_RE",
    "OSCNEXT_I3_FNAME_RE",
    "MC_ONLY_KEYS",
    "set_explicit_dtype",
    "dict2struct",
    "maptype2np",
    "extract_files_individually",
    "extract_season",
    "extract_run",
    "combine_runs",
    "run_icetray_converter",
    "ConvertI3ToNumpy",
    "test_OSCNEXT_I3_FNAME_RE",
]

from collections import OrderedDict
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
from copy import deepcopy
from glob import glob
from multiprocessing import Pool, cpu_count
from numbers import Integral, Number
import os
import re
from shutil import rmtree
import sys
from tempfile import mkdtemp
import time

import numpy as np
from six import string_types

from i3cols import cols
from i3cols import dtypes as dt
from i3cols.utils import expand, mkdir, nsort_key_func

try:
    from processing.samples.oscNext.verification.general_mc_data_harvest_and_plot import (
        ALL_OSCNEXT_VARIABLES,
    )

    OSCNEXTKEYS = [k.split(".")[0] for k in ALL_OSCNEXT_VARIABLES.keys()]
except ImportError:
    OSCNEXTKEYS = []

# TODO: communicate a "quit NOW!" message to worker threads
# TODO: make `extract_run` use `extract_files_individually` then simply re-name
#   the category index


I3_DATA_RUN_RE = re.compile(r"Run(?P<run>[0-9]+)", flags=re.IGNORECASE)
"""Matches MC "run" dirs, e.g. '140000' & data run dirs, e.g. 'Run00125177'"""

RUN_DIR_RE = re.compile(r"(?P<pfx>Run)?(?P<run>[0-9]+)", flags=re.IGNORECASE)
"""Matches MC "run" dirs, e.g. '140000' & data run dirs, e.g. 'Run00125177'"""

IC_SEASON_DIR_RE = re.compile(
    r"((?P<detector>IC)(?P<configuration>[0-9]+)\.)(?P<year>(20)?[0-9]{2})",
    flags=re.IGNORECASE,
)
"""Matches data season dirs, e.g. 'IC86.11' or 'IC86.2011'"""

I3_FNAME_RE = re.compile(
    r"(?P<basename>.*)\.i3(?P<compr_exts>(\..*)*)", flags=re.IGNORECASE
)

OSCNEXT_I3_FNAME_RE = re.compile(
    r"""
    (?P<basename>oscNext_(?P<kind>\S+?)
        (_IC86\.(?P<season>[0-9]+))?       #  only present for data
        _level(?P<level>[0-9]+)
        .*?                                #  other infixes, e.g. "addvars"
        _v(?P<levelver>[0-9.]+)
        _pass(?P<pass>[0-9]+)
        (_Run|\.)(?P<run>[0-9]+)           # data run pfxd by "_Run", MC by "."
        ((_Subrun|\.)(?P<subrun>[0-9]+))?  # data subrun pfxd by "_Subrun", MC by "."
    )
    \.i3
    (?P<compr_exts>(\..*)*)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


MC_ONLY_KEYS = set(["I3MCWeightDict", "I3MCTree", "I3GENIEResultDict"])
"""Keys that are only valid for Monte Carlo simulation"""


MYKEYS = [
    "I3EventHeader",
    "I3TriggerHierarchy",
    "I3MCTree",
    "I3MCWeightDict",
    "I3GENIEResultDict",
    "L5_SPEFit11",
    "LineFit_DC",
    "SRTTWOfflinePulsesDC",
    "SRTTWOfflinePulsesDCTimeRange",
    "SplitInIcePulses",
    "SplitInIcePulsesTimeRange",
    "L5_oscNext_bool",
    "L6_oscNext_bool",
]
DFLT_KEYS = sorted(set(OSCNEXTKEYS + MYKEYS))


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

    match = I3_FNAME_RE.match(path)
    if match:
        groupdict = match.groupdict()
        info.update(groupdict)

    match = re.match(
        "(?P<detector>IC(?:79|86))([^0-9](?P<season>[0-9]+))", path, re.IGNORECASE
    )
    if match:
        info.update(match.groupdict)

    match = re.match(
        "level(?P<level>[0-9]+)(.?v(?P<levelver>[0-9.]+))?", path, re.IGNORECASE
    )
    if match:
        groupdict = match.groupdict()
        info.update(groupdict)

    match = re.match("pass([0-9][a-z]?)", path, re.IGNORECASE)
    if match:
        info["pass"] = match.groups()[0]

    match = re.match("([0-9]+)", path, re.IGNORECASE)
    if match:
        info["run"] = match.groups()[0]

    match = re.match("part([0-9]+)", path, re.IGNORECASE)
    if match:
        info["part"] = match.groups()[0]

    match = re.match("subrun(?P<subrun>[0-9]+(_[0-9]+)?)", path, re.IGNORECASE)
    if match:
        info.update(match.groupdict())

    for key, val in list(info.items()):
        if not val:
            info.pop(key)

    return info


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


def find_gcd_for_data_file(datafilename, gcd_dir, recurse=True):
    """Given a data file's name, attempt to extract which run it came from, and
    find the GCD for that run in `gcd_dir`.

    Parameters
    ----------
    datafilename
    gcd_dir
    recurse : bool

    Returns
    -------
    gcd_file_path

    Raises
    ------
    ValueError
        If a GCD file cannot be found

    """
    gcd_dir = expand(gcd_dir)
    assert os.path.isdir(gcd_dir)

    match = I3_DATA_RUN_RE.match(os.path.basename(datafilename))
    if match:
        run = match.groupdict()["run"].lstrip("0")
        run_gcd_re = re.compile(
            r"run0*{}.*gcd.*\.i3.*".format(run), flags=re.IGNORECASE
        )
        for dirpath, dirs, files in os.walk(gcd_dir, followlinks=True):
            if recurse:
                dirs.sort(key=nsort_key_func)
            else:
                del dirs[:]
            files.sort(key=nsort_key_func)
            for filename in files:
                if run_gcd_re.match(filename):
                    return os.path.join(dirpath, filename)

    raise IOError(
        'Could not find GCD in dir "{}" for data run file "{}"'.format(
            gcd_dir, datafilename
        )
    )


def extract_files_individually(
    paths,
    outdir,
    index_and_concatenate,
    gcd=None,
    sub_event_stream=None,
    keys=None,
    overwrite=True,
    compress=False,
    tempdir=None,
    keep_tempfiles_on_fail=False,
    procs=cpu_count(),
):
    """Exctract i3 files "individually" (if specified, an appropriate GCD file
    will also be read before a given i3 data file -- e.g., if pulses are to be
    extracted) into i3cols (directories + numpy) format in `outdir`.

    Parameters
    ----------
    paths : str or iterable thereof
    outdir : str
    index_and_concatenate : bool
    gcd : str, iterable thereof, or None; optional
    sub_event_stream : str, iterable thereof, or None; optional
    keys : str, iterable thereof, or None; optional
    overwrite : bool, optional
    compress : bool, optional
    tempdir : str, optional
    keep_tempfiles_on_fail : bool, optional
    procs : int >= 1, optional

    Notes
    -----
    It is recommended to use this function if either of the following is true:

        * It is necessary to know from which file extracted events came after
            being extracted (e.g., normalizing Monte Carlo requires knowing the
            number of source files)

        * The i3 files are "small": I.e., the time to extract a single i3 file
            is significantly larger than the time to execute the
            `run_icetray_converter` function (this instantiates
            ConvertI3ToNumpy and creates an icetray to process the (gcd+) i3
            data file

    If neither of the above hold (e.g., for all subruns within a single data
    run), it is recommended to simply call `run_icetray_converter` directly, or
    use the higher-level `extract_season` function which already does for each
    run wihile also attempting to extract all runs in parallel.

    The output column array directories are either:

        1. Written directly within `outdir`; a
            "sourcefile__category_index.npy" array is created in `outdir` to
            index into the concatenated array. This category index can be
            re-formulated later with more intelligent labeling specific to the
            filenames (e.g., if each file represents a subrun, the subrun
            number can be used instead of the full file name). Note that, in
            the case of ambiguous filenames, path(s) to the files are included
            in the index to disambiguate which source file contributed which
            events

        2. Written to subdirectories within `outdir` named after the source
            file's basename (with ".i3*" extension(s) removed)

    See also
    --------
    run_icetray_converter
    extract_season
    extract_run

    """
    if not overwrite:
        raise NotImplementedError("For now, you MUST set `overwrite` to True")

    if isinstance(paths, string_types):
        paths = [paths]
    full_paths = [expand(p) for p in sorted(paths, key=nsort_key_func)]

    # Formulate enough of the path to disambiguate all files from one another.

    simplified_paths = []
    for path in full_paths:
        head, tail = os.path.split(path)
        match = I3_FNAME_RE.match(tail)
        if match:
            groupdict = match.groupdict()
            tail = groupdict["basename"]
        simplified_paths.append(os.path.join(head, tail))

    # Remove the common part, and make sure root dir isn't referenced (if not
    # index_and_concatenate, we will join as a subfolder of `outdir`; doing
    # join(outdir, "/absolute/path") yields "/absolute/path" which will write
    # to the source dir
    fewest_path_elements = np.inf
    split_sps = []
    for simplified_path in simplified_paths:
        split_path = simplified_path.split(os.path.sep)
        split_sps.append(split_path)
        fewest_path_elements = min(fewest_path_elements, len(split_path))

    # TODO: when we drop py2, py3 has os.path.commonpath!
    for n_parts_common in range(fewest_path_elements):
        st = set(tuple(p[: n_parts_common + 1]) for p in split_sps)
        if len(st) != 1:
            break

    simplified_paths = [
        os.path.sep.join(p[n_parts_common:]).lstrip("/") for p in split_sps
    ]

    # Duplicate (ignoring compression extension(s)) paths are illegal.
    if len(set(simplified_paths)) < len(simplified_paths):
        raise ValueError("Duplicated paths detected: {}".format(paths))

    gcd_is_dir = False
    if gcd is not None:
        gcd = expand(gcd)
        gcd_is_dir = os.path.isdir(gcd)
        if not gcd_is_dir:
            assert os.path.isfile(gcd)

    if isinstance(sub_event_stream, string_types):
        sub_event_stream = [sub_event_stream]

    if isinstance(keys, string_types):
        keys = [keys]

    if tempdir is not None:
        if not index_and_concatenate:
            print("`tempdir` is not used if `index_and_concatenate` is False")
        else:
            tempdir = expand(tempdir)

    if keys is None:
        print("Extracting all keys in all files")
    else:
        if not overwrite:
            existing_arrays, _ = cols.find_array_paths(outdir, keys=keys)
            existing_keys = set(existing_arrays.keys())
            redundant_keys = existing_keys.intersection(keys)
            if redundant_keys:
                print("Will not extract existing keys:", sorted(redundant_keys))
                keys = [k for k in keys if k not in redundant_keys]

        if len(keys) == 0:
            print("No keys to extract!")
            return

        print("Keys remaining to extract:", keys)

    full_tempdir = None
    paths_to_compress = []
    results = []
    sourcefile_array_map = OrderedDict()
    pool = None
    if procs > 1:
        pool = Pool(min(procs, len(full_paths)))
    try:
        if index_and_concatenate:
            if tempdir is not None:
                mkdir(tempdir)
            full_tempdir = mkdtemp(dir=tempdir)

        for full_path, simplified_path in zip(full_paths, simplified_paths):
            if index_and_concatenate:
                thisfile_outdir = os.path.join(full_tempdir, simplified_path)
                sourcefile_array_map[simplified_path] = thisfile_outdir
            else:
                thisfile_outdir = os.path.join(outdir, simplified_path)
                if compress:
                    paths_to_compress.append(thisfile_outdir)

            if gcd is None:
                extract_paths = [full_path]
            else:
                gcd_file = deepcopy(gcd)
                if gcd_is_dir:
                    gcd_file = find_gcd_for_data_file(
                        datafilename=os.path.basename(simplified_path), gcd_dir=gcd
                    )
                extract_paths = [gcd_file, full_path]

            kwargs = dict(
                paths=extract_paths,
                outdir=thisfile_outdir,
                sub_event_stream=sub_event_stream,
                keys=keys,
            )

            if procs == 1:
                run_icetray_converter(**kwargs)
            else:
                results.append(
                    pool.apply_async(run_icetray_converter, tuple(), kwargs)
                )

        if pool is not None:
            pool.close()
            pool.join()

        for result in results:
            result.get()

        if index_and_concatenate:
            for simplified_path, thisfile_outdir in list(sourcefile_array_map.items()):
                sourcefile_array_map[simplified_path], _ = cols.find_array_paths(
                    thisfile_outdir
                )
            cols.index_and_concatenate_arrays(
                category_array_map=sourcefile_array_map,
                category_name="sourcefile",
                outdir=outdir,
            )
            if compress:
                paths_to_compress = outdir

    except:
        if full_tempdir is not None:
            if keep_tempfiles_on_fail:
                print(
                    'Temp dir/files will NOT be removed; see "{}" to inspect'
                    ' and manually remove'.format(full_tempdir)
                )
            else:
                try:
                    rmtree(full_tempdir)
                except Exception as err:
                    print(err)
        raise

    else:
        if full_tempdir is not None:
            try:
                rmtree(full_tempdir)
            except Exception as err:
                print(err)

    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception as err:
                print(err)

    if compress:
        # TODO: keys weren't created by this function are also compressed by
        # using `recurse`, but we don't know what kesy were generated except in
        # particular cases. Do we care that this could compress files it didn't
        # create?
        cols.compress(paths=paths_to_compress, recurse=True, keep=False, procs=procs)


def extract_season(
    path,
    outdir=None,
    tempdir=None,
    gcd=None,
    keys=None,
    overwrite=False,
    mmap=True,
    keep_tempfiles_on_fail=False,
    procs=cpu_count(),
):
    """E.g. .. ::

        data/level7_v01.04/IC86.14

    Parameters
    ----------
    path : str
    outdir : str
    tempdir : str or None, optional
        Intermediate arrays will be written to this directory.
    gcd : str or None, optional
    keys : str, iterable thereof, or None; optional
    overwrite : bool, optional
    mmap : bool, optional
    keep_tempfiles_on_fail : bool, optional

    """
    path = expand(path)
    assert os.path.isdir(path), path

    outdir = expand(outdir)
    if tempdir is not None:
        tempdir = expand(tempdir)

    pool = None
    try:
        if tempdir is not None:
            mkdir(tempdir)
        full_tempdir = mkdtemp(dir=tempdir)

        if gcd is None:
            gcd = path
        assert isinstance(gcd, string_types)
        gcd = expand(gcd)
        assert os.path.isdir(gcd)

        if isinstance(keys, string_types):
            keys = [keys]

        match = IC_SEASON_DIR_RE.match(os.path.basename(path))
        assert match, 'path not a season directory? "{}"'.format(os.path.basename(path))
        groupdict = match.groupdict()

        year_str = groupdict["year"]
        assert year_str is not None

        print("outdir:", outdir)
        print("full_tempdir:", full_tempdir)

        if keys is None:
            print("extracting all keys in all files")
        else:
            if not overwrite:
                existing_arrays, _ = cols.find_array_paths(outdir)
                existing_keys = set(existing_arrays.keys())
                redundant_keys = existing_keys.intersection(keys)
                if redundant_keys:
                    print("will not extract existing keys:", sorted(redundant_keys))
                    keys = [k for k in keys if k not in redundant_keys]
            invalid_keys = MC_ONLY_KEYS.intersection(keys)
            if invalid_keys:
                print(
                    "MC-only keys {} were specified but this is data, so these"
                    " will be skipped.".format(sorted(invalid_keys))
                )
            keys = [k for k in keys if k not in MC_ONLY_KEYS]
            print("keys remaining to extract:", keys)

            if len(keys) == 0:
                print("nothing to do!")
                return

        run_dirpaths = []
        for basepath in sorted(os.listdir(path), key=nsort_key_func):
            match = RUN_DIR_RE.match(basepath)
            if not match:
                continue
            groupdict = match.groupdict()
            run_int = np.uint32(groupdict["run"])
            run_dirpaths.append((run_int, os.path.join(path, basepath)))
        # Ensure sorting by run
        run_dirpaths.sort()

        t0 = time.time()

        run_tempdirs = []

        if procs > 1:
            pool = Pool(procs)
        for run, run_dirpath in run_dirpaths:
            run_tempdir = os.path.join(full_tempdir, "Run{}".format(run))
            mkdir(run_tempdir)
            run_tempdirs.append(run_tempdir)

            kw = dict(
                path=run_dirpath,
                outdir=run_tempdir,
                gcd=gcd,
                keys=keys,
                overwrite=True,
                mmap=mmap,
                keep_tempfiles_on_fail=keep_tempfiles_on_fail,
                procs=procs,
            )
            if procs == 1:
                extract_run(**kw)
            else:
                pool.apply_async(extract_run, tuple(), kw)
        if procs > 1:
            pool.close()
            pool.join()

        combine_runs(path=full_tempdir, outdir=outdir, keys=keys, mmap=mmap)

    except:
        if not keep_tempfiles_on_fail:
            try:
                rmtree(full_tempdir)
            except Exception as err:
                print(err)
        raise

    else:
        try:
            rmtree(full_tempdir)
        except Exception as err:
            print(err)

    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception as err:
                print(err)

    print(
        '{} s to extract season path "{}" to "{}"'.format(
            time.time() - t0, path, outdir
        )
    )


def extract_run(
    path,
    outdir=None,
    tempdir=None,
    gcd=None,
    sub_event_stream=None,
    keys=None,
    overwrite=False,
    mmap=True,
    keep_tempfiles_on_fail=False,
    procs=cpu_count(),
):
    """E.g. .. ::

        data/level7_v01.04/IC86.14/Run00125177
        genie/level7_v01.04/140000

    Note that what can be considered "subruns" for both data and MC are
    represented as files in both, at least for this version of oscNext.

    Parameters
    ----------
    path : str
    outdir : str
    tempdir : str or None, optional
        Intermediate arrays will be written to this directory.
    gcd : str or None, optional
    sub_event_stream : str, sequence thereof, or None; optional
    keys : str, iterable thereof, or None; optional
    overwrite : bool, optional
    mmap : bool, optional
    keep_tempfiles_on_fail : bool, optional

    """
    path = expand(path)
    assert os.path.isdir(path), path

    outdir = expand(outdir)
    if tempdir is not None:
        tempdir = expand(tempdir)

    if isinstance(sub_event_stream, string_types):
        sub_event_stream = [sub_event_stream]

    if isinstance(keys, string_types):
        keys = [keys]

    pool = None
    mkdir(outdir)
    try:
        if tempdir is not None:
            mkdir(tempdir)
        full_tempdir = mkdtemp(dir=tempdir)

        match = RUN_DIR_RE.match(os.path.basename(path))
        assert match, 'path not a run directory? "{}"'.format(os.path.basename(path))
        groupdict = match.groupdict()

        is_data = groupdict["pfx"] is not None
        is_mc = not is_data
        run_str = groupdict["run"]
        run_int = int(groupdict["run"].lstrip("0"))

        if is_mc:
            print("is_mc")
            assert isinstance(gcd, string_types) and os.path.isfile(expand(gcd)), str(gcd)
            gcd = expand(gcd)
        else:
            print("is_data")
            if gcd is None:
                gcd = path
            assert isinstance(gcd, string_types)
            gcd = expand(gcd)
            if not os.path.isfile(gcd):
                assert os.path.isdir(gcd)
                # TODO: use DATA_GCD_FNAME_RE
                gcd = glob(os.path.join(gcd, "*Run{}*GCD*.i3*".format(run_str)))
                assert len(gcd) == 1, gcd
                gcd = expand(gcd[0])

        if keys is None:
            print("extracting all keys in all files")
        else:
            if not overwrite:
                existing_arrays, _ = cols.find_array_paths(outdir, keys=keys)
                existing_keys = set(existing_arrays.keys())
                redundant_keys = existing_keys.intersection(keys)
                if redundant_keys:
                    print("will not extract existing keys:", sorted(redundant_keys))
                    keys = [k for k in keys if k not in redundant_keys]
            if is_data:
                invalid_keys = MC_ONLY_KEYS.intersection(keys)
                if invalid_keys:
                    print(
                        "MC-only keys {} were specified but this is data, so these"
                        " will be skipped.".format(sorted(invalid_keys))
                    )
                keys = [k for k in keys if k not in MC_ONLY_KEYS]

            print("keys remaining to extract:", keys)

            if len(keys) == 0:
                print("nothing to do!")
                return

        subrun_filepaths = []
        for basepath in sorted(os.listdir(path), key=nsort_key_func):
            match = OSCNEXT_I3_FNAME_RE.match(basepath)
            if not match:
                continue
            groupdict = match.groupdict()
            assert int(groupdict["run"]) == run_int
            subrun_int = int(groupdict["subrun"])
            subrun_filepaths.append((subrun_int, os.path.join(path, basepath)))
        # Ensure sorting by subrun
        subrun_filepaths.sort()

        t0 = time.time()
        if is_mc:
            arrays = OrderedDict()
            requests = OrderedDict()

            subrun_tempdirs = []
            if procs > 1:
                pool = Pool(procs)

            for subrun, fpath in subrun_filepaths:
                print(fpath)
                paths = [gcd, fpath]

                subrun_tempdir = os.path.join(full_tempdir, "subrun{}".format(subrun))
                mkdir(subrun_tempdir)
                subrun_tempdirs.append(subrun_tempdir)

                kw = dict(
                    paths=paths,
                    sub_event_stream=sub_event_stream,
                    keys=keys,
                    outdir=subrun_tempdir,
                )
                if procs == 1:
                    arrays[subrun] = run_icetray_converter(**kw)
                else:
                    requests[subrun] = pool.apply_async(
                        run_icetray_converter, tuple(), kw
                    )
            if procs > 1:
                for key, result in requests.items():
                    arrays[key] = result.get()

            cols.index_and_concatenate_arrays(
                arrays, category_name="subrun", outdir=outdir, mmap=mmap,
            )

        else:  # is_data
            paths = [gcd] + [fpath for _, fpath in subrun_filepaths]
            run_icetray_converter(
                paths=paths, sub_event_stream=sub_event_stream, keys=keys, outdir=outdir
            )

    except Exception:
        if not keep_tempfiles_on_fail:
            try:
                rmtree(full_tempdir)
            except Exception as err:
                print(err)
        raise

    else:
        try:
            rmtree(full_tempdir)
        except Exception as err:
            print(err)

    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception as err:
                print(err)

    print(
        '{} s to extract run path "{}" to "{}"'.format(time.time() - t0, path, outdir)
    )


def combine_runs(path, outdir, keys=None, mmap=True):
    """
    Parameters
    ----------
    path : str
        IC86.XX season directory or MC directory that contains
        already-extracted arrays

    outdir : str
        Store concatenated arrays to this directory

    keys : str, iterable thereof, or None; optional
        Only preserver these keys. If None, preserve all keys found in all
        subpaths

    mmap : bool
        Note that if `mmap` is True, ``load_contained_paths`` will be called
        with `inplace=False` or else too many open files will result

    """
    path = expand(path)
    assert os.path.isdir(path), str(path)
    outdir = expand(outdir)

    run_dirs = []
    for subname in sorted(os.listdir(path), key=nsort_key_func):
        subpath = os.path.join(path, subname)
        if not os.path.isdir(subpath):
            continue
        match = RUN_DIR_RE.match(subname)
        if not match:
            continue
        groupdict = match.groupdict()
        run_str = groupdict["run"]
        run_int = int(run_str.lstrip("0"))
        run_dirs.append((run_int, subpath))
    # Ensure sorting by numerical run number
    run_dirs.sort()

    print("{} run dirs found".format(len(run_dirs)))

    array_map = OrderedDict()
    existing_category_indexes = OrderedDict()

    for run_int, run_dir in run_dirs:
        array_map[run_int], csi = cols.find_array_paths(run_dir, keys=keys)
        if csi:
            existing_category_indexes[run_int] = csi

    mkdir(outdir)
    cols.index_and_concatenate_arrays(
        category_array_map=array_map,
        existing_category_indexes=existing_category_indexes,
        category_name="run",
        category_dtype=np.uint32,  # see retro_types.I3EVENTHEADER_T
        outdir=outdir,
        mmap=mmap,
    )


def run_icetray_converter(paths, outdir, sub_event_stream, keys):
    """Simple function callable, e.g., by subprocesses (i.e., to run in
    parallel)

    Parameters
    ----------
    paths
    outdir
    sub_event_stream
    keys

    Returns
    -------
    arrays : list of dict

    """
    from I3Tray import I3Tray

    converter = ConvertI3ToNumpy()

    tray = I3Tray()
    tray.AddModule(_type="I3Reader", _name="reader", FilenameList=paths)
    tray.Add(
        _type=converter,
        _name="ConvertI3ToNumpy",
        sub_event_stream=sub_event_stream,
        keys=keys,
    )
    tray.Execute()
    tray.Finish()

    arrays = converter.finalize_icetray(outdir=outdir)

    del tray, I3Tray

    return arrays


class ConvertI3ToNumpy(object):
    """
    Convert icecube objects to Numpy typed objects
    """

    __slots__ = [
        "icetray",
        "dataio",
        "dataclasses",
        "i3_scalars",
        "custom_funcs",
        "getters",
        "mapping_str_simple_scalar",
        "mapping_str_structured_scalar",
        "mapping_str_attrs",
        "attrs",
        "unhandled_types",
        "frame",
        "failed_keys",
        "frame_data",
    ]

    def __init__(self):
        # pylint: disable=unused-variable, unused-import
        from icecube import icetray, dataio, dataclasses, recclasses, simclasses

        try:
            from icecube import millipede
        except ImportError:
            millipede = None

        try:
            from icecube import santa
        except ImportError:
            santa = None

        try:
            from icecube import genie_icetray
        except ImportError:
            genie_icetray = None

        try:
            from icecube import tpx
        except ImportError:
            tpx = None

        self.icetray = icetray
        self.dataio = dataio
        self.dataclasses = dataclasses

        self.i3_scalars = {
            icetray.I3Bool: np.bool8,
            icetray.I3Int: np.int32,
            dataclasses.I3Double: np.float64,
            dataclasses.I3String: np.string0,
        }

        self.custom_funcs = {
            dataclasses.I3MCTree: self.extract_flat_mctree,
            dataclasses.I3RecoPulseSeries: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMap: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMapMask: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMapUnion: self.extract_flat_pulse_series,
            dataclasses.I3SuperDSTTriggerSeries: self.extract_seq_of_same_type,
            dataclasses.I3TriggerHierarchy: self.extract_flat_trigger_hierarchy,
            dataclasses.I3VectorI3Particle: self.extract_singleton_seq_to_scalar,
            dataclasses.I3DOMCalibration: self.extract_i3domcalibration,
        }

        self.getters = {recclasses.I3PortiaEvent: (dt.I3PORTIAEVENT_T, "Get{}")}

        self.mapping_str_simple_scalar = {
            dataclasses.I3MapStringDouble: np.float64,
            dataclasses.I3MapStringInt: np.int32,
            dataclasses.I3MapStringBool: np.bool8,
        }

        self.mapping_str_structured_scalar = {}
        if genie_icetray:
            self.mapping_str_structured_scalar[
                genie_icetray.I3GENIEResultDict
            ] = dt.I3GENIERESULTDICT_SCALARS_T

        self.mapping_str_attrs = {dataclasses.I3FilterResultMap: dt.I3FILTERRESULT_T}

        self.attrs = {
            icetray.I3RUsage: dt.I3RUSAGE_T,
            icetray.OMKey: dt.OMKEY_T,
            dataclasses.TauParam: dt.TAUPARAM_T,
            dataclasses.LinearFit: dt.LINEARFIT_T,
            dataclasses.SPEChargeDistribution: dt.SPECHARGEDISTRIBUTION_T,
            dataclasses.I3Direction: dt.I3DIRECTION_T,
            dataclasses.I3EventHeader: dt.I3EVENTHEADER_T,
            dataclasses.I3FilterResult: dt.I3FILTERRESULT_T,
            dataclasses.I3Position: dt.I3POSITION_T,
            dataclasses.I3Particle: dt.I3PARTICLE_T,
            dataclasses.I3ParticleID: dt.I3PARTICLEID_T,
            dataclasses.I3VEMCalibration: dt.I3VEMCALIBRATION_T,
            dataclasses.SPEChargeDistribution: dt.SPECHARGEDISTRIBUTION_T,
            dataclasses.I3SuperDSTTrigger: dt.I3SUPERDSTTRIGGER_T,
            dataclasses.I3Time: dt.I3TIME_T,
            dataclasses.I3TimeWindow: dt.I3TIMEWINDOW_T,
            recclasses.I3DipoleFitParams: dt.I3DIPOLEFITPARAMS_T,
            recclasses.I3LineFitParams: dt.I3LINEFITPARAMS_T,
            recclasses.I3FillRatioInfo: dt.I3FILLRATIOINFO_T,
            recclasses.I3FiniteCuts: dt.I3FINITECUTS_T,
            recclasses.I3DirectHitsValues: dt.I3DIRECTHITSVALUES_T,
            recclasses.I3HitStatisticsValues: dt.I3HITSTATISTICSVALUES_T,
            recclasses.I3HitMultiplicityValues: dt.I3HITMULTIPLICITYVALUES_T,
            recclasses.I3TensorOfInertiaFitParams: dt.I3TENSOROFINERTIAFITPARAMS_T,
            recclasses.I3Veto: dt.I3VETO_T,
            recclasses.I3CLastFitParams: dt.I3CLASTFITPARAMS_T,
            recclasses.I3CscdLlhFitParams: dt.I3CSCDLLHFITPARAMS_T,
            recclasses.I3DST16: dt.I3DST16_T,
            recclasses.DSTPosition: dt.DSTPOSITION_T,
            recclasses.I3StartStopParams: dt.I3STARTSTOPPARAMS_T,
            recclasses.I3TrackCharacteristicsValues: dt.I3TRACKCHARACTERISTICSVALUES_T,
            recclasses.I3TimeCharacteristicsValues: dt.I3TIMECHARACTERISTICSVALUES_T,
            recclasses.CramerRaoParams: dt.CRAMERRAOPARAMS_T,
        }
        if millipede:
            self.attrs[
                millipede.gulliver.I3LogLikelihoodFitParams
            ] = dt.I3LOGLIKELIHOODFITPARAMS_T
        if santa:
            self.attrs[santa.I3SantaFitParams] = dt.I3SANTAFITPARAMS_T

        # Define types we know we don't handle; these will be expanded as new
        # types are encountered to avoid repeatedly failing on the same types

        self.unhandled_types = set(
            [
                dataclasses.I3Geometry,
                dataclasses.I3Calibration,
                dataclasses.I3DetectorStatus,
                dataclasses.I3DOMLaunchSeriesMap,
                dataclasses.I3MapKeyVectorDouble,
                dataclasses.I3RecoPulseSeriesMapApplySPECorrection,
                dataclasses.I3SuperDST,
                dataclasses.I3TimeWindowSeriesMap,
                dataclasses.I3VectorDouble,
                dataclasses.I3VectorOMKey,
                dataclasses.I3VectorTankKey,
                dataclasses.I3MapKeyDouble,
                recclasses.I3DSTHeader16,
            ]
        )
        if tpx:
            self.unhandled_types.add(tpx.I3TopPulseInfoSeriesMap)

        self.frame = None
        self.failed_keys = set()
        self.frame_data = []

    def __call__(self, frame, sub_event_stream=None, keys=None):
        """Allows calling the instantiated class directly, which is the
        mechanism IceTray uses (including requiring `frame` as the first
        argument)

        Parameters
        ----------
        frame : icetray.I3Frame
        sub_event_stream : str, iterable thereof, or None; optional
            Only process frames from these sub event streams. If None, process
            all sub event streams.
        keys : str, iterable thereof, or None, optional
            Extract only these keys

        Returns
        -------
        False
            This disallows frames from being pushed to subsequent modules. I
            don't know why I picked this value. Probably not the "correct"
            value, so modify if this is an issue or there is a better way.

        """
        if frame.Stop != self.icetray.I3Frame.Physics:
            return False

        if sub_event_stream is not None:
            if isinstance(sub_event_stream, string_types):
                if frame["I3EventHeader"].sub_event_stream != sub_event_stream:
                    return False
            elif frame["I3EventHeader"].sub_event_stream not in set(sub_event_stream):
                return False

        frame_data = self.extract_frame(frame, keys=keys)
        self.frame_data.append(frame_data)
        return False

    def finalize_icetray(self, outdir=None):
        """Construct arrays and cleanup data saved when running via icetray
        (i.e., the __call__ method)

        Parameters
        ----------
        outdir : str or None, optional
            If string, interpret as path to a directory in which to save the
            arrays (they are written to memory-mapped files to avoid excess
            memory usage). If None, exclusively construct the arrays in memory
            (do not save to disk).

        Returns
        -------
        arrays
            See `construct_arrays` for format of `arrays`

        """
        arrays = cols.construct_arrays(self.frame_data, outdir=outdir)
        del self.frame_data[:]
        return arrays

    def extract_frame(self, frame, keys=None):
        """Extract icetray frame objects to numpy typed objects

        Parameters
        ----------
        frame : icetray.I3Frame
        keys : str, iterable thereof, or None; optional

        """
        self.frame = frame

        auto_mode = False
        if keys is None:
            auto_mode = True
            keys = frame.keys()
        elif isinstance(keys, str):
            keys = [keys]
        keys = sorted(set(keys).difference(self.failed_keys))

        extracted_data = {}

        for key in keys:
            try:
                value = frame[key]
            except Exception:
                if auto_mode:
                    self.failed_keys.add(key)
                # else:
                #    extracted_data[key] = None
                continue

            try:
                np_value = self.extract_object(value)
            except Exception:
                print("failed on key {}".format(key))
                raise

            # if auto_mode and np_value is None:
            if np_value is None:
                continue

            extracted_data[key] = np_value

        return extracted_data

    def extract_object(self, obj, to_numpy=True):
        """Convert an object from a frame to a Numpy typed object.

        Note that e.g. extracting I3RecoPulseSeriesMap{Mask,Union} requires
        that `self.frame` be assigned the current frame to work.

        Parameters
        ----------
        obj : frame object
        to_numpy : bool, optional

        Returns
        -------
        np_obj : numpy-typed object or None

        """
        obj_t = type(obj)

        if obj_t in self.unhandled_types:
            return None

        dtype = self.i3_scalars.get(obj_t, None)
        if dtype:
            val = dtype(obj.value)
            if to_numpy:
                return val
            return val, dtype

        dtype_fmt = self.getters.get(obj_t, None)
        if dtype_fmt:
            return self.extract_getters(obj, *dtype_fmt, to_numpy=to_numpy)

        dtype = self.mapping_str_simple_scalar.get(obj_t, None)
        if dtype:
            return dict2struct(obj, set_explicit_dtype_func=dtype, to_numpy=to_numpy)

        dtype = self.mapping_str_structured_scalar.get(obj_t, None)
        if dtype:
            return maptype2np(obj, dtype=dtype, to_numpy=to_numpy)

        dtype = self.mapping_str_attrs.get(obj_t, None)
        if dtype:
            return self.extract_mapscalarattrs(obj, to_numpy=to_numpy)

        dtype = self.attrs.get(obj_t, None)
        if dtype:
            return self.extract_attrs(obj, dtype, to_numpy=to_numpy)

        func = self.custom_funcs.get(obj_t, None)
        if func:
            return func(obj, to_numpy=to_numpy)

        # New unhandled type found
        print("WARNING: found new unhandled type: {}".format(obj_t))
        self.unhandled_types.add(obj_t)

        return None

    @staticmethod
    def extract_flat_trigger_hierarchy(obj, to_numpy=True):
        """Flatten a trigger hierarchy into a linear sequence of triggers,
        labeled such that the original hiercarchy can be recreated

        Parameters
        ----------
        obj : I3TriggerHierarchy
        to_numpy : bool, optional

        Returns
        -------
        flat_triggers : shape-(N-trigers,) numpy.ndarray of dtype FLAT_TRIGGER_T

        """
        iterattr = obj.items if hasattr(obj, "items") else obj.iteritems

        level_tups = []
        flat_triggers = []

        for level_tup, trigger in iterattr():
            level_tups.append(level_tup)
            level = len(level_tup) - 1
            if level == 0:
                parent_idx = -1
            else:
                parent_idx = level_tups.index(level_tup[:-1])
            # info_tup, _ = self.extract_attrs(trigger, TRIGGER_T, to_numpy=False)
            key = trigger.key
            flat_triggers.append(
                (
                    level,
                    parent_idx,
                    (
                        trigger.time,
                        trigger.length,
                        trigger.fired,
                        (key.source, key.type, key.subtype, key.config_id or 0),
                    ),
                )
            )

        if to_numpy:
            return np.array(flat_triggers, dtype=dt.FLAT_TRIGGER_T)

        return flat_triggers, dt.FLAT_TRIGGER_T

    def extract_flat_mctree(
        self,
        mctree,
        parent=None,
        parent_idx=-1,
        level=0,
        max_level=-1,
        flat_particles=None,
        to_numpy=True,
    ):
        """Flatten an I3MCTree into a sequence of particles with additional
        metadata "level" and "parent" for easily reconstructing / navigating the
        tree structure if need be.

        Parameters
        ----------
        mctree : icecube.dataclasses.I3MCTree
            Tree to flatten into a numpy array

        parent : icecube.dataclasses.I3Particle, optional

        parent_idx : int, optional

        level : int, optional

        max_level : int, optional
            Recurse to but not beyond `max_level` depth within the tree. Primaries
            are level 0, secondaries level 1, tertiaries level 2, etc. Set to
            negative value to capture all levels.

        flat_particles : appendable sequence or None, optional

        to_numpy : bool, optional


        Returns
        -------
        flat_particles : list of tuples or ndarray of dtype `FLAT_PARTICLE_T`


        Examples
        --------
        This is a recursive function, with defaults defined for calling simply for
        the typical use case of flattening an entire I3MCTree and producing a
        numpy.ndarray with the results. .. ::

            flat_particles = extract_flat_mctree(frame["I3MCTree"])

        """
        if flat_particles is None:
            flat_particles = []

        if max_level < 0 or level <= max_level:
            if parent:
                daughters = mctree.get_daughters(parent)
            else:
                level = 0
                parent_idx = -1
                daughters = mctree.get_primaries()

            if daughters:
                # Record index before we started appending
                idx0 = len(flat_particles)

                # First append all daughters found
                for daughter in daughters:
                    info_tup, _ = self.extract_attrs(
                        daughter, dt.I3PARTICLE_T, to_numpy=False
                    )
                    flat_particles.append((level, parent_idx, info_tup))

                # Now recurse, appending any granddaughters (daughters to these
                # daughters) at the end
                for daughter_idx, daughter in enumerate(daughters, start=idx0):
                    self.extract_flat_mctree(
                        mctree=mctree,
                        parent=daughter,
                        parent_idx=daughter_idx,
                        level=level + 1,
                        max_level=max_level,
                        flat_particles=flat_particles,
                        to_numpy=False,
                    )

        if to_numpy:
            return np.array(flat_particles, dtype=dt.FLAT_PARTICLE_T)

        return flat_particles, dt.FLAT_PARTICLE_T

    def extract_flat_pulse_series(self, obj, frame=None, to_numpy=True):
        """Flatten a pulse series into a 1D array of ((<OMKEY_T>), <PULSE_T>)

        Parameters
        ----------
        obj : dataclasses.I3RecoPUlseSeries{,Map,MapMask,MapUnion}
        frame : iectray.I3Frame, required if obj is {...Mask, ...Union}
        to_numpy : bool, optional

        Returns
        -------
        flat_pulses : shape-(N-pulses) numpy.ndarray of dtype FLAT_PULSE_T

        """
        if isinstance(
            obj,
            (
                self.dataclasses.I3RecoPulseSeriesMapMask,
                self.dataclasses.I3RecoPulseSeriesMapUnion,
            ),
        ):
            if frame is None:
                frame = self.frame
            obj = obj.apply(frame)

        flat_pulses = []
        for omkey, pulses in obj.items():
            omkey = (omkey.string, omkey.om, omkey.pmt)
            for pulse in pulses:
                info_tup, _ = self.extract_attrs(
                    pulse, dtype=dt.PULSE_T, to_numpy=False
                )
                flat_pulses.append((omkey, info_tup))

        if to_numpy:
            return np.array(flat_pulses, dtype=dt.FLAT_PULSE_T)

        return flat_pulses, dt.FLAT_PULSE_T

    def extract_singleton_seq_to_scalar(self, seq, to_numpy=True):
        """Extract a sole object from a sequence and treat it as a scalar.
        E.g., I3VectorI3Particle that, by construction, contains just one
        particle


        Parameters
        ----------
        seq : sequence
        to_numpy : bool, optional


        Returns
        -------
        obj

        """
        assert len(seq) == 1
        return self.extract_object(seq[0], to_numpy=to_numpy)

    def extract_attrs(self, obj, dtype, to_numpy=True):
        """Extract attributes of an object (and optionally, recursively, attributes
        of those attributes, etc.) into a numpy.ndarray based on the specification
        provided by `dtype`.


        Parameters
        ----------
        obj
        dtype : numpy.dtype
        to_numpy : bool, optional


        Returns
        -------
        vals : tuple or shape-(1,) numpy.ndarray of dtype `dtype`

        """
        vals = []
        if isinstance(dtype, np.dtype):
            descr = dtype.descr
        elif isinstance(dtype, Sequence):
            descr = dtype
        else:
            raise TypeError("{}".format(dtype))

        for name, subdtype in descr:
            val = getattr(obj, name)
            if isinstance(subdtype, (str, np.dtype)):
                vals.append(val)
            elif isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    out = self.extract_attrs(val, subdtype, to_numpy=False)
                assert out is not None, "{}: {} {}".format(name, subdtype, val)
                info_tup, _ = out
                vals.append(info_tup)
            else:
                raise TypeError("{}".format(subdtype))

        # Numpy converts tuples correctly; lists are interpreted differently
        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=dtype)[0]

        return vals, dtype

    def extract_mapscalarattrs(self, mapping, subdtype=None, to_numpy=True):
        """Convert a mapping (containing string keys and scalar-typed values)
        to a single-element Numpy array from the values of `mapping`, using
        keys defined by `subdtype.names`.

        Use this function if you already know the `subdtype` you want to end up
        with. Use `retro.utils.misc.dict2struct` directly if you do not know
        the dtype(s) of the mapping's values ahead of time.


        Parameters
        ----------
        mapping : mapping from strings to scalars

        dtype : numpy.dtype
            If scalar dtype, convert via `utils.dict2struct`. If structured
            dtype, convert keys specified by the struct field names and values
            are converted according to the corresponding type.


        Returns
        -------
        array : shape-(1,) numpy.ndarray of dtype `dtype`


        See Also
        --------
        dict2struct
            Convert from a mapping to a numpy.ndarray, dynamically building `dtype`
            as you go (i.e., this is not known a priori)

        """
        keys = mapping.keys()
        if not isinstance(mapping, OrderedDict):
            keys.sort()

        out_vals = []
        out_dtype = []

        if subdtype is None:  # infer subdtype from values in mapping
            for key in keys:
                val = mapping[key]
                info_tup, subdtype = self.extract_object(val, to_numpy=False)
                out_vals.append(info_tup)
                out_dtype.append((key, subdtype))
        else:  # scalar subdtype
            for key in keys:
                out_vals.append(mapping[key])
                out_dtype.append((key, subdtype))

        out_vals = tuple(out_vals)

        if to_numpy:
            return np.array([out_vals], dtype=out_dtype)[0]

        return out_vals, out_dtype

    def extract_getters(self, obj, dtype, fmt="Get{}", to_numpy=True):
        """Convert an object whose data has to be extracted via methods that
        behave like getters (e.g., .`xyz = get_xyz()`).


        Parameters
        ----------
        obj
        dtype
        fmt : str
        to_numpy : bool, optional


        Examples
        --------
        To get all of the values of an I3PortiaEvent: .. ::

            extract_getters(frame["PoleEHESummaryPulseInfo"], dtype=dt.I3PORTIAEVENT_T, fmt="Get{}")

        """
        vals = []
        for name, subdtype in dtype.descr:
            getter_attr_name = fmt.format(name)
            getter_func = getattr(obj, getter_attr_name)
            val = getter_func()
            if not isinstance(subdtype, str) and isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    raise ValueError(
                        "Failed to convert name {} val {} type {}".format(
                            name, val, type(val)
                        )
                    )
                val, _ = out
            # if isinstance(val, self.icetray.OMKey):
            #    val = self.extract_attrs(val, dtype=dt.OMKEY_T, to_numpy=False)
            vals.append(val)

        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=dtype)[0]

        return vals, dtype

    def extract_seq_of_same_type(self, seq, to_numpy=True):
        """Convert a sequence of objects, all of the same type, to a numpy array of
        that type.

        Parameters
        ----------
        seq : seq of N objects all of same type
        to_numpy : bool, optional

        Returns
        -------
        out_seq : list of N tuples or shape-(N,) numpy.ndarray of `dtype`

        """
        assert len(seq) > 0

        # Convert first object in sequence to get dtype
        val0 = seq[0]
        val0_tup, val0_dtype = self.extract_object(val0, to_numpy=False)
        data_tups = [val0_tup]

        # Convert any remaining objects
        for obj in seq[1:]:
            data_tups.append(self.extract_object(obj, to_numpy=False)[0])

        if to_numpy:
            return np.array(data_tups, dtype=val0_dtype)

        return data_tups, val0_dtype

    def extract_i3domcalibration(self, obj, to_numpy=True):
        """Extract the information from an I3DOMCalibration frame object"""
        vals = []
        for name, subdtype in dt.I3DOMCALIBRATION_T.descr:
            val = getattr(obj, name)
            if name == "dom_cal_version":
                if val == "unknown":
                    val = (-1, -1, -1)
                else:
                    val = tuple(int(x) for x in val.split("."))
            elif isinstance(subdtype, (str, np.dtype)):
                pass
            elif isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    raise ValueError(
                        "{} {} {} {}".format(name, subdtype, val, type(val))
                    )
                val, _ = out
            else:
                raise TypeError(str(subdtype))
            vals.append(val)

        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=dt.I3DOMCALIBRATION_T)[0]

        return vals, dt.I3DOMCALIBRATION_T


def test_IC_DATA_RUN_RE():
    """Unit tests for OSCNEXT_I3_FNAME_RE."""
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


def test_OSCNEXT_I3_FNAME_RE():

    """Unit tests for OSCNEXT_I3_FNAME_RE."""
    # pylint: disable=line-too-long
    test_cases = [
        (
            "oscNext_data_IC86.12_level5_v01.04_pass2_Run00120028_Subrun00000000.i3.zst",
            {
                "basename": "oscNext_data_IC86.12_level5_v01.04_pass2_Run00120028_Subrun00000000",
                "compr_exts": ".zst",
                "kind": "data",
                "level": "5",
                "pass": "2",
                "levelver": "01.04",
                #'misc': '',
                "run": "00120028",
                "season": "12",
                "subrun": "00000000",
            },
        ),
        (
            "oscNext_data_IC86.18_level7_addvars_v01.04_pass2_Run00132761.i3.zst",
            {
                "basename": "oscNext_data_IC86.18_level7_addvars_v01.04_pass2_Run00132761",
                "compr_exts": ".zst",
                "kind": "data",
                "level": "7",
                "pass": "2",
                "levelver": "01.04",
                #'misc': 'addvars',
                "run": "00132761",
                "season": "18",
                "subrun": None,
            },
        ),
        (
            "oscNext_genie_level5_v01.01_pass2.120000.000216.i3.zst",
            {
                "basename": "oscNext_genie_level5_v01.01_pass2.120000.000216",
                "compr_exts": ".zst",
                "kind": "genie",
                "level": "5",
                "pass": "2",
                "levelver": "01.01",
                #'misc': '',
                "run": "120000",
                "season": None,
                "subrun": "000216",
            },
        ),
        (
            "oscNext_noise_level7_v01.03_pass2.888003.000000.i3.zst",
            {
                "basename": "oscNext_noise_level7_v01.03_pass2.888003.000000",
                "compr_exts": ".zst",
                "kind": "noise",
                "level": "7",
                "pass": "2",
                "levelver": "01.03",
                #'misc': '',
                "run": "888003",
                "season": None,
                "subrun": "000000",
            },
        ),
        (
            "oscNext_muongun_level5_v01.04_pass2.139011.000000.i3.zst",
            {
                "basename": "oscNext_muongun_level5_v01.04_pass2.139011.000000",
                "compr_exts": ".zst",
                "kind": "muongun",
                "level": "5",
                "pass": "2",
                "levelver": "01.04",
                #'misc': '',
                "run": "139011",
                "season": None,
                "subrun": "000000",
            },
        ),
        (
            "oscNext_corsika_level5_v01.03_pass2.20788.000000.i3.zst",
            {
                "basename": "oscNext_corsika_level5_v01.03_pass2.20788.000000",
                "compr_exts": ".zst",
                "kind": "corsika",
                "level": "5",
                "pass": "2",
                "levelver": "01.03",
                #'misc': '',
                "run": "20788",
                "season": None,
                "subrun": "000000",
            },
        ),
    ]

    for test_input, expected_output in test_cases:
        try:
            match = OSCNEXT_I3_FNAME_RE.match(test_input)
            groupdict = match.groupdict()

            ref_keys = set(expected_output.keys())
            actual_keys = set(groupdict.keys())
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
                actual_val = groupdict[key]
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