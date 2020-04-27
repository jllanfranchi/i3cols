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
    "find_gcd_for_data_file",
    "extract_files_separately",
    "extract_files_as_one",
    "extract_season",
    "combine_runs_or_subruns",
]

from collections import OrderedDict

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from contextlib import suppress
from multiprocessing import Pool
import os
import re
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from six import string_types

from i3cols import cols, regexes, utils
from i3cols.i3_to_numpy_converter import run_icetray_converter

try:
    from processing.samples.oscNext.verification.general_mc_data_harvest_and_plot import (
        ALL_OSCNEXT_VARIABLES,
    )

    OSCNEXTKEYS = [k.split(".")[0] for k in ALL_OSCNEXT_VARIABLES.keys()]
except ImportError:
    OSCNEXTKEYS = []

# TODO: communicate a "quit NOW!" message to worker threads
# TODO: use logging module
# TODO: profile ... record time(s) to extract key name and times per type
# TODO: add compress tasks to existing pool that is handling extraction


def find_gcd_for_data_file(datafilepath, gcd_dir, recurse=True):
    """Given a data file's name, attempt to extract which run it came from, and
    find the GCD for that run in `gcd_dir`.

    Parameters
    ----------
    datafilepath : str
    gcd_dir : str
    recurse : bool

    Returns
    -------
    gcd_file_path

    Raises
    ------
    ValueError
        If a GCD file cannot be found

    """
    gcd_dir = utils.expand(gcd_dir)
    assert os.path.isdir(gcd_dir)

    match = regexes.I3_RUN_RE.search(os.path.basename(datafilepath))
    if not match:
        raise ValueError(
            'Unable to extract a run number from datafilepath="{}"'.format(datafilepath)
        )

    groupdict = match.groupdict()
    # NB: str(int(...)) strips leading 0's and gives "0" if run is all 0's
    run = str(int(groupdict["run"]))
    run_gcd_re = re.compile(
        r"run0*{}[^0-9].*gcd.*\.i3.*".format(run), flags=re.IGNORECASE
    )

    # Shortcut: if i3 data file is inside GCD dir, check the same dir
    #   as the i3 file first (this is true for at least oscNext i3 files)
    gcd_dirs = gcd_dir.split(os.path.sep)
    datafiledir = os.path.dirname(datafilepath)
    file_dirs = utils.expand(datafiledir).split(os.path.sep)
    if len(gcd_dirs) <= len(file_dirs) and all(
        gd == fd for gd, fd in zip(gcd_dirs, file_dirs)
    ):
        for test_filename in os.listdir(datafiledir):
            test_filepath = os.path.join(datafiledir, test_filename)
            if os.path.isfile(test_filepath) and run_gcd_re.search(test_filename):
                return test_filepath

    for dirpath, dirs, files in os.walk(gcd_dir, followlinks=True):
        if recurse:
            dirs.sort(key=utils.nsort_key_func)
        else:
            del dirs[:]
        files.sort(key=utils.nsort_key_func)
        for filename in files:
            if run_gcd_re.search(filename):
                return os.path.join(dirpath, filename)

    raise IOError(
        'Could not find GCD in dir "{}" for data run file "{}"'.format(
            gcd_dir, datafilepath
        )
    )


def extract_files_separately(
    paths,
    outdir,
    concatenate_and_index,
    index_name=None,
    category_xform=None,
    gcd=None,
    sub_event_stream=None,
    keys=None,
    exclude_keys=None,
    overwrite=False,
    compress=False,
    tempdir=None,
    keep_tempfiles_on_fail=False,
    procs=1,
):
    """Exctract i3 files separately (if `gcd` is specified, an appropriate GCD
    file will also be read before each given i3 data file -- e.g., if pulses
    are to be extracted) into i3cols (directories + numpy) format in `outdir`.

    Parameters
    ----------
    paths : str or iterable thereof
        Paths to i3 files to be extracted; note the order of files provided is
        the order they are extracted. Use e.g. `i3cols.utils.nsort_key_func`
        from Python or `ls -v`, `sort -V`, etc. from the command line to
        achieve sensible sorting

    outdir : str
        Resulting i3cols column directories (or .npz files if `compress` is
        True) are placed in `outdir` if `concatenate_and_index` is True or
        within subdirectories inside of `outdir` if `concatenate_and_index` is
        False

    concatenate_and_index : bool
        Whether to concatenate the columns extracted from each individual i3
        file. If so, a category index is created for the concatenated columns
        to indicate which values came from which file (see `index_name` and
        `category_xform` args)

    index_name : None or str, optional
        By default, category index (if `concatenate_and_index` is True) is named
        "sourcefile" (so the full filename is "sourcefile__category_index.npy")
        due to the default for catgory naming convention (see
        `category_xform`)

    category_xform : callable or None, optional
        Create a name for the categories in a category index (if
        `concatenate_and_index` is True) or the name of the subdirectories
        created within `outdir` (if `concatenate_and_index` is False). Called
        via `category_xform(full_path)`, i.e. with the
        user/variable-expanded absolute path (but not with symlinks resolved)
        of each file which has events extracted from it. If None is specified
        (the default), categories are named by the the uncommon trailing parts
        of each source i3 file's path (with i3 & compression extensions
        removed).

    gcd : str, iterable thereof, or None; optional
        If `gcd` is None, no GCD file is read prior to extracting each file in
        `paths`. If `gcd` is a path to a single GCD file, this file is read in
        before each file in `paths` is extracted; in pseudo-code: extract(gcd,
        path); extract(gcd, path); etc. If `gcd` is a path to a directory, the
        directory is searched for a GCD file matching a "Run" specification in
        the source filename.

    sub_event_stream : str, iterable thereof, or None; optional
        Only extract frames with the specified sub event streams; if None,
        extract all sub event streams.

    keys, exclude_keys : str or callable, iterable thereof, or None; optional
        See `i3cols.cols.get_valid_key_func` for acceptable values and what
        they translate to. For a key to be extracted, it must meet the
        criteria of `keys` AND NOT match the criteria of `exclude_keys`, with
        the special values that `keys=None` matches _everything_ while
        `exclude_keys=None` matches _nothing_. Keys within a frame not meething
        these criteria are simply ignored.

    overwrite : bool, optional
        Currently this has to be True, but in the future `overwrite=False`
        logic should be worked out to error out before much/any extraction
        occurs to avoid overwriting files and performing redundant extractions

    compress : bool, optional
        After the extraction is complete, compress the column directories into
        `.npz` files (and remove the original directories)?

    tempdir : str, optional
        If `concatenate_and_index` is True, the individually extracted arrays
        are placed in a sub-directory within `tempdir` before
        indexing/concatenating and placing the final results in `outdir`.
        `tempdir` is unused if `concatenate_and_index` is False.

    keep_tempfiles_on_fail : bool, optional
        If `concatenate_and_index` is True and an error occurs, tempfiles that
        are created within `tempdir` will be kept. (Otherwise, these are
        automatically deleted, if at all possible.)

    procs : int >= 1, optional
        Number of processes to use for extracting files in parallel.

    Notes
    -----
    You MUST use this function (as opposed to `extract_files_as_one`) if

        * it is necessary to know from which file extracted events came after
            being extracted (e.g., normalizing Monte Carlo requires knowing the
            number of source files)

    It is not inefficient to use this function if

        * the i3 files are "large": I.e., the time to extract a single i3 file
            is significantly larger than the time to execute the
            `run_icetray_converter` function (this instantiates
            I3ToNumpyConverter and creates an icetray to process the (gcd+) i3
            data file

    If neither of the above hold (e.g., for all subruns within a single data
    run -- these are both small and we do not need to retain the subrun), it is
    recommended to simply call `run_icetray_converter` directly, or use the
    higher-level `extract_season` function which already does for each run
    while also attempting to extract all runs in parallel.

    The output column array directories are either:

        1. Written directly within `outdir`; a "{index_name}__categ_index.npy"
            array is created in `outdir` to index into the concatenated array.
            Category names are derived via `category_xform`.

        2. Written to subdirectories within `outdir`, each subdirectory named
            via `category_xform`

    See also
    --------
    extract_files_as_one
    extract_season

    """
    if isinstance(paths, string_types):
        paths = [paths]
    full_paths = [utils.expand(p) for p in sorted(paths, key=utils.nsort_key_func)]

    simplified_paths = utils.simplify_paths(full_paths)

    # Duplicate (ignoring compression extension(s)) paths are illegal
    if len(set(simplified_paths)) < len(simplified_paths):
        raise ValueError("Duplicated paths detected: {}".format(paths))

    if category_xform is None:
        categories = simplified_paths
    else:
        categories = [category_xform(full_path) for full_path in full_paths]

    num_unique_categories = len(set(categories))
    if num_unique_categories != len(categories):
        raise ValueError(
            "Duplicated categories detected: {}".format(list(zip(paths, categories)))
        )

    gcd_is_dir = False
    if gcd is not None:
        gcd = utils.expand(gcd)
        gcd_is_dir = os.path.isdir(gcd)
        if not gcd_is_dir:
            assert os.path.isfile(gcd)

    if isinstance(sub_event_stream, string_types):
        sub_event_stream = [sub_event_stream]

    if tempdir is not None:
        if not concatenate_and_index:
            print("NOTE: `tempdir` is not used if `concatenate_and_index` is False")
        else:
            tempdir = utils.expand(tempdir)

    orig_keys = keys
    orig_exclude_keys = exclude_keys

    categ_keys = keys
    categ_exclude_keys = exclude_keys

    nothing_to_do = False

    if concatenate_and_index and not overwrite:
        nothing_to_do, categ_keys, categ_exclude_keys = cols.filter_keys_from_existing(
            outdir=outdir, keys=orig_keys, exclude_keys=orig_exclude_keys
        )
        if nothing_to_do:
            print("Nothing to extract")
            return

    procs = min(procs, len(full_paths))

    my_tempdir = None
    paths_to_compress = []
    results = []
    category_array_map = OrderedDict()

    pool = None
    if procs > 1:
        pool = Pool(procs)
    try:
        if concatenate_and_index:
            if tempdir is not None:
                utils.mkdir(tempdir)
            my_tempdir = mkdtemp(dir=tempdir)

        for full_path, category in zip(full_paths, categories):
            if isinstance(category, string_types):
                category_dirname = category
            elif isinstance(category, Iterable):
                category_dirname = os.path.join(*(str(x) for x in category))
            else:
                category_dirname = index_name + str(category)

            if concatenate_and_index:
                category_outdir = os.path.join(my_tempdir, category_dirname)
                category_array_map[category] = category_outdir
            else:
                category_outdir = os.path.join(outdir, category_dirname)
                if compress:
                    paths_to_compress.append(category_outdir)
                if not overwrite:
                    (
                        nothing_to_do,
                        categ_keys,
                        categ_exclude_keys,
                    ) = cols.filter_keys_from_existing(
                        outdir=category_outdir,
                        keys=orig_keys,
                        exclude_keys=orig_exclude_keys,
                    )
                    if nothing_to_do:
                        print(
                            'Nothing to extract for category {} / outdir "{}"'.format(
                                category, category_outdir
                            )
                        )
                        continue

            if gcd is None:
                extract_paths = [full_path]
            else:
                if gcd_is_dir:
                    gcd_file = find_gcd_for_data_file(
                        datafilepath=full_path, gcd_dir=gcd
                    )
                else:
                    gcd_file = gcd
                extract_paths = [gcd_file, full_path]

            kw = dict(
                paths=extract_paths,
                outdir=category_outdir,
                sub_event_stream=sub_event_stream,
                keys=categ_keys,
                exclude_keys=categ_exclude_keys,
            )
            if procs == 1:
                run_icetray_converter(**kw)
            else:
                results.append(pool.apply_async(run_icetray_converter, tuple(), kw))

        if pool is not None:
            pool.close()
            pool.join()

        for result in results:
            result.get()

        if concatenate_and_index:
            for category, category_outdir in list(category_array_map.items()):
                category_array_map[category], _ = cols.find_array_paths(category_outdir)
            cols.concatenate_and_index_cols(
                category_array_map=category_array_map,
                index_name=index_name,
                outdir=outdir,
            )
            if compress:
                paths_to_compress = outdir

    except:
        if my_tempdir is not None and os.path.isdir(my_tempdir):
            if keep_tempfiles_on_fail:
                print(
                    'Temp dir/files will NOT be removed; see "{}" to inspect'
                    " and manually remove".format(my_tempdir)
                )
            else:
                try:
                    rmtree(my_tempdir)
                except Exception as err:
                    print(err)
        raise

    else:
        if my_tempdir is not None and os.path.isdir(my_tempdir):
            try:
                rmtree(my_tempdir)
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
        cols.compress(
            paths=paths_to_compress,
            keys=orig_keys,
            exclude_keys=orig_exclude_keys,
            recurse=True,
            keep=False,
            procs=procs,
        )


def extract_files_as_one(
    paths,
    outdir,
    gcd=None,
    sub_event_stream=None,
    keys=None,
    exclude_keys=None,
    overwrite=True,
    compress=False,
    procs=1,
):
    """Exctract i3 files as if they are one. All information about file
    boundaries is lost (unless this is already encoded in the data being
    extracted, e.g. if I3EventHeader's Run or SubRun corresponds to the files).

    Parameters
    ----------
    paths : str or iterable thereof

    outdir : str
        Resulting i3cols column directories (or .npz files if `compress` is
        True) are placed in this directory (it is created, including any parent
        directories, if it does not already exist)

    gcd : str, iterable thereof, or None; optional
        If `gcd` is None, no GCD file is read prior to extracting each file in
        `paths`. If `gcd` is a path to a single GCD file, this file is read in
        before each file in `paths` is extracted; in pseudo-code: extract(gcd,
        path); extract(gcd, path); etc. If `gcd` is a path to a directory, the
        directory is searched for a GCD file matching a "Run" specification in
        the source filename.

    sub_event_stream : str, iterable thereof, or None; optional
        Only extract frames with the specified sub event streams; if None,
        extract all sub event streams.

    keys, exclude_keys : str or callable, iterable thereof, or None; optional
        See `i3cols.cols.get_valid_key_func` for acceptable values and what
        they translate to. For a key to be extracted, it must meet the
        criteria of `keys` AND NOT match the criteria of `exclude_keys`, with
        the special values that `keys=None` matches _everything_ while
        `exclude_keys=None` matches _nothing_. Keys within a frame not meething
        these criteria are simply ignored.

    overwrite : bool, optional
        Currently this has to be True, but in the future `overwrite=False`
        logic should be worked out to error out before much/any extraction
        occurs to avoid overwriting files and performing redundant extractions

    compress : bool, optional
        After the extraction is complete, compress the column directories into
        `.npz` files (and remove the original directories)?

    procs : int >= 1, optional
        Only used by the `i3cols.cols.compress` (if `compress` is True), as the
        extraction of multiple files as one cannot currently run in parallel.

    Notes
    -----
    It is recommended to use this function if both of the following are true:

        * It is not necessary to know from which file extracted events came after
            being extracted

        * The i3 files are "small": I.e., the time to extract a single i3 file
            is similar to or less than the time to execute the
            `run_icetray_converter` function (this instantiates
            I3ToNumpyConverter and creates an icetray to process the (gcd+) i3
            data file

    If one  of the above does not hold, it is recommended to call
    `extract_files_separately`.

    See also
    --------
    extract_files_separately
    extract_season

    """
    if isinstance(paths, string_types):
        paths = [paths]
    paths = [utils.expand(p) for p in paths]

    if gcd is not None:
        gcd = utils.expand(gcd)
        if os.path.isdir(gcd):
            # Find GCD file required for each data file, but only add to
            # `new_paths` when the GCD _changes_, thereby minimizing the number
            # of GCD files that need to be read during the extraction process
            previous_gcd_file_path = None
            new_paths = []
            for path in paths:
                gcd_file_path = find_gcd_for_data_file(path, gcd)
                if gcd_file_path != previous_gcd_file_path:
                    new_paths.append(gcd_file_path)
                    previous_gcd_file_path = gcd_file_path
                new_paths.append(path)
            paths = new_paths
        else:
            assert os.path.isfile(gcd)
            paths = paths.insert(0, gcd)

    if isinstance(sub_event_stream, string_types):
        sub_event_stream = [sub_event_stream]

    orig_keys = keys
    orig_exclude_keys = exclude_keys

    categ_keys = keys
    categ_exclude_keys = exclude_keys

    if not overwrite:
        nothing_to_do, categ_keys, categ_exclude_keys = cols.filter_keys_from_existing(
            outdir=outdir, keys=orig_keys, exclude_keys=orig_exclude_keys
        )
        if nothing_to_do:
            print("Nothing to extract")
            return

    run_icetray_converter(
        paths=paths,
        outdir=outdir,
        sub_event_stream=sub_event_stream,
        keys=categ_keys,
        exclude_keys=categ_exclude_keys,
    )

    if compress:
        cols.compress(
            paths=outdir,
            keys=orig_keys,
            exclude_keys=orig_exclude_keys,
            recurse=True,
            keep=False,
            procs=procs,
        )


def extract_season(
    path,
    outdir,
    concatenate_and_index,
    gcd=None,
    sub_event_stream=None,
    keys=None,
    exclude_keys=None,
    overwrite=False,
    compress=False,
    tempdir=None,
    keep_tempfiles_on_fail=False,
    procs=1,
):
    """E.g. .. ::

        data/level7_v01.04/IC86.14

    Parameters
    ----------
    path : str
        Path to the directory containing the season's run directories (and
        those should contain the run's subrun .i3 files)

    outdir : str
        Write extracted info to this dir. Column directories (or .npz files)
        are either written to "{outdir}/run{run}/" for each run if
        `concatenate_and_index` is False, or directly to "{outdir}/"  if
        `concatenate_and_index` is True. In the latter case,
        "{outdir}/run__category_index.npy" is written out as well.

    concatenate_and_index : bool
        Concatenate all the season's runs together into large columns and
        create a "run__category_index.npy" array to indicate which data in the
        large columns belongs to which run.

    gcd : str or None, optional

    keys, exclude_keys : str or callable, iterable thereof, or None; optional
        See `i3cols.cols.get_valid_key_func` for acceptable values and what
        they translate to. For a key to be extracted, it must meet the
        criteria of `keys` AND NOT match the criteria of `exclude_keys`, with
        the special values that `keys=None` matches _everything_ while
        `exclude_keys=None` matches _nothing_. Keys within a frame not meething
        these criteria are simply ignored.

    overwrite : bool, optional
    compress : bool, optional
    tempdir : str or None, optional
        Intermediate column arrays will be written to this directory.
    keep_tempfiles_on_fail : bool, optional
    procs : int >= 1, optional

    """
    path = utils.expand(path)
    assert os.path.isdir(path), str(path)
    outdir = utils.expand(outdir)
    if tempdir is not None:
        tempdir = utils.expand(tempdir)

    orig_keys = keys
    orig_exclude_keys = exclude_keys

    categ_keys = keys
    categ_exclude_keys = exclude_keys

    if concatenate_and_index and not overwrite:
        nothing_to_do, categ_keys, categ_exclude_keys = cols.filter_keys_from_existing(
            outdir=outdir, keys=orig_keys, exclude_keys=orig_exclude_keys
        )
        if nothing_to_do:
            print("Nothing to extract")
            return

    run_dirpaths = []
    for basepath in sorted(os.listdir(path), key=utils.nsort_key_func):
        run = None
        with suppress(ValueError):
            run = utils.i3_run_category_xform(basepath)
        if run is None:
            continue
        run_dirpaths.append((run, os.path.join(path, basepath)))
    # Sort ascending by numeric run number
    run_dirpaths.sort()

    procs = min(procs, len(run_dirpaths))

    index_name = "run"

    my_tempdir = None
    paths_to_compress = []
    results = []
    category_array_map = OrderedDict()

    pool = None
    if procs > 1:
        pool = Pool(procs)
    try:
        if tempdir is not None:
            utils.mkdir(tempdir)
        my_tempdir = mkdtemp(dir=tempdir)

        for run, run_dirpath in run_dirpaths:
            category = run

            # Find and organize subrun i3 files within the run directory

            subrun_paths = []
            for filename in os.listdir(run_dirpath):
                path = os.path.join(run_dirpath, filename)
                if not os.path.isfile(path):
                    continue
                match = regexes.I3_OSCNEXT_FNAME_RE.match(os.path.basename(path))
                if match:
                    groupdict = match.groupdict()
                    filename_run = np.uint32(int(groupdict["run"]))
                    if filename_run != run:
                        raise ValueError(
                            "run in file={} != run dir run={}".format(filename_run, run)
                        )
                    subrun = np.uint32(int(groupdict["subrun"]))
                    subrun_paths.append((subrun, path))
            all_subruns = [sr_p[0] for sr_p in subrun_paths]
            if len(set(all_subruns)) != len(subrun_paths):
                raise ValueError(
                    "Duplicate subruns found, will result in ambiguity. {}".format(
                        list(zip(subrun_paths, all_subruns))
                    )
                )
            # Sort ascending by numeric subrun number; just keep path
            subrun_paths = [sr_p[1] for sr_p in sorted(subrun_paths)]

            category_dirname = index_name + str(category)

            if concatenate_and_index:
                category_outdir = os.path.join(my_tempdir, category_dirname)
                category_array_map[category] = category_outdir
            else:
                category_outdir = os.path.join(outdir, category_dirname)
                if compress:
                    paths_to_compress.append(category_outdir)
                if not overwrite:
                    (
                        nothing_to_do,
                        categ_keys,
                        categ_exclude_keys,
                    ) = cols.filter_keys_from_existing(
                        outdir=category_outdir,
                        keys=orig_keys,
                        exclude_keys=orig_exclude_keys,
                    )
                    if nothing_to_do:
                        print(
                            'Nothing to extract for category {} / outdir "{}"'.format(
                                category, category_outdir
                            )
                        )
                        continue

            kw = dict(
                paths=subrun_paths,
                outdir=category_outdir,
                gcd=gcd,
                sub_event_stream=sub_event_stream,
                keys=categ_keys,
                exclude_keys=categ_exclude_keys,
                overwrite=True if concatenate_and_index else overwrite,
                compress=False,
                procs=procs,
            )
            if procs == 1:
                extract_files_as_one(**kw)
            else:
                results.append(pool.apply_async(extract_files_as_one, tuple(), kw))

        if pool is not None:
            pool.close()
            pool.join()

        for result in results:
            result.get()

        if concatenate_and_index:
            for category, category_outdir in list(category_array_map.items()):
                category_array_map[category], _ = cols.find_array_paths(category_outdir)
            cols.concatenate_and_index_cols(
                category_array_map=category_array_map,
                index_name=index_name,
                outdir=outdir,
            )
            if compress:
                paths_to_compress = outdir

    except:
        if my_tempdir is not None and os.path.isdir(my_tempdir):
            if keep_tempfiles_on_fail:
                print(
                    'Temp dir/files will NOT be removed; see "{}" to inspect'
                    " and manually remove".format(my_tempdir)
                )
            else:
                try:
                    rmtree(my_tempdir)
                except Exception as err:
                    print(err)
        raise

    else:
        if my_tempdir is not None and os.path.isdir(my_tempdir):
            try:
                rmtree(my_tempdir)
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
        cols.compress(
            paths=paths_to_compress,
            keys=orig_keys,
            exclude_keys=orig_exclude_keys,
            recurse=True,
            keep=False,
            procs=procs,
        )


# TODO: eliminate `combine_runs_or_subruns` function, replace with generic
#   `i3cols.cols.concatenate` function with args sufficient to handle combining
#   runs or subruns
def combine_runs_or_subruns(path, outdir, keys=None, mmap=True):
    """
    Parameters
    ----------
    path : str
        IC86.XX season directory or MC directory that contains
        already-extracted arrays

    outdir : str
        Store concatenated column directories or .npz files in this directory

    keys : str, iterable thereof, or None; optional
        Only preserver these keys. If None, preserve all keys found in all
        subpaths

    mmap : bool
        Note that if `mmap` is True, ``load_contained_paths`` will be called
        with `inplace=False` or else too many open files will result

    """
    path = utils.expand(path)
    assert os.path.isdir(path), str(path)
    outdir = utils.expand(outdir)

    is_key_valid = cols.get_valid_key_func(keys)

    run_dirs = []
    for subname in sorted(os.listdir(path), key=utils.nsort_key_func):
        subpath = os.path.join(path, subname)
        if not os.path.isdir(subpath):
            continue
        match = regexes.I3_RUN_DIR_RE.match(subname)
        if not match:
            continue
        groupdict = match.groupdict()
        run_str = groupdict["run"]
        run_int = np.uint32(int(run_str))
        run_dirs.append((run_int, subpath))
    # Ensure sorting by numerical run number
    run_dirs.sort()

    print("{} run dirs found".format(len(run_dirs)))

    array_map = OrderedDict()
    existing_category_indexes = OrderedDict()

    for run_int, run_dir in run_dirs:
        array_map[run_int], csi = cols.find_array_paths(run_dir, keys=is_key_valid)
        if csi:
            existing_category_indexes[run_int] = csi

    utils.mkdir(outdir)
    cols.concatenate_and_index_cols(
        category_array_map=array_map,
        existing_category_indexes=existing_category_indexes,
        index_name="run",
        category_dtype=np.uint32,  # see retro_types.I3EVENTHEADER_T
        outdir=outdir,
        mmap=mmap,
    )
