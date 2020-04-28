#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Core functions for creating and working with columnar arrays in memory and on
disk
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
    "CATEGORY_INDEX_POSTFIX",
    "LEGAL_ARRAY_NAMES",
    "load",
    "save_item",
    "check_outdir_and_keys",
    "filter_keys_from_existing",
    "get_valid_key_func",
    "find_array_paths",
    "load_contained_paths",
    "compress",
    "decompress",
    "construct_arrays",
    "concatenate_and_index_cols",
]


from collections import defaultdict, OrderedDict

try:
    from collections.abc import (
        Mapping,
        MutableMapping,
        MutableSequence,
        Sequence,
    )
except ImportError:
    from collections import Mapping, MutableMapping, MutableSequence, Sequence
import copy
import enum
import fnmatch
import functools
import multiprocessing
import operator
import os
import re
import shutil
import sys
import time

import numpy as np
from six import string_types

from i3cols import dtypes, regexes, utils


# TODO: optional npz compression/decompression of key dirs built-in to functions
# TODO: optional versioning on write and read
#   If version specified, write to that or read from that. On read, if version
#   specified but only "bare" key, load the bare key (assume it is valid across
#   versions).
# TODO: arrays with significantly fewer than one entry per event: use optional
#   masking array or reference to another column's "valid" array?
#   * Per-scalar (i.e., per-event) that live in directory like
#     category__index.npy files makes most sense
#   * Per-key alongside "data", "index", and "valid" arrays within key dir?
# TODO: allow metadata (e.g., JSON?) in each column directory; maybe also
#   within the directory holding the columns? How to handle upon concatenation
#   and/or splitting events up?
#   * Can create a list of all metadata that has been combined together (never
#     remove); then a metadata_index which essentially records "metadata entry
#     0 applies to indices 0, 32, 45-180, ...". This also allows new metadata
#     entries to be created appended that apply to the whole set of events,
#     etc. Just a bit of bookkeeping that needs to be done when splitting and
#     combining
#   How to handle a JSON file with .npz compression of the directory?
#   * Just store the file as bytes (np.uint8)


CATEGORY_INDEX_POSTFIX = "__categ_index"
"""Category scalar index files are named {category}{CATEGORY_INDEX_POSTFIX}"""

LEGAL_ARRAY_NAMES = ("data", "index", "valid")
"""Array names produced / read as data containers within "items" (the items
extracted from an I3 file)"""

ARRAY_FNAMES = {n: "{}.npy".format(n) for n in LEGAL_ARRAY_NAMES}
"""Basic npy filenames associated with each legal (recoginzed) array"""


class MatchSpecialCase(enum.IntEnum):
    """Special cases for `keys` and `exclude_keys` matching keys"""

    UNKNOWN = -1
    MATCH_NOTHING = 0
    MATCH_EVERYTHING = 1


def load(path, keys=None, exclude_keys=None, mmap=True):
    """Find and load arrays within `path`.

    Parameters
    ----------
    path : str
    keys : str, iterable thereof, or None; optional
    mmap : bool, optional

    Returns
    -------
    arrays
    category_indexes

    """
    arrays, category_indexes = find_array_paths(
        path=path, keys=keys, exclude_keys=exclude_keys
    )
    load_contained_paths(arrays, inplace=True, mmap=mmap)
    load_contained_paths(category_indexes, inplace=True, mmap=mmap)
    return arrays, category_indexes


def save_item(path, key, data, valid=None, index=None, overwrite=False):
    """Save a single item (1 data array and possibly one valid arrya and one
    index array) to disk.

    Parameters
    ----------
    path : str
    key : str
    data : numpy ndarray
    valid : numpy ndarray or None, optional if all `data` are valid
    index : numpy ndarray or None, only specify if `data` is scalar
    overwrite : bool, optional

    """
    path = utils.expand(path)
    if os.path.exists(path):
        assert os.path.isdir(path)
        if not overwrite:
            existing_cols, _ = find_array_paths(path, keys=key)
            if existing_cols:
                raise IOError(
                    'Key {} already exists at path(s) "{}"'.format(
                        key, existing_cols[key]
                    )
                )

    outdirpath = os.path.join(path, key)

    outfpaths_saved = []
    parent_outdir_created = utils.mkdir(outdirpath)
    try:
        outfpaths_arrays = []
        for name, array in [("data", data), ("valid", valid), ("index", index)]:
            if array is None:
                continue
            outfpath = os.path.join(outdirpath, name + ".npy")
            outfpaths_arrays.append((outfpath, array))
        for outfpath, array in outfpaths_arrays:
            np.save(outfpath, array)
            outfpaths_saved.append(outfpath)
    except:
        if parent_outdir_created is not None:
            try:
                shutil.rmtree(parent_outdir_created)
            except Exception as err:
                print(err)
        else:
            for outfpath in outfpaths_saved:
                try:
                    os.remove(outfpath)
                except Exception as err:
                    print(err)
        raise


def check_outdir_and_keys(outdir=None, outkeys=None, overwrite=False):
    """Validate `outdir` and determine if files would be overwritten"""
    if outdir is not None:
        outdir = utils.expand(outdir)
        if os.path.exists(outdir):
            assert os.path.isdir(outdir)

    if not overwrite and outdir is not None:
        outarrays, _ = find_array_paths(outdir, keys=outkeys)
        existing_keys = sorted(set(outkeys).intersection(outarrays.keys()))
        if existing_keys:
            raise IOError(
                'keys {} already exist in outdir "{}"'.format(existing_keys, outdir)
            )

    return outdir


def filter_keys_from_existing(outdir, keys=None, exclude_keys=None):
    """Augment `exclude_keys` with existing arrays in an output directory to
    avoid overwriting those arrays.

    Parameters
    ----------
    outdir : str
    keys, exclude_keys

    Returns
    -------
    exclude_keys
    nothing_to_do : bool

    """
    outdir = utils.expand(outdir)
    if os.path.isdir(outdir):
        existing_cols, _ = find_array_paths(
            path=outdir, keys=keys, exclude_keys=exclude_keys
        )

        new_exclude_keys = list(existing_cols.keys())

        if new_exclude_keys:
            if exclude_keys is None:
                exclude_keys = new_exclude_keys
            elif isinstance(exclude_keys, string_types) or callable(exclude_keys):
                exclude_keys = [exclude_keys] + new_exclude_keys
            else:
                exclude_keys += new_exclude_keys

    valid_info, invalid_info = expand_keys(keys=keys, exclude_keys=exclude_keys)

    nothing_to_do = (
        valid_info["match_special"] == MatchSpecialCase.MATCH_NOTHING
        or invalid_info["match_special"] == MatchSpecialCase.MATCH_EVERYTHING
    )

    if keys is None or valid_info["match_special"] == MatchSpecialCase.MATCH_EVERYTHING:
        updated_keys = None
    else:
        updated_keys = sorted(
            functools.reduce(
                operator.add,
                (list(s) for k, s in valid_info.items() if k != "match_special"),
            )
        )

    if invalid_info["match_special"] == MatchSpecialCase.MATCH_NOTHING:
        updated_exclude_keys = None
    else:
        updated_exclude_keys = sorted(
            functools.reduce(
                operator.add,
                (list(s) for k, s in invalid_info.items() if k != "match_special"),
            )
        )

    # print("updated_keys:\n{}\n".format(updated_keys))
    # print("updated_exclude_keys:\n{}\n".format(updated_exclude_keys))

    return nothing_to_do, updated_keys, updated_exclude_keys


def _is_key_match_protofunc(key, key_funcs, named_keys, rgxs):
    """If any match criterion is met, return True; if NO match
    criterion is met, return False."""
    for func in key_funcs:
        if func(key):
            return True
    if key in named_keys:
        return True
    for rgx in rgxs:
        if rgx.match(key):
            return True
    return False


def _generate_key_match_function(info):
    return functools.partial(
        _is_key_match_protofunc,
        key_funcs=tuple(f for f in info["key_funcs"]),
        named_keys=copy.deepcopy(info["named_keys"]),
        rgxs=tuple(
            re.compile(fnmatch.translate(pattern), flags=re.IGNORECASE)
            for pattern in sorted(info["glob_patterns"])
        ),
    )


def expand_keys(keys, exclude_keys):
    """
    Parameters
    ----------
    keys
    exclude_keys

    Returns
    -------
    valid_info, invalid_info : dict

    """
    infos = {}
    for is_valid_logic, keys_spec in [(True, keys), (False, exclude_keys)]:
        info = infos[is_valid_logic] = dict(
            match_special=MatchSpecialCase.UNKNOWN,
            named_keys=set(),
            glob_patterns=set(),
            key_funcs=set(),
        )

        set_names = [name for name in info if name != "match_special"]

        if keys_spec is None:
            if is_valid_logic:
                info["match_special"] = MatchSpecialCase.MATCH_EVERYTHING
            else:
                info["match_special"] = MatchSpecialCase.MATCH_NOTHING
            for set_name in set_names:
                info[set_name].clear()
            continue

        # Make isolated string or callable into a singleton list so we
        # can handle the same way as everything else
        if isinstance(keys_spec, string_types) or callable(keys_spec):
            keys_spec = [keys_spec]

        # Make any iterable into a (mutable) set
        keys_spec = set(keys_spec)

        files_read = set()

        while keys_spec:
            key = keys_spec.pop()

            if callable(key):
                info["key_funcs"].add(key)

            # All other cases: key must be a string

            elif os.path.isfile(utils.expand(key)):
                abspath = utils.expand(key)
                if abspath in files_read:
                    continue
                files_read.add(abspath)
                with open(abspath, "r") as fh:
                    txt = fh.read()
                keys_spec.update(k.strip() for k in txt.strip().split("\n") if k)

            elif key.strip() == "*":
                info["match_special"] = MatchSpecialCase.MATCH_EVERYTHING
                for set_name in set_names:
                    info[set_name].clear()
                break

            elif "*" in key or "?" in key or ("[" in key and "]" in key):
                info["glob_patterns"].add(key)

            else:
                info["named_keys"].add(key)

    valid_info = infos.pop(True)
    invalid_info = infos.pop(False)

    # Simplify

    for info in [valid_info, invalid_info]:
        if info["match_special"] != MatchSpecialCase.MATCH_EVERYTHING and all(
            len(info[set_name]) == 0 for set_name in set_names
        ):
            info["match_special"] = MatchSpecialCase.MATCH_NOTHING

    # Filter valid_info based on invalid_info...

    # If everything is invalid: clear out valid_info & make it match nothing
    if invalid_info["match_special"] == MatchSpecialCase.MATCH_EVERYTHING:
        valid_info["match_special"] = MatchSpecialCase.MATCH_NOTHING
        for set_name in set_names:
            valid_info[set_name].clear()

    # If some things are invalid: clear out valid_info keys that meet an
    # invalid_info criterion
    elif invalid_info["match_special"] != MatchSpecialCase.MATCH_NOTHING:
        # Modifying named_keys, so wrap set being iterated over in list()
        for key in list(valid_info["named_keys"]):
            if (
                key in invalid_info["named_keys"]
                or any(
                    re.match(fnmatch.translate(pattern), key, flags=re.IGNORECASE)
                    for pattern in invalid_info["glob_patterns"]
                )
                or any(func(key) for func in invalid_info["key_funcs"])
            ):
                valid_info["named_keys"].remove(key)

        # Check if we emptied the valid_info
        if valid_info["match_special"] != MatchSpecialCase.MATCH_EVERYTHING and all(
            len(info[set_name]) == 0 for set_name in set_names
        ):
            info["match_special"] = MatchSpecialCase.MATCH_NOTHING
            for set_name in set_names:
                info[set_name].clear()

    return valid_info, invalid_info


def get_valid_key_func(keys=None, exclude_keys=None):
    r"""Turn `keys` and `exclude_keys` into a function to validate key names.

    Parameters
    ----------
    keys, exclude_keys : str or callable, iterable thereof, or None; optional
        If `keys` is:
            * A string which represents a path to a file, it
              is interpreted as a keys file. This file must contain
              "\n"-separated key names; do NOT use single- or double-quotation
              marks in the file to surround key names.
            * A string or iterable thereof of key names, including optional
              glob patterns. See Python's `fnmatch` module for more
              description, but legal Glob patterns are "?" for any single
              character, "*" for any number of any character, and "[abf]" to
              match specific characters, in this case one of "a", "b", or "f".
            * A callable is assumed to take a single string (key name) as its
              argument and return either True or False.
            * `keys=None` matches all keys while `exclude_keys=None` matches
              _no_ keys

    Returns
    -------
    is_key_valid : callable
        Callable that returns True or False given a akey. Call via .. ::

            is_valid = is_key_valid(key)

    """
    valid_info, invalid_info = expand_keys(keys, exclude_keys)

    # No keys are valid cases

    if (
        valid_info["match_special"] == MatchSpecialCase.MATCH_NOTHING
        or invalid_info["match_special"] == MatchSpecialCase.MATCH_EVERYTHING
    ):
        return lambda key: False

    # Define non-trivial functions

    if valid_info["match_special"] == MatchSpecialCase.UNKNOWN:
        valid_func = _generate_key_match_function(valid_info)
    else:  # valid_info["match_special"] == MatchSpecialCase.MATCH_EVERYTHING
        # No point in specifying a function that always returns True
        valid_func = None

    if invalid_info["match_special"] == MatchSpecialCase.UNKNOWN:
        invalid_func = _generate_key_match_function(invalid_info)
    else:  # invalid_info["match_special"] == MatchSpecialCase.MATCH_NOTHING
        # No point in specifying: `not func(key)` is always True
        invalid_func = None

    # Define final `is_key_valid` function, composing the above as necessary

    if valid_func is not None and invalid_func is not None:
        is_key_valid = lambda key: valid_func(key) and not invalid_func(key)

    elif valid_func is not None and invalid_func is None:
        is_key_valid = valid_func

    elif valid_func is None and invalid_func is not None:
        is_key_valid = lambda key: not invalid_func(key)

    else:  # is_valid is None and is_invalid is None:
        is_key_valid = lambda key: True

    return is_key_valid


def find_array_paths(path, keys=None, exclude_keys=None):
    """Find arrays and category indexes.

    Parameters
    ----------
    path : str
        Path to a

            * single column directory (e.g. I3EventHeader, which contains files
                data.npy, etc.)
            * single column .npz archive (e.g. I3EventHeader.npz, which
                contains arrays "data", etc.)
            * column directory containing one or more of the above

    keys, exclude_keys : str or callable, iterable thereof, or None; optional
        See `get_valid_key_func` for acceptable values and what they translate
        to. For a key to be extracted, it must meet the criteria of `keys` AND
        NOT match the criteria of `exclude_keys`, with the special values that
        `keys=None` matches _everything_ while `exclude_keys=None` matches
        _nothing_. Keys within a frame not meething these criteria are simply
        ignored.

    Returns
    -------
    arrays
    category_indexes

    """
    orig_path = path
    path = utils.expand(orig_path)

    is_key_valid = get_valid_key_func(keys=keys, exclude_keys=exclude_keys)

    arrays = OrderedDict()
    category_indexes = OrderedDict()
    unrecognized = []

    cat_idx_plus_ext = CATEGORY_INDEX_POSTFIX + ".npy"

    if os.path.isfile(path):
        match = regexes.KEY_NAME_RE.match(os.path.basename(path))
        if not match:
            raise ValueError(
                'Path unrecognizable as a i3cols key name: "{}"'.format(orig_path)
            )

        key_name = match.groupdict()["key_name"]
        if not is_key_valid(key_name):
            return arrays, category_indexes

        # Code below handles directories, so pass the directory this file sits
        # in but specify to only select this key. This allows re-use of the
        # code that finds category indexes

        path = os.path.dirname(path)
        is_key_valid = get_valid_key_func(keys=key_name)

    dir_listing = sorted(os.listdir(path), key=utils.nsort_key_func)

    for name in dir_listing:
        subpath = os.path.join(path, name)

        if os.path.isfile(subpath):
            if name.endswith(cat_idx_plus_ext):
                category = name[: -len(cat_idx_plus_ext)]
                category_indexes[category] = subpath
                continue

            if name.endswith(".npz"):
                key = name[:-4]
                if not is_key_valid(key):
                    continue

                is_array = False
                npz = np.load(subpath)
                try:
                    contents = set(npz.keys())

                    if "data" in contents:
                        is_array = True
                    else:
                        unrecognized.append(subpath)
                        continue

                    for array_name in LEGAL_ARRAY_NAMES:
                        if array_name in contents:
                            contents.remove(array_name)
                finally:
                    npz.close()

                if is_array:
                    arrays[key] = subpath
                    for array_name in contents:
                        unrecognized.append(subpath + "/" + array_name)
                else:
                    unrecognized.append(subpath)

                continue

        if not os.path.isdir(subpath):
            unrecognized.append(subpath)
            continue

        array_d = OrderedDict()
        contents = set(os.listdir(subpath))
        for array_name in LEGAL_ARRAY_NAMES:
            fname = array_name + ".npy"
            if fname in contents:
                array_d[array_name] = os.path.join(subpath, fname)
                contents.remove(fname)

        for subname in sorted(contents):
            unrecognized.append(os.path.join(subpath, subname))

        if array_d and is_key_valid(name):
            arrays[name] = array_d

    if not arrays and not category_indexes:
        print(
            'WARNING: no arrays or category indexes found; is path "{}" a key'
            " directory?".format(path)
        )
    elif unrecognized:
        print("WARNING: Unrecognized paths ignored: {}".format(unrecognized))

    return arrays, category_indexes


def load_contained_paths(obj, inplace=False, mmap=False):
    """If `obj` or any sub-element of `obj` is a (string) path to a file, load
    the file and replace that element with the object loaded from the file.

    Unhandled containers or objects (whether that is `obj` itself or child
    objects within `obj`) are simply returned, unmodified.

    Parameters
    ----------
    obj

    inplace: bool, optional
        Only valid if all objects and sub-objects that are containers are
        mutable

    mmap : bool, optional
        Load numpy ".npy" files memory mapped. Only applies to ".npy" for now,
        `mmap` is ignored for all other files.

    Returns
    -------
    obj
        If the input `obj` is a container type, the `obj` returned is the same
        object (if inplace=True) or a new object (if inplace=False) but with
        string paths replaced by the contents of the files they refer to. Note
        that all mappings (and .npz file paths) are converted to OrderedDict,
        where a conversion is necessary.

    """
    # Record kwargs beyond `obj` for recursively calling
    my_kwargs = dict(inplace=inplace, mmap=mmap)

    if isinstance(obj, string_types):  # numpy strings evaluate False
        if os.path.isfile(obj):
            _, ext = os.path.splitext(obj)
            if ext == ".npy":
                obj = np.load(obj, mmap_mode="r" if mmap else None)
            elif ext == ".npz":
                npz = np.load(obj)
                try:
                    obj = OrderedDict(npz.items())
                finally:
                    npz.close()
            # TODO : other file types?

    elif isinstance(obj, Mapping):
        if inplace:
            assert isinstance(obj, MutableMapping)
            out_d = obj
        else:
            out_d = OrderedDict()
        for key in obj.keys():
            try:
                out_d[key] = load_contained_paths(obj[key], **my_kwargs)
            except:
                print("key '{}' failed to load".format(key))
                raise
        obj = out_d

    elif isinstance(obj, Sequence):  # numpy ndarrays evaluate False
        if inplace:
            assert isinstance(obj, MutableSequence)
            for i, val in enumerate(obj):
                try:
                    obj[i] = load_contained_paths(val, **my_kwargs)
                except:
                    print("index={}, obj {} failed to load".format(i, val))
                    raise
        else:
            obj = type(obj)(load_contained_paths(val, **my_kwargs) for val in obj)

    return obj


def compress(paths, keys=None, exclude_keys=None, recurse=True, keep=False, procs=1):
    """Compress any key directories found in any path in `paths` (including
    path itself, if it is a key directory) using Numpy's `savez_compressed` to
    produce "{key}.npz" files.

    Parameters
    ----------
    paths

    keys : str, iterable thereof, or None; optional
        Only look to compress key directories by these names

    recurse : bool, optional

    keep : bool, optional
        Keep the original key directory even after successfully compressing it

    procs : int >= 1, optional

    """
    if isinstance(paths, string_types):
        paths = [paths]
    paths = [utils.expand(p) for p in paths]
    for path in paths:
        assert os.path.isdir(path), path

    is_key_valid = get_valid_key_func(keys=keys, exclude_keys=exclude_keys)

    pool = None
    if procs > 1:
        pool = multiprocessing.Pool(procs)

    try:
        for path in paths:
            for dirpath, dirs, files in os.walk(path):
                if recurse:
                    dirs.sort(key=utils.nsort_key_func)
                else:
                    del dirs[:]
                if (
                    "data.npy" not in files
                    or len(dirs) > 0  # subdirectories
                    or set(files).difference(ARRAY_FNAMES.values())  # extra files
                    or not is_key_valid(dirpath)
                ):
                    continue

                args = (dirpath, copy.deepcopy(files), keep)
                if procs == 1:
                    _compress(*args)
                else:
                    pool.apply_async(_compress, args)

                del dirs[:]
                del files[:]

    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception as err:
                print(err)


def _compress(dirpath, files, keep):
    t0 = time.time()

    # sys.stdout.write('compressing "{}"...'.format(dirpath))
    # sys.stdout.flush()

    array_d = OrderedDict()
    for array_name, array_fname in ARRAY_FNAMES.items():
        if array_fname in files:
            array_d[array_name] = np.load(os.path.join(dirpath, array_fname))
    if not array_d:
        return

    archivepath = os.path.join(dirpath.rstrip("/") + ".npz")
    np.savez_compressed(archivepath, **array_d)
    if not keep:
        try:
            shutil.rmtree(dirpath)
        except Exception as err:
            print("WARNING: unable to remove dir {}".format(dirpath))
            print(err)

    sys.stdout.write('"{}"  {:.3f} s\n'.format(dirpath, time.time() - t0))
    sys.stdout.flush()


def decompress(paths, keys=None, exclude_keys=None, recurse=True, keep=False, procs=1):
    """Decompress any key archive files (end in .npz and contain "data" and
    possibly other arrays) found within `path` (including `path`, if it is a
    key archive).

    Parameters
    ----------
    paths

    keys : str, iterable thereof, or None; optional
        Only look to decompress keys by these names (ignore others)

    recurse : bool
        If `path` is a directory, whether to recurse into subdirectories to
        look for archive files to decompress

    keep : bool, optional
        Keep the original key directory even after successfully compressing it

    procs : int >= 1, optional

    """
    if isinstance(paths, string_types):
        paths = [paths]
    paths = [utils.expand(p) for p in paths]

    is_key_valid = get_valid_key_func(keys=keys, exclude_keys=exclude_keys)

    pool = None
    if procs > 1:
        pool = multiprocessing.Pool(procs)

    try:
        for path in paths:
            if (
                os.path.isfile(path)
                and path.endswith(".npz")
                and is_key_valid(path[:-4])
            ):
                kwargs = dict(
                    dirpath=os.path.dirname(path),
                    filename=os.path.basename(path),
                    keep=keep,
                )
                if procs == 1:
                    _decompress(**kwargs)
                else:
                    pool.apply_async(_decompress, tuple(), kwargs)
                continue

            # else: is directory

            for dirpath, dirnames, filenames in os.walk(path):
                if recurse:
                    dirnames.sort(key=utils.nsort_key_func)
                else:
                    del dirnames[:]

                for filename in filenames:
                    if filename.endswith(".npz"):
                        if not is_key_valid(filename[:-4]):
                            continue
                        kwargs = dict(dirpath=dirpath, filename=filename, keep=keep)
                        if procs == 1:
                            _decompress(**kwargs)
                        else:
                            pool.apply_async(_decompress, tuple(), kwargs)

    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception as err:
                print(err)


def _decompress(dirpath, filename, keep):
    """Decompress legal columnar arrays in a "<dirpath>/basename.npz" archive
    to "<dirpath>/basename/<array_name>.npy". If successful and not `keep`,
    remove the original archive file.

    Parameters
    ----------
    dirpath : str
        Path up to but not including `filename`

    filename : str
        E.g., I3EventHeader.npz

    keep : bool
        Keep original archive file after successfully decompressing

    Returns
    -------
    is_columnar_archive : bool

    """
    t0 = time.time()

    filepath = os.path.join(dirpath, filename)

    key = filename[:-4]
    keydirpath = os.path.join(dirpath, key)
    array_d = OrderedDict()

    npz = np.load(filepath)
    try:
        contents = set(npz.keys())
        if len(contents.difference(LEGAL_ARRAY_NAMES)) > 0:
            return False

        for array_name in LEGAL_ARRAY_NAMES:
            if array_name in npz:
                array_d[array_name] = npz[array_name]

    finally:
        npz.close()

    # sys.stdout.write('decompressing "{}"...'.format(filepath))
    # sys.stdout.flush()

    subfilepaths_created = []
    parent_dir_created = utils.mkdir(keydirpath)
    try:
        for array_name, array in array_d.items():
            arraypath = os.path.join(keydirpath, array_name + ".npy")
            np.save(arraypath, array)
            subfilepaths_created.append(arraypath)
    except:
        if parent_dir_created is not None:
            try:
                shutil.rmtree(parent_dir_created)
            except Exception as err:
                print(err)
        else:
            for subfilepath in subfilepaths_created:
                try:
                    os.remove(subfilepath)
                except Exception as err:
                    print(err)
        raise

    if not keep:
        os.remove(filepath)

    sys.stdout.write('"{}"  {:.3f} s\n'.format(filepath, time.time() - t0))
    sys.stdout.flush()

    return True


def construct_arrays(data, delete_while_filling=False, outdir=None):
    """Construct arrays to collect same-key scalars / vectors across frames

    Parameters
    ----------
    data : dict or iterable thereof
    delete_while_filling : bool
    outdir : str

    Returns
    -------
    arrays : dict

    """
    if isinstance(data, Mapping):
        data = [data]
    data = list(data)

    parent_outdir_created = None
    if isinstance(outdir, string_types):
        outdir = utils.expand(outdir)
        parent_outdir_created = utils.mkdir(outdir)

    try:

        # Get type and size info

        scalar_dtypes = {}
        vector_dtypes = {}

        num_frames = len(data)
        for frame_d in data:
            # Must get all vector values for all frames to get both dtype and
            # total length, but only need to get a scalar value once to get its
            # dtype
            for key in set(frame_d.keys()).difference(scalar_dtypes.keys()):
                val = frame_d[key]
                # if val is None:
                #    continue
                dtype = val.dtype

                if np.isscalar(val):
                    scalar_dtypes[key] = dtype
                else:
                    if key not in vector_dtypes:
                        vector_dtypes[key] = [0, dtype]  # length, type
                    vector_dtypes[key][0] += len(val)

        # Construct empty arrays

        scalar_arrays = {}
        vector_arrays = {}

        scalar_arrays_paths = {}
        vector_arrays_paths = {}

        for key, dtype in scalar_dtypes.items():
            # Until we know we need one (i.e., when an event is missing this
            # `key`), the "valid" mask array is omitted
            if outdir is not None:
                dpath = os.path.join(outdir, key)
                utils.mkdir(dpath)
                data_array_path = os.path.join(dpath, "data.npy")
                scalar_arrays_paths[key] = dict(data=data_array_path)
                data_array = np.lib.format.open_memmap(
                    data_array_path, mode="w+", shape=(num_frames,), dtype=dtype
                )
            else:
                data_array = np.empty(shape=(num_frames,), dtype=dtype)
            scalar_arrays[key] = dict(data=data_array)

        # `vector_arrays` contains "data" and "index" arrays.
        # `index` has the same number of entries as the scalar arrays,
        # and each entry points into the corresponding `data` array to
        # determine which vector data correspond to this scalar datum

        for key, (length, dtype) in vector_dtypes.items():
            if outdir is not None:
                dpath = os.path.join(outdir, key)
                utils.mkdir(dpath)
                data_array_path = os.path.join(dpath, "data.npy")
                index_array_path = os.path.join(dpath, "index.npy")
                vector_arrays_paths[key] = dict(
                    data=data_array_path, index=index_array_path
                )
                data_array = np.lib.format.open_memmap(
                    data_array_path, mode="w+", shape=(length,), dtype=dtype
                )
                index_array = np.lib.format.open_memmap(
                    index_array_path,
                    mode="w+",
                    shape=(num_frames,),
                    dtype=dtypes.START_STOP_T,
                )
            else:
                data_array = np.empty(shape=(length,), dtype=dtype)
                index_array = np.empty(shape=(num_frames,), dtype=dtypes.START_STOP_T)
            vector_arrays[key] = dict(data=data_array, index=index_array)

        # Fill the arrays

        for frame_idx, frame_d in enumerate(data):
            for key, array_d in scalar_arrays.items():
                val = frame_d.get(key, None)
                if val is None:
                    if "valid" not in array_d:
                        if outdir is not None:
                            dpath = os.path.join(outdir, key)
                            valid_array_path = os.path.join(dpath, "valid.npy")
                            scalar_arrays_paths[key]["valid"] = valid_array_path
                            valid_array = np.lib.format.open_memmap(
                                valid_array_path,
                                mode="w+",
                                shape=(num_frames,),
                                dtype=np.bool8,
                            )
                            valid_array[:] = True
                        else:
                            valid_array = np.ones(shape=(num_frames,), dtype=np.bool8)
                        array_d["valid"] = valid_array
                    array_d["valid"][frame_idx] = False
                else:
                    array_d["data"][frame_idx] = val
                    if delete_while_filling:
                        del frame_d[key]

            for key, array_d in vector_arrays.items():
                index = array_d["index"]
                if frame_idx == 0:
                    prev_stop = 0
                else:
                    prev_stop = index[frame_idx - 1]["stop"]

                start = int(prev_stop)

                val = frame_d.get(key, None)
                if val is None:
                    index[frame_idx] = (start, start)
                else:
                    length = len(val)
                    stop = start + length
                    index[frame_idx] = (start, stop)
                    array_d["data"][start:stop] = val[:]
                    if delete_while_filling:
                        del index, frame_d[key]

        arrays = scalar_arrays
        arrays.update(vector_arrays)

        arrays_paths = scalar_arrays_paths
        arrays_paths.update(vector_arrays_paths)

    except Exception:
        if parent_outdir_created is not None:
            try:
                shutil.rmtree(parent_outdir_created)
            except Exception as err:
                print(err)
        raise

    if outdir is None:
        return arrays

    del arrays, scalar_arrays, vector_arrays

    return arrays_paths


def concatenate(
    paths, outdir, keys=None, exclude_keys=None, index_name=None, category_xform=None,
):
    """
    Parameters
    ----------
    paths : str or iterable thereof
        Look within each directory defined in `paths` to find column stores
        (either .npz files or directories containing "data.npy", etc. files) to
        concatenate. If no column stores are found there, look in any
        subdirectories found within each path in `paths`.

    outdir : str
    keys :
    exclude_keys :
    index_name : str or None, optional
    category_xform : callable or None, optional

    """
    # TODO: Allow paths to individual column store(s) (.npz file or directory
    #   containing "data.npy", etc.) to be concatenated? (requires copying over
    #   category indexes, etc., from same director(ies)...)

    # Normalize args

    if isinstance(paths, string_types):
        paths = [paths]
    paths = [utils.expand(path) for path in paths]

    index_name, category_xform, category_is_global = utils.handle_category_index_args(
        index_name=index_name, category_xform=category_xform
    )

    # Find paths to concatenate

    to_concat = OrderedDict()
    for path in paths:
        arrays, category_indexes = find_array_paths(path, keys, exclude_keys)
        if arrays or category_indexes:
            to_concat[path] = (arrays, category_indexes)
            continue

        subdirs = (os.path.join(path, d) for d in os.listdir(path))
        subdirs = [d for d in subdirs if os.path.isdir(d)]

        for subdir in subdirs:
            arrays, category_indexes = find_array_paths(subdir, keys, exclude_keys)
            if arrays or category_indexes:
                to_concat[subdir] = (arrays, category_indexes)

    # Create a properly-sorted list of (category, path) tuples

    if category_is_global:
        all_paths = list(to_concat.keys())
        # Note that sorting is performed by (category, path)
        category_path_map = sorted(
            zip(category_xform(all_paths), all_paths), key=utils.nsort_key_func
        )
    else:
        category_path_map = sorted(
            ((category_xform(path), path) for path in to_concat.keys()),
            key=utils.nsort_key_func,
        )

    # Make `category_array_map` (include arrays, exclude category_indexes)

    category_array_map = OrderedDict(
        (category, to_concat[path][0]) for category, path in category_path_map
    )

    # Perform the concatenation

    concatenate_and_index_cols(
        category_array_map=category_array_map, index_name=index_name, outdir=outdir
    )


def concatenate_and_index_cols(
    category_array_map,
    existing_category_indexes=None,
    index_name=None,
    category_dtype=None,
    outdir=None,
    mmap=True,
):
    """A given scalar array might or might not be present in each tup

    Parameters
    ----------
    category_array_map : OrderedDict
        Keys are the categories (e.g., run number or subrun number), values
        are array dicts and/or paths to Numpy .npz files containing these.
        An array dict must contain the key "data"; if vector data, it has to
        have key "index"; and, optionally, it has key "valid". Values are either Numpy
        arrays, or a string path to the array on disk (to be loaded via np.load).

    existing_category_indexes : mapping of mappings or None, optional

    index_name : str, optional
        Name of the index; what describes the categories being indexed? This is
        used both to formulate the structured dtype used for the index and to
        formulate the file name it is saved to (if `outdir` is specified).

    category_dtype : numpy.dtype or None, optional
        If None, use the type Numpy infers, found via .. ::

            category_dtype = np.array(list(category_array_map.keys())).dtype

    outdir : str or None, optional
        If specified, category index and arrays are written to disk within this
        directory

    Returns
    -------
    category_indexes : dict
        Minimally contains key `index_name` with value a
        shape-(num_categories,) numpy ndarray of custom dtype .. ::

            [(index_name, category_dtype), ("index", dtypes.START_STOP_T)]

    arrays : dict of dicts containing arrays

    """
    if existing_category_indexes:
        raise NotImplementedError(
            "concatenating existing_category_indexes not implemented"
        )

    if index_name is None:
        index_name = "category"

    parent_outdir_created = None
    if outdir is not None:
        outdir = utils.expand(outdir)
        parent_outdir_created = utils.mkdir(outdir)

    try:

        load_contained_paths_kw = dict(mmap=mmap, inplace=not mmap)

        # Datatype of each data array (same key must have same dtype regardless
        # of which category)
        key_dtypes = OrderedDict()

        # All scalar data arrays and vector index arrays in one category must have
        # same length as one another; record this length for each category
        category_scalar_array_lens = OrderedDict()

        # scalar data, scalar valid, and vector index arrays will have this length
        total_scalar_len = 0

        # vector data has different total length for each item; make a counter
        # for these using defaultdict
        total_vector_lens = defaultdict(int)

        # Record any keys that, for any category, already have a valid array
        # created, as these keys will require valid arrays to be created and
        # filled
        keys_with_valid_arrays = set()

        # Record keys that contain vector data
        vector_keys = set()

        # Get and validate metadata about arrays

        for n, (category, array_dicts) in enumerate(category_array_map.items()):
            scalar_array_len = None
            for key, array_d in array_dicts.items():
                array_d = load_contained_paths(array_d, **load_contained_paths_kw)

                data = array_d["data"]
                valid = array_d.get("valid", None)
                index = array_d.get("index", None)

                is_scalar = index is None

                if scalar_array_len is None:
                    if is_scalar:
                        scalar_array_len = len(data)
                    else:
                        scalar_array_len = len(index)
                elif is_scalar and len(data) != scalar_array_len:
                    raise ValueError(
                        "category={}, key={}, ref len={}, this len={}".format(
                            category, key, scalar_array_len, len(data)
                        )
                    )

                if valid is not None:
                    keys_with_valid_arrays.add(key)

                    if len(valid) != scalar_array_len:
                        raise ValueError(
                            "category={}, key={}, ref len={}, this len={}".format(
                                category, key, scalar_array_len, len(valid)
                            )
                        )

                if index is not None:
                    vector_keys.add(key)
                    total_vector_lens[key] += len(data)

                    if len(index) != scalar_array_len:
                        raise ValueError(
                            "category={}, key={}, ref len={}, this len={}".format(
                                category, key, scalar_array_len, len(index)
                            )
                        )

                dtype = data.dtype
                existing_dtype = key_dtypes.get(key, None)
                if existing_dtype is None:
                    key_dtypes[key] = dtype
                elif dtype != existing_dtype:
                    raise TypeError(
                        "category={}, key={}, dtype={}, existing_dtype={}".format(
                            category, key, dtype, existing_dtype
                        )
                    )

            if scalar_array_len is None:
                scalar_array_len = 0

            category_scalar_array_lens[category] = scalar_array_len
            total_scalar_len += scalar_array_len
            print(
                "category {}, {}={}: scalar array len={},"
                " total scalar array len={}".format(
                    n, index_name, category, scalar_array_len, total_scalar_len
                )
            )

        # Create the index

        # Use simple numpy array for now for ease of working with dtypes, ease and
        # consistency in saving to disk; this can be expaned to a dict for easy
        # key/value acces or numba.typed.Dict for direct use in Numba

        categories = np.array(list(category_array_map.keys()), dtype=category_dtype)
        category_dtype = categories.dtype
        category_index_dtype = np.dtype(
            [(index_name, category_dtype), ("index", dtypes.START_STOP_T)]
        )
        if outdir is not None:
            category_index = np.lib.format.open_memmap(
                os.path.join(outdir, index_name + CATEGORY_INDEX_POSTFIX + ".npy"),
                mode="w+",
                shape=(len(categories),),
                dtype=category_index_dtype,
            )
        else:
            category_index = np.empty(
                shape=(len(categories),), dtype=category_index_dtype
            )

        # Populate the category index

        start = 0
        for i, (category, array_len) in enumerate(
            zip(categories, category_scalar_array_lens.values())
        ):
            stop = start + array_len
            value = np.array([(start, stop)], dtype=dtypes.START_STOP_T)[0]
            category_index[i] = (category, value)
            start = stop

        # Record keys that are missing in one or more categories

        all_keys = set(key_dtypes.keys())
        keys_with_missing_data = set()
        for category, array_dicts in category_array_map.items():
            keys_with_missing_data.update(all_keys.difference(array_dicts.keys()))

        # Create and populate `data` arrays and any necessary `valid` arrays

        # N.b. missing vector arrays DO require valid array so that the resulting
        # "index" array (which spans all categories) has the same number of
        # elements as scalar arrays

        keys_requiring_valid_array = set.union(
            keys_with_missing_data, keys_with_valid_arrays
        )

        concatenated_arrays = OrderedDict()
        for key, dtype in key_dtypes.items():
            if key in vector_keys:
                data_len = total_vector_lens[key]
            else:
                data_len = total_scalar_len

            # Create big data array

            if outdir is not None:
                dpath = os.path.join(outdir, key)
                utils.mkdir(dpath)
                data = np.lib.format.open_memmap(
                    os.path.join(dpath, "data.npy"),
                    mode="w+",
                    shape=(data_len,),
                    dtype=dtype,
                )
            else:
                data = np.empty(shape=(data_len,), dtype=dtype)

            # Create big valid array if needed

            valid = None
            if key in keys_requiring_valid_array:
                if outdir is not None:
                    dpath = os.path.join(outdir, key)
                    utils.mkdir(dpath)
                    valid = np.lib.format.open_memmap(
                        os.path.join(dpath, "valid.npy"),
                        mode="w+",
                        shape=(total_scalar_len,),
                        dtype=np.bool8,
                    )
                else:
                    valid = np.empty(shape=(total_scalar_len,), dtype=np.bool8)

            # Create big index array if vector data

            index = None
            if key in vector_keys:
                if outdir is not None:
                    dpath = os.path.join(outdir, key)
                    utils.mkdir(dpath)
                    index = np.lib.format.open_memmap(
                        os.path.join(dpath, "index.npy"),
                        mode="w+",
                        shape=(total_scalar_len,),
                        dtype=dtypes.START_STOP_T,
                    )
                else:
                    index = np.empty(shape=(total_scalar_len,), dtype=np.bool8)

            # Fill chunks of the big arrays from each category

            vector_start = vector_stop = 0
            for category, array_dicts in category_array_map.items():
                scalar_start, scalar_stop = category_index[
                    category_index[index_name] == category
                ][0]["index"]

                key_arrays = array_dicts.get(key, None)
                if key_arrays is None:
                    valid[scalar_start:scalar_stop] = False
                    continue

                key_arrays = load_contained_paths(key_arrays, **load_contained_paths_kw)
                data_ = key_arrays["data"]
                if key not in vector_keys:  # scalar data
                    data[scalar_start:scalar_stop] = data_
                else:  # vector data
                    # N.b.: copy values to a new array in memory, necessary if
                    # mmaped file AND necessary because we don't want to modify
                    # original array
                    index_ = np.copy(key_arrays["index"])

                    vector_stop = vector_start + len(data_)

                    if vector_start != 0:
                        index_["start"] += vector_start
                        index_["stop"] += vector_start
                    index[scalar_start:scalar_stop] = index_
                    del index_

                    data[vector_start:vector_stop] = data_

                    vector_start = vector_stop
                del data_

                valid_ = key_arrays.get("valid", None)
                if valid_ is not None:
                    valid[scalar_start:scalar_stop] = valid_
                    del valid_
                elif valid is not None:
                    valid[scalar_start:scalar_stop] = True

            concatenated_arrays[key] = OrderedDict()
            concatenated_arrays[key]["data"] = data
            if index is not None:
                concatenated_arrays[key]["index"] = index
            if valid is not None:
                concatenated_arrays[key]["valid"] = valid

        category_indexes = OrderedDict()
        category_indexes[index_name] = category_index
        # TODO: put concatenated existing_category_indexes here, too

    except Exception:
        if parent_outdir_created is not None:
            try:
                shutil.rmtree(parent_outdir_created)
            except Exception as err:
                print(err)
        raise

    return category_indexes, concatenated_arrays
