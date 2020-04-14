#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Interface for various functions within the i3cols module
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

__all__ = ["main"]


from argparse import ArgumentParser
from inspect import getargspec
from multiprocessing import cpu_count
import sys

import numpy as np
from six import string_types

from i3cols import cols, extract, phys, utils


def main(description=__doc__):
    """Command line interface"""
    # pylint: disable=line-too-long

    parser = ArgumentParser(description=description)
    subparsers = parser.add_subparsers()

    all_sp = []

    # Extract files individually has unique kwargs

    subparser = subparsers.add_parser("extract_files_separately")
    all_sp.append(subparser)
    subparser.set_defaults(func=extract.extract_files_separately)
    subparser.add_argument("--outdir", required=True)
    subparser.add_argument("--index-and-concatenate", action="store_true")
    subparser.add_argument("--index-name", default="sourcefile")
    subparser.add_argument(
        "--category-xform", default=None, choices=["subrun"],
    )
    subparser.add_argument("--gcd", default=None)
    subparser.add_argument("--sub-event-stream", nargs="+", default=None)
    subparser.add_argument("--keys", nargs="+", default=None)
    subparser.add_argument("--overwrite", action="store_true")
    subparser.add_argument("--compress", action="store_true")
    subparser.add_argument("--tempdir", default=None)
    subparser.add_argument("--keep-tempfiles-on-fail", action="store_true")
    subparser.add_argument("--procs", type=int, default=cpu_count())

    # Extract i3 files as if they are one large i3 file (interspersing GCD
    # files if specified and as appropriate)

    subparser = subparsers.add_parser("extract_files_as_one")
    all_sp.append(subparser)
    subparser.set_defaults(func=extract.extract_files_as_one)
    subparser.add_argument("--outdir", required=True)
    subparser.add_argument("--gcd", default=None)
    subparser.add_argument("--sub-event-stream", nargs="+", default=None)
    subparser.add_argument("--keys", nargs="+", default=None)
    subparser.add_argument("--overwrite", action="store_true")
    subparser.add_argument("--compress", action="store_true")
    subparser.add_argument("--procs", type=int, default=cpu_count())

    # Extract i3 data season files (look for "Run" directories, and "subrun"
    # .i3 files within each run directory).

    subparser = subparsers.add_parser("extract_season")
    all_sp.append(subparser)
    subparser.set_defaults(func=extract.extract_season)
    subparser.add_argument("--outdir", required=True)
    subparser.add_argument("--index-and-concatenate", action="store_true")
    subparser.add_argument("--gcd", default=None)
    subparser.add_argument("--sub-event-stream", nargs="+", default=None)
    subparser.add_argument("--keys", nargs="+", default=None)
    subparser.add_argument("--overwrite", action="store_true")
    subparser.add_argument("--compress", action="store_true")
    subparser.add_argument("--tempdir", default=None)
    subparser.add_argument("--keep-tempfiles-on-fail", action="store_true")
    subparser.add_argument("--procs", type=int, default=cpu_count())

    # Combine runs is unique

    parser_combine_runs = subparsers.add_parser("combine_runs")
    all_sp.append(parser_combine_runs)
    parser_combine_runs.add_argument("--outdir", required=True)
    parser_combine_runs.add_argument("--keys", nargs="+", default=None)
    parser_combine_runs.add_argument("--no-mmap", action="store_true")
    parser_combine_runs.set_defaults(func=extract.combine_runs)

    # Compress / decompress are similar

    parser_compress = subparsers.add_parser("compress")
    all_sp.append(parser_compress)
    parser_compress.set_defaults(func=cols.compress)

    parser_decompress = subparsers.add_parser("decompress")
    all_sp.append(parser_decompress)
    parser_decompress.set_defaults(func=cols.decompress)

    for subparser in [parser_compress, parser_decompress]:
        subparser.add_argument("--keys", nargs="+", default=None)
        subparser.add_argument("-k", "--keep", action="store_true")
        subparser.add_argument("-r", "--recurse", action="store_true")
        subparser.add_argument("--procs", type=int, default=cpu_count())

    # Simple functions that add columns derived from existing columns (post-proc)

    def func_wrapper(func):
        def runit(paths, outdir, outdtype, overwrite):
            if outdir is None:
                outdir = path
            func(paths, outdir=outdir, outdtype=outdtype, overwrite=overwrite)

        return runit

    for funcname in [
        "fit_genie_rw_syst",
        "calc_genie_weighted_aeff",
        "calc_normed_weights",
    ]:
        subparser = subparsers.add_parser(funcname)
        all_sp.append(subparser)
        subparser.set_defaults(func=func_wrapper(getattr(phys, funcname)))
        subparser.add_argument("--outdtype", required=False)
        subparser.add_argument("--outdir", required=False)
        subparser.add_argument("--overwrite", action="store_true")

    # More complicated add-column post-processing functions

    def compute_coszen_wrapper(
        path, key_path, outdir, outkey=None, outdtype=None, overwrite=False
    ):
        if outdir is None:
            outdir = path

        if isinstance(outdtype, string_types):
            if hasattr(np, outdtype):
                outdtype = getattr(np, outdtype)
            else:
                outdtype = np.dtype(outdtype)

        phys.compute_coszen(
            path=path,
            key_path=key_path,
            outkey=outkey,
            outdtype=outdtype,
            outdir=outdir,
            overwrite=overwrite,
        )

    parser_compute_coszen = subparsers.add_parser("compute_coszen")
    all_sp.append(parser_compute_coszen)
    parser_compute_coszen.set_defaults(func=compute_coszen_wrapper)
    parser_compute_coszen.add_argument("--key-path", nargs="+", required=True)
    parser_compute_coszen.add_argument("--outdtype", required=False)
    parser_compute_coszen.add_argument("--outdir", required=False)
    parser_compute_coszen.add_argument("--overwrite", action="store_true")

    # Add args common to all

    for subparser in all_sp:
        subparser.add_argument(
            "-0",
            action="store_true",
            help="split stdin by null chars (e.g. find -print0 | thisprog)",
        )
        args = getargspec(subparser.get_default("func")).args
        path_argname = None
        if "paths" in args:
            subparser.add_argument("paths", nargs="*", default=sys.stdin)
            subparser.add_argument("--sort", action="store_true")
        elif "path" in args:
            subparser.add_argument("path", nargs="*", default=sys.stdin)

    # Parse command line

    kwargs = vars(parser.parse_args())

    # Translate command line arguments that don't match functiona arguments

    if (
        "keys" in kwargs
        and kwargs["keys"] is not None
        and set(["", "all"]).intersection(kwargs["keys"])
    ):
        kwargs["keys"] = None

    if "no_mmap" in kwargs:
        kwargs["mmap"] = not kwargs.pop("no_mmap")

    category_xform = kwargs.pop("category_xform", None)
    if category_xform is not None:
        if category_xform == "subrun":
            category_xform = extract.i3_subrun_category_xform
            index_name = kwargs.pop("index_name", None)
            if index_name is not None and index_name != "subrun":
                print(
                    "WARNING: renaming `index_name` to 'subrun' in"
                    " accordance with `category_xform`"
                )
            kwargs["index_name"] = "subrun"
        else:
            raise ValueError(category_xform)
        kwargs["category_xform"] = category_xform

    # Run appropriate function

    func = kwargs.pop("func", None)
    if func is None:
        parser.parse_args(["--help"])
        return

    for path_argname in ["path", "paths"]:
        if path_argname not in kwargs:
            continue
        path = kwargs.pop(path_argname)
        splitter = "\0" if kwargs.pop("0") else "\n"
        if hasattr(path, "read"):  # stdin
            if path.isatty():
                parser.parse_args(["--help"])
                return

            # "if p" removes trailing empty string
            path = [p for p in path.read().split(splitter) if p]

            if len(path) == 1:
                path = path[0]

        # extract a single str from the list
        if len(path) == 1:
            path = path[0]

        kwargs[path_argname] = path
        break

    if "paths" in kwargs and "sort" in kwargs:
        if kwargs.pop("sort"):
            kwargs["paths"] = sorted(kwargs["paths"], key=utils.nsort_key_func)

    func(**kwargs)


if __name__ == "__main__":
    main()
