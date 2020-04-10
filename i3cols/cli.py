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
from functools import partial
from multiprocessing import cpu_count

import numpy as np
from six import string_types

from i3cols import cols, extract, phys


def main(description=__doc__):
    """Command line interface"""
    # pylint: disable=line-too-long

    parser = ArgumentParser(description=description)
    subparsers = parser.add_subparsers()

    # Extract season and run have similar arguments; can use generic `extract`
    # for these, too

    parser_extract = subparsers.add_parser("extract")
    parser_extract.set_defaults(func=extract.extract)

    parser_extract_season = subparsers.add_parser("extract_season")
    parser_extract_season.set_defaults(func=extract.extract_season)

    parser_extract_run = subparsers.add_parser("extract_run")
    parser_extract_run.set_defaults(func=extract.extract_run)

    for subparser in [parser_extract, parser_extract_season, parser_extract_run]:
        subparser.add_argument("path")
        subparser.add_argument("--gcd", default=None)
        subparser.add_argument("--outdir", required=True)
        subparser.add_argument("--tempdir", default="/tmp")
        subparser.add_argument("--overwrite", action="store_true")
        subparser.add_argument("--no-mmap", action="store_true")
        subparser.add_argument("--keep-tempfiles-on-fail", action="store_true")
        subparser.add_argument("--procs", type=int, default=cpu_count())

    parser_extract.add_argument("--keys", nargs="+", default=extract.DFLT_KEYS)
    parser_extract_run.add_argument("--keys", nargs="+", default=extract.DFLT_KEYS)
    parser_extract_season.add_argument(
        "--keys",
        nargs="+",
        default=[k for k in extract.DFLT_KEYS if k not in extract.MC_ONLY_KEYS],
    )

    # Extracting a single file has unique kwargs

    parser_extract_file = subparsers.add_parser("extract_file")
    parser_extract_file.set_defaults(func=extract.extract)
    parser_extract_file.add_argument("path")
    parser_extract_file.add_argument("--outdir", required=True)
    parser_extract_file.add_argument("--keys", nargs="+", default=None)

    # Combine runs is unique

    parser_combine_runs = subparsers.add_parser("combine_runs")
    parser_combine_runs.add_argument("path")
    parser_combine_runs.add_argument("--outdir", required=True)
    parser_combine_runs.add_argument("--keys", nargs="+", default=None)
    parser_combine_runs.add_argument("--no-mmap", action="store_true")
    parser_combine_runs.set_defaults(func=extract.combine_runs)

    # Compress / decompress are similar

    parser_compress = subparsers.add_parser("compress")
    parser_compress.set_defaults(func=cols.compress)

    parser_decompress = subparsers.add_parser("decompress")
    parser_decompress.set_defaults(func=cols.decompress)

    for subparser in [parser_compress, parser_decompress]:
        subparser.add_argument("path")
        subparser.add_argument("--keys", nargs="+", default=None)
        subparser.add_argument("-k", "--keep", action="store_true")
        subparser.add_argument("-r", "--recurse", action="store_true")
        subparser.add_argument("--procs", type=int, default=cpu_count())

    # Simple functions that add columns derived from existing columns (post-proc)

    def func_wrapper(func, path, outdir, outdtype, overwrite):
        if outdir is None:
            outdir = path
        func(path, outdir=outdir, outdtype=outdtype, overwrite=overwrite)

    for funcname in [
        "fit_genie_rw_syst",
        "calc_genie_weighted_aeff",
        "calc_normed_weights",
    ]:
        subparser = subparsers.add_parser(funcname)
        subparser.set_defaults(func=partial(func_wrapper, func=getattr(phys, funcname)))
        subparser.add_argument("path")
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
    parser_compute_coszen.set_defaults(func=compute_coszen_wrapper)
    parser_compute_coszen.add_argument("path")
    parser_compute_coszen.add_argument("--key-path", nargs="+", required=True)
    parser_compute_coszen.add_argument("--outdtype", required=False)
    parser_compute_coszen.add_argument("--outdir", required=False)
    parser_compute_coszen.add_argument("--overwrite", action="store_true")

    # Parse command line

    kwargs = vars(parser.parse_args())

    # Translate command line arguments that don't match functiona arguments

    if "keys" in kwargs and kwargs["keys"] is not None and set(["", "all"]).intersection(kwargs["keys"]):
        kwargs["keys"] = None

    if "no_mmap" in kwargs:
        kwargs["mmap"] = not kwargs.pop("no_mmap")

    # Run appropriate function

    func = kwargs.pop("func", None)
    if func is None:
        parser.parse_args(["--help"])
        return

    func(**kwargs)


if __name__ == "__main__":
    main()
