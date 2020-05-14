#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Interface for functions within the i3cols module
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


import argparse
import inspect
import os
import sys

from i3cols import cols, extract, utils


GCD_HELP = """Specify the path to a single GCD file (required in the case of
Monte Carlo simulation) or the path to a directory containing data run GCD
files"""

SUB_EVENT_STREAM_HELP = """Only take frames from specified sub event stream(s).
If none are specified, no filtering is performed based on sub event stream"""

KEYS_HELP = r"""Specify to only operate on specific key(s); if --keys is not
specified, operate on all keys (except for things specified by --exclude-keys).
If KEYS is a path to a  file, the file is read as "\n"-separated key names (do
NOT use quotes within the file). Globbing is allowed in key names (both at the
command line and in a keys file): "*" means match 0 or more of any character,
"?" means match one of any character, and square brackets denote character
matches, e.g., "[afq]" means match one of "a", "f", or "q". All glob pattern
matches are performed case-insensitive, but exact key names are matched
case-sensitive."""

EXCLUDE_KEYS_HELP = r"""Specify to exclude particular keys. Similar to --keys,
filenames are read as "\n"-separated key names, and globbing patterns are legal
on the command line or in the file."""

OVERWRITE_HELP = """Overwrite existing files"""

COMPRESS_HELP = """Compress resulting column directories"""

TEMPDIR_HELP = """Place temporary files within this directory; if not
specified, /tmp is chosen. In any case, a new and unique directory is created
within the specified TEMPDIR to avoid file collisions with other processes
(default: /tmp)"""

KEEP_TEMPFILES_HELP = """If an error occurs, do NOT remove temporary files"""

PROCS_HELP = """Number of subprocesses (i.e., cores or hyperthreads) to use for
processing. (default: %(default)d)"""

OUTDIR_HELP = "Store final results within this directory"


def main(description=__doc__):
    """Command line interface"""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="""For detailed help on a command, specify the
        command and -h or --help""",
    )

    all_sp = []

    # Extract files separately

    subparser = subparsers.add_parser(
        name="extr_sep", help="""Extract i3 file(s) separately"""
    )
    all_sp.append(subparser)
    subparser.set_defaults(func=extract.extract_files_separately)
    subparser.add_argument("--outdir", required=True, help=OUTDIR_HELP)
    subparser.add_argument(
        "--concatenate-and-index-by",
        required=False,
        choices=list(utils.ALL_CATEGORY_XFORMS.keys()),
        help="""Concatenate the individually extracted files and index using
        this characteristic of the source file names and/or paths""",
    )
    subparser.add_argument("--gcd", default=None, help=GCD_HELP)
    subparser.add_argument(
        "--sub-event-stream", nargs="+", default=None, help=SUB_EVENT_STREAM_HELP
    )
    subparser.add_argument("--keys", nargs="+", default=None, help=KEYS_HELP)
    subparser.add_argument(
        "--exclude-keys", nargs="+", default=None, help=EXCLUDE_KEYS_HELP
    )
    subparser.add_argument("--overwrite", action="store_true", help=OVERWRITE_HELP)
    subparser.add_argument("--compress", action="store_true", help=COMPRESS_HELP)
    subparser.add_argument("--tempdir", default=None, help=TEMPDIR_HELP)
    subparser.add_argument(
        "--keep-tempfiles-on-fail", action="store_true", help=KEEP_TEMPFILES_HELP
    )
    subparser.add_argument("--procs", type=int, default=1, help=PROCS_HELP)

    # Extract i3 files as if they are one large i3 file (interspersing GCD
    # files if specified and as appropriate)

    subparser = subparsers.add_parser(
        "extr_as_one",
        help="""Extract i3 file(s) as if they are one long i3 file,
        optionally interspersing GCD files if specified and only as needed""",
    )
    all_sp.append(subparser)
    subparser.set_defaults(func=extract.extract_files_as_one)
    subparser.add_argument("--outdir", required=True, help=OUTDIR_HELP)
    subparser.add_argument("--gcd", default=None, help=GCD_HELP)
    subparser.add_argument(
        "--sub-event-stream", nargs="+", default=None, help=SUB_EVENT_STREAM_HELP
    )
    subparser.add_argument("--keys", nargs="+", default=None, help=KEYS_HELP)
    subparser.add_argument(
        "--exclude-keys", nargs="+", default=None, help=EXCLUDE_KEYS_HELP
    )
    subparser.add_argument("--overwrite", action="store_true", help=OVERWRITE_HELP)
    subparser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the extracted individual column directories",
    )
    subparser.add_argument(
        "--procs",
        type=int,
        default=1,
        help="""Only has an effect if --compress is specified; extracting files
        runs in a single process (default: %(default)d)""",
    )

    # Extract i3 data season files (look for "Run" directories, and "subrun"
    # .i3 files within each run directory). Each run is extracted via
    # "extract_files_as_one" but runs can be extracted in parallel.

    subparser = subparsers.add_parser(
        "extr_season",
        help="""Extract a data season. All subruns within a run are extracted
        as a single file (within a single process), but each run can be extracted
        independently of one another (i.e., can be done in parallel)""",
    )
    all_sp.append(subparser)
    subparser.set_defaults(func=extract.extract_season)
    subparser.add_argument("--outdir", required=True, help=OUTDIR_HELP)
    subparser.add_argument(
        "--concatenate-and-index",
        action="store_true",
        help="""Concatenate the extracted runs and create an index of where
        each of the season's runs is in the concatenated data""",
    )
    subparser.add_argument("--gcd", default=None, help=GCD_HELP)
    subparser.add_argument(
        "--sub-event-stream", nargs="+", default=None, help=SUB_EVENT_STREAM_HELP
    )
    subparser.add_argument("--keys", nargs="+", default=None, help=KEYS_HELP)
    subparser.add_argument(
        "--exclude-keys", nargs="+", default=None, help=EXCLUDE_KEYS_HELP
    )
    subparser.add_argument("--overwrite", action="store_true", help=OVERWRITE_HELP)
    subparser.add_argument("--compress", action="store_true", help=COMPRESS_HELP)
    subparser.add_argument("--tempdir", default=None, help=TEMPDIR_HELP)
    subparser.add_argument(
        "--keep-tempfiles-on-fail", action="store_true", help=KEEP_TEMPFILES_HELP
    )
    subparser.add_argument("--procs", type=int, default=1, help=PROCS_HELP)

    # Concatenate files, augmenting existing indexes and optionally adding a
    # new index

    subparser = subparsers.add_parser(
        "cat",
        help="""Concatenate columns. Each path can be a directory containing
        columns or a directory of such directories."""
    )
    all_sp.append(subparser)
    subparser.set_defaults(func=cols.concatenate)
    subparser.add_argument("--outdir", required=True, help=OUTDIR_HELP)
    subparser.add_argument("--keys", nargs="+", default=None, help=KEYS_HELP)
    subparser.add_argument(
        "--exclude-keys", nargs="+", default=None, help=EXCLUDE_KEYS_HELP
    )
    subparser.add_argument(
        "--index-name",
        required=True,
        choices=list(utils.ALL_CATEGORY_XFORMS.keys()),
        help="""Concatenate using this characteristic of the source file names
        and/or paths""",
    )

    # Compress / decompress are similar

    parser_compress = subparsers.add_parser(
        "compress",
        help="Compress column directories (containing npy files) as npz files",
    )
    all_sp.append(parser_compress)
    parser_compress.set_defaults(func=cols.compress)

    parser_decompress = subparsers.add_parser(
        "decompress",
        help="Decompress column npz files into directories containing npy files",
    )
    all_sp.append(parser_decompress)
    parser_decompress.set_defaults(func=cols.decompress)

    for subparser in [parser_compress, parser_decompress]:
        subparser.add_argument("--keys", nargs="+", default=None, help=KEYS_HELP)
        subparser.add_argument(
            "--exclude-keys", nargs="+", default=None, help=EXCLUDE_KEYS_HELP
        )
        subparser.add_argument(
            "-k", "--keep", action="store_true", help="Keep original files"
        )
        subparser.add_argument(
            "-r",
            "--recurse",
            action="store_true",
            help="Search recursively within path(s) for files",
        )
        subparser.add_argument("--procs", type=int, default=1, help=PROCS_HELP)

    # Simple functions that add columns derived from existing columns (post-proc)

    # TODO: single function to perform oscNext post-processing (not repeating
    #       already-performed operations)?

    # for funcname in [
    #    "fit_genie_rw_syst",
    #    "calc_genie_weighted_aeff",
    #    "calc_normed_weights",
    # ]:
    #    subparser = subparsers.add_parser(funcname)
    #    all_sp.append(subparser)
    #    subparser.set_defaults(func=getattr(phys, funcname))
    #    subparser.add_argument("--outdtype", required=False)
    #    subparser.add_argument("--outdir", required=True, help=OUTDIR_HELP)
    #    subparser.add_argument("--overwrite", action="store_true")

    ## More complicated add-column post-processing functions

    # def compute_coszen_wrapper(
    #    path, key_path, outdir, outkey=None, outdtype=None, overwrite=False
    # ):
    #    if outdir is None:
    #        outdir = path

    #    if isinstance(outdtype, string_types):
    #        if hasattr(np, outdtype):
    #            outdtype = getattr(np, outdtype)
    #        else:
    #            outdtype = np.dtype(outdtype)

    #    phys.compute_coszen(
    #        path=path,
    #        key_path=key_path,
    #        outkey=outkey,
    #        outdtype=outdtype,
    #        outdir=outdir,
    #        overwrite=overwrite,
    #    )

    # parser_compute_coszen = subparsers.add_parser("compute_coszen")
    # all_sp.append(parser_compute_coszen)
    # parser_compute_coszen.set_defaults(func=compute_coszen_wrapper)
    # parser_compute_coszen.add_argument("--key-path", nargs="+", required=True)
    # parser_compute_coszen.add_argument("--outdtype", required=False)
    # parser_compute_coszen.add_argument("--outdir", required=False, help=OUTDIR_HELP)
    # parser_compute_coszen.add_argument("--overwrite", action="store_true")

    # Add args common to all

    for subparser in all_sp:
        subparser.add_argument(
            "-0",
            action="store_true",
            help="Split stdin by null chars; e.g. find . -print0 | {} -0".format(
                os.path.basename(sys.argv[0])
            ),
        )
        args = inspect.getargspec(subparser.get_default("func")).args
        path_argname = None
        if "paths" in args:
            subparser.add_argument(
                "paths", nargs="*", default=sys.stdin, help="Paths to files to process"
            )
            subparser.add_argument(
                "--sort",
                action="store_true",
                help="""Whether to sort input paths in ascending order; parses
                numerical parts of filenames as numbers such that, e.g., x2y
                comes before x10y""",
            )
        elif "path" in args:
            subparser.add_argument(
                "path",
                nargs="*",
                default=sys.stdin,
                help="Path to directory to process",
            )

    # Parse command line

    kwargs = vars(parser.parse_args())

    # Translate command line arguments that don't match functiona arguments

    if "no_mmap" in kwargs:
        kwargs["mmap"] = not kwargs.pop("no_mmap")

    if "concatenate_and_index_by" in kwargs:
        index_name = kwargs.pop("concatenate_and_index_by")
        kwargs["index_name"] = index_name
        kwargs["concatenate_and_index"] = index_name is not None

        index_name_category_xform_map = {None: None}  # doesn't matter, not indexing
        index_name_category_xform_map.update(utils.ALL_CATEGORY_XFORMS)

        if index_name not in index_name_category_xform_map:
            raise ValueError("Invalid / unhandled index '{}'".format(index_name))

        kwargs["category_xform"] = index_name_category_xform_map[index_name]

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
