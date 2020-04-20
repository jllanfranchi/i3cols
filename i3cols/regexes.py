# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order


"""
Project-wide useful regular expressions
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
    "KEY_NAME_RE",
    "I3_FNAME_RE",
    "I3_DETECTOR_SEASON_RE",
    "I3_LEVEL_LEVELVER_RE",
    "I3_PASS_RE",
    "I3_RUN_RE",
    "I3_SUBRUN_RE",
    "I3_PART_RE",
    "I3_SUBRUN_DIR_RE",
    "I3_RUN_DIR_RE",
    "IC_SEASON_DIR_RE",
    "I3_OSCNEXT_FNAME_RE",
    "test_OSCNEXT_I3_FNAME_RE",
]

import re
import sys


KEY_NAME_RE = re.compile(
    r"""
    (?P<key_name>[^ \t\n]+)  # see: icetray/private/icetray/I3Frame.cxx : validate_name
    ([ ](?P<version>\S+))?   # version delimited by space followed by 1+ non-whitespace
    (?P<compr_exts>(\.(npz|zst|bz2|gz))+)  # allow multiple compr exts(?)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

I3_FNAME_RE = re.compile(
    r"(?P<basename>.*)\.i3(?P<compr_exts>(\..*)*)", flags=re.IGNORECASE
)

I3_DETECTOR_SEASON_RE = re.compile(
    "(?P<detector>IC(?:79|86))([^0-9](?P<season>[0-9]+))", flags=re.IGNORECASE
)

I3_LEVEL_LEVELVER_RE = re.compile(
    "level(?P<level>[0-9]+)(.?v(?P<levelver>[0-9.]+))?", flags=re.IGNORECASE
)

I3_PASS_RE = re.compile("pass(?P<pass>[0-9][a-z]?)", flags=re.IGNORECASE)

I3_RUN_RE = re.compile(r"Run(?P<run>[0-9]+)", flags=re.IGNORECASE)

I3_SUBRUN_RE = re.compile("subrun(?P<subrun>[0-9]+(_[0-9]+)?)", flags=re.IGNORECASE)

I3_PART_RE = re.compile("part(?P<part>[0-9]+)", flags=re.IGNORECASE)

I3_SUBRUN_DIR_RE = re.compile(r"(?P<pfx>subrun)(?P<subrun>[0-9]+)", flags=re.IGNORECASE)
"""Matches MC "run" dirs, e.g. '140000' & data run dirs, e.g. 'Run00125177'"""

I3_RUN_DIR_RE = re.compile(r"(?P<pfx>Run)?(?P<run>[0-9]+)", flags=re.IGNORECASE)
"""Matches MC "run" dirs, e.g. '140000' & data run dirs, e.g. 'Run00125177'"""

IC_SEASON_DIR_RE = re.compile(
    r"((?P<detector>IC)(?P<configuration>[0-9]+)\.)(?P<year>(20)?[0-9]{2})",
    flags=re.IGNORECASE,
)
"""Matches data season dirs, e.g. 'IC86.11' or 'IC86.2011'"""

I3_OSCNEXT_FNAME_RE = re.compile(
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


def test_OSCNEXT_I3_FNAME_RE():
    """Unit tests for I3_OSCNEXT_FNAME_RE."""
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
            match = I3_OSCNEXT_FNAME_RE.match(test_input)
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


if __name__ == "__main__":
    test_OSCNEXT_I3_FNAME_RE()
