#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Operate on physical quantities stored in columns
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
    "EM_CASCADE_PTYPES",
    "HADR_CASCADE_PTYPES",
    "CASCADE_PTYPES",
    "TRACK_PTYPES",
    "INVISIBLE_PTYPES",
    "ELECTRONS",
    "MUONS",
    "TAUS",
    "NUES",
    "NUMUS",
    "NUTAUS",
    "NEUTRINOS",
    "calc_genie_weighted_aeff",
    "fit_genie_rw_syst",
    "get_most_energetic_primary",
    "get_most_energetic_primary_neutrino",
    "get_most_energetic_primary_muon",
]

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from copy import deepcopy

import numba
import numpy as np
from six import string_types

from i3cols.cols import check_outdir_and_keys, load, save_item
from i3cols.enums import ParticleType
import i3cols.dtypes as dt
from i3cols.utils import expand, get_widest_float_dtype


EM_CASCADE_PTYPES = (
    ParticleType.EMinus,
    ParticleType.EPlus,
    ParticleType.Brems,
    ParticleType.DeltaE,
    ParticleType.PairProd,
    ParticleType.Gamma,
    ParticleType.Pi0,
)
"""Particle types parameterized as electromagnetic cascades,
from clsim/python/GetHybridParameterizationList.py"""


HADR_CASCADE_PTYPES = (
    ParticleType.Hadrons,
    ParticleType.Neutron,
    ParticleType.PiPlus,
    ParticleType.PiMinus,
    ParticleType.K0_Long,
    ParticleType.KPlus,
    ParticleType.KMinus,
    ParticleType.PPlus,
    ParticleType.PMinus,
    ParticleType.K0_Short,
    ParticleType.Eta,
    ParticleType.Lambda,
    ParticleType.SigmaPlus,
    ParticleType.Sigma0,
    ParticleType.SigmaMinus,
    ParticleType.Xi0,
    ParticleType.XiMinus,
    ParticleType.OmegaMinus,
    ParticleType.NeutronBar,
    ParticleType.LambdaBar,
    ParticleType.SigmaMinusBar,
    ParticleType.Sigma0Bar,
    ParticleType.SigmaPlusBar,
    ParticleType.Xi0Bar,
    ParticleType.XiPlusBar,
    ParticleType.OmegaPlusBar,
    ParticleType.DPlus,
    ParticleType.DMinus,
    ParticleType.D0,
    ParticleType.D0Bar,
    ParticleType.DsPlus,
    ParticleType.DsMinusBar,
    ParticleType.LambdacPlus,
    ParticleType.WPlus,
    ParticleType.WMinus,
    ParticleType.Z0,
    ParticleType.NuclInt,
    ParticleType.TauPlus,
    ParticleType.TauMinus,
)
"""Particle types parameterized as hadronic cascades,
from clsim/CLSimLightSourceToStepConverterPPC.cxx with addition of TauPlus and
TauMinus"""


CASCADE_PTYPES = EM_CASCADE_PTYPES + HADR_CASCADE_PTYPES
"""Particle types classified as either EM or hadronic cascades"""


TRACK_PTYPES = (ParticleType.MuPlus, ParticleType.MuMinus)
"""Particle types classified as tracks"""


INVISIBLE_PTYPES = (
    ParticleType.Neutron,  # long decay time exceeds trigger window
    ParticleType.K0,
    ParticleType.K0Bar,
    ParticleType.NuE,
    ParticleType.NuEBar,
    ParticleType.NuMu,
    ParticleType.NuMuBar,
    ParticleType.NuTau,
    ParticleType.NuTauBar,
)
"""Invisible particles (at least to low-energy IceCube triggers)"""

ELECTRONS = (ParticleType.EPlus, ParticleType.EMinus)
MUONS = (ParticleType.MuPlus, ParticleType.MuMinus)
TAUS = (ParticleType.TauPlus, ParticleType.TauMinus)
NUES = (ParticleType.NuE, ParticleType.NuEBar)
NUMUS = (ParticleType.NuMu, ParticleType.NuMuBar)
NUTAUS = (ParticleType.NuTau, ParticleType.NuTauBar)
NEUTRINOS = NUES + NUMUS + NUTAUS


def calc_genie_weighted_aeff(path, outdtype=None, outdir=None, overwrite=False):
    """Calculate weighted effective area in [GeV cm**2 sr] for GENIE events
    This EXCLUDES flux/osc, e.g. weight = weighted_aeff * flux * osc

    Parameters
    ----------
    path : str
    outdtype : None or numpy dtype, optional
        If not specified, defaults to widest floating point dtype in input
        columns
    outdir : str, optional
    overwrite : bool

    Returns
    -------
    weighted_aeff

    """
    outkey = "weighted_aeff"
    outdir = check_outdir_and_keys(outdir, outkeys=[outkey], overwrite=overwrite)

    arrays, scalar_ci = load(path, keys=["I3MCWeightDict"], mmap=True)

    i3mcwd = arrays["I3MCWeightDict"]["data"]
    num_files = len(scalar_ci["subrun"])

    # Calculate weighted effective area
    # Normalise by number of input files that have been processed into this
    # output file
    # Note that if you combine mutliple output files in your analysis you will
    # need to normalise by that number still
    #
    # Notes:
    #   * the 1e-4 conversion factor goes from cm^2 to m^2 (PISA flux tables
    #       are stored in # m^2)
    #   * num_files = number of "originally-formatted" (i.e. 1:1 with
    #       simulation i3 files, not split or combined in any way) I3 files
    #       combined to make this MC set
    #   * gen_ratio = nu/nubar ratio in simulation

    one_weight = i3mcwd["OneWeight"]
    n_events = i3mcwd["NEvents"]
    gen_ratio = i3mcwd["gen_ratio"]

    if outdtype is None:
        outdtype = get_widest_float_dtype(
            dtypes=[x.dtype for x in [one_weight, n_events, gen_ratio]]
        )

    weighted_aeff = (1e-4 * one_weight / (num_files * n_events * gen_ratio)).astype(
        outdtype
    )

    if outdir is not None:
        save_item(path=path, key=outkey, data=weighted_aeff, overwrite=overwrite)

    return weighted_aeff


def fit_genie_rw_syst(obj, outdtype=None, outdir=None, overwrite=False):
    """Fit 2nd-order polynomial to GENIE reweighting systematics, optionally
    saving the array to disk.

    Parameters
    ----------
    obj : str, mapping, or numpy ndarray
    outdtype : None or numpy dtype, optional
    outdir : str or None, optional
    overwrite : bool, optional

    Returns
    -------
    fit_coeffs : numpy ndarray
        Structured dtype .. ::

            (syst name, (linear, quadratic))

    Examples
    --------
    E.g., obtain the rw_AhtBY fit coeff arrays by

    >>> lin_ary = fit_coeffs["rw_AhtBY"]["linear"]
    >>> quad_ary = fit_coeffs["rw_AhtBY"]["quadratic"]

    or for a single event, implicitly (i.e., you have to get the order of the
    dtype correct)

    >>> lin, quad = fit_coeffs[idx]["rw_AhtBY"]

    or, more explicitly and independent of dtype order,

    >>> lin, quad = fit_coeffs[idx]["rw_AhtBY"][["linear", "quadratic"]]

    """
    # TODO: if a `valid` array exists, make sure to skip invalid events!
    # TODO: parallelize (but must write to shared memory & probably chunk data
    #   up into larger blocks than per-event, or else multiprocessing overhead
    #   is too slow)
    if isinstance(obj, string_types):
        obj = expand(obj)
        arrays, _ = load(obj, keys="I3GENIEResultDict", mmap=True)
        grd_array = arrays["I3GENIEResultDict"]["data"]
    elif isinstance(obj, Mapping):
        grd_array = obj["I3GENIEResultDict"]["data"]
    elif isinstance(obj, np.ndarray):
        grd_array = obj

    outkey = "GENIE_rw_syst_fit_coeffs"
    outdir = check_outdir_and_keys(outdir, outkeys=outkey, overwrite=overwrite)

    rw_syst_names = sorted(n for n in grd_array.dtype.names if n.startswith("rw_"))

    if outdtype is None:
        outdtype = get_widest_float_dtype(
            dtypes=[grd_array[n].dtype for n in rw_syst_names]
        )

    # Only storing linear and quadratic fit coefficients
    coeff_t = np.dtype([("linear", outdtype), ("quadratic", outdtype)])

    # Super-dtype used for output array composed of one coeff_t per systematic
    syst_fits_coeff_t = np.dtype([(n, coeff_t) for n in rw_syst_names])

    # x values used in all polynomial fits...
    # TODO: are these the number of std. devs.?
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float64)

    # Establish we won't overwrite anything unless we want to before doing the
    # work

    # Extract info and find fit coefficients

    fit_coeffs = np.empty(shape=len(grd_array), dtype=syst_fits_coeff_t)

    for rw_syst_name in rw_syst_names:
        in_yvalues = grd_array[rw_syst_name]

        this_coeffs = fit_coeffs[rw_syst_name]

        # Fit to all-1's: slope (linear) and curvature (quadratic) both = 0
        all_ones_mask = np.all(in_yvalues == 1, axis=1)
        this_coeffs[all_ones_mask] = 0

        for idx in np.argwhere(~all_ones_mask).flat:
            in_y = in_yvalues[idx]
            inv_wght_i = 1 / grd_array[idx]["wght"]
            y = (
                in_y[0] * inv_wght_i,
                in_y[1] * inv_wght_i,
                1,
                in_y[2] * inv_wght_i,
                in_y[3] * inv_wght_i,
            )

            # Note that np.polynomial.polynomial.polyfit returns the
            # coefficients from low to high order (as opposed to np.polyfit)
            (
                _,
                this_coeffs[idx]["linear"],
                this_coeffs[idx]["quadratic"],
            ) = np.polynomial.polynomial.polyfit(x, y, deg=2)

    # Save array to disk

    if outdir is not None:
        save_item(path=outdir, key=outkey, data=fit_coeffs, overwrite=overwrite)

    return fit_coeffs


def calc_normed_weights(path, outdtype=None, outdir=None, overwrite=False):
    """Normalize Monte Carlo weights by dividing by the number of i3 files that
    comprise this MC set.

    """
    outkey = "normed_weight"
    outdir = check_outdir_and_keys(outdir=outdir, outkeys=[outkey], overwrite=overwrite)

    arrays, scalar_ci = load(path, keys=["I3MCWeightDict"], mmap=True)
    num_files = len(scalar_ci["subrun"])
    weight = arrays["I3MCWeightDict"]["data"]["weight"]

    if outdtype is None:
        outdtype = get_widest_float_dtype(weight.dtype)

    normed_weight = (weight / num_files).astype(outdtype)

    if outdir is not None:
        save_item(path=outdir, key=outkey, data=normed_weight, overwrite=overwrite)

    return normed_weight


def get_nuflavint(path, outdtype=None, outdir=None, overwrite=False):
    """
    """
    outkey = "nuflavint"
    outdir = check_outdir_and_keys(outdir=outdir, outkeys=[outkey], overwrite=overwrite)

    arrays, scalar_ci = load(path, keys=["I3MCTree", "I3MCWeightDict"], mmap=True)
    num_files = len(scalar_ci["subrun"])
    weight = arrays["I3MCWeightDict"]["data"]["weight"]

    if outdtype is None:
        outdtype = np.dtype(
            [("pdg_encoding", np.int32), ("interaction_type", np.uint8)]
        )

    normed_weight = (weight / num_files).astype(outdtype)

    if outdir is not None:
        save_item(path=outdir, key=outkey, data=normed_weight, overwrite=overwrite)

    return normed_weight


def coszen_key_from_zen_key_path(key_path):
    """Create a reasonable key name for a "cosine(zenith)" field that is
    derived from a "zenith" field at key path `key_path`

    Renaming rules:
        * If final path element contains word "zenith" or "zen", replace this
            word with "coszen"
        * Otherwise, add the word "coszen" as a final part
        * Finally, join all parts with double-underscores: "__"

    Parameters
    ----------
    key_path : str or iterable thereof

    Returns
    -------
    outkey

    """
    if isinstance(key_path, string_types):
        key_path = [key_path]
    key_path = list(key_path)

    outkey_parts = deepcopy(key_path)
    if "zenith" in outkey_parts[-1]:
        outkey_parts[-1] = outkey_parts[-1].replace("zenith", "coszen")
    elif "zen" in outkey_parts[-1]:
        outkey_parts[-1] = key_path[-1].replace("zen", "coszen")
    else:
        outkey_parts.append("coszen")
    outkey = "__".join(outkey_parts)

    return outkey


def compute_coszen(
    path, key_path, outkey=None, outdtype=None, outdir=None, overwrite=False
):
    """Compute cosine of a zenith field.

    Parameters
    ----------
    path : str
        Path to the key directory
    key_path : str or iterable thereof
        Path to traverse to get to the zenith field. E.g. .. ::

            key_path = ["L4_iLineFit", "dir", "zenith"]

        will traverse the `arrays` at `path` via .. ::

            arrays["L4_iLineFit"]["data"]["dir"]["zenith"]

    outkey : None or str, optional
        Name of key to save coszen values to; if None provided, will use
        `coszen_key_from_zen_key_path` function to derive a reasonable name based on
        the `key_path`
    outdtype : None or numpy dtype, optional
        Data type to convert results to (_after_ applying `numpy.cos` to the
        input zenith values, regardless of what dtype those are). If not
        specified, defaults to the dtype of the input data
    outdir : None or str, optional
        Output key directory. If None is provided, no file is written
    overwrite : bool
        Whether to overwrite `outkey` in `outdir` if it alrady exists

    Returns
    -------
    coszen : shape-(len(data),) numpy ndarray of dtype `outdtype`

    """
    # TODO: make generic not just to cos: numpy ufuncs can be fast & don't
    #   require all the machinery of `apply`
    # TODO: use a `valid` array if it exists at `path[..key_path..]`

    if isinstance(key_path, string_types):
        key_path = [key_path]
    key_path = list(key_path)

    if outkey is None:
        outkey = coszen_key_from_zen_key_path(key_path)

    outdir = check_outdir_and_keys(outdir, outkeys=outkey, overwrite=overwrite)

    arrays, _ = load(path, keys=key_path[0], mmap=True)

    data = arrays[key_path[0]]["data"]

    for key in key_path[1:]:
        data = data[key]

    if outdtype is None:
        outdtype = get_widest_float_dtype(data.dtype)

    out = np.cos(data).astype(outdtype)

    if outdir is not None:
        save_item(path=outdir, key=outkey, data=out, overwrite=overwrite)

    return out


@numba.njit(fastmath=True, cache=True, error_model="numpy")
def get_most_energetic_primary(flat_particles, class_abs_pdg_codes):
    """Get most energetic primary particle, filtered by specific PDG codes"""
    most_energetic_primary = np.empty(shape=1, dtype=dt.I3PARTICLE_T)[0]
    most_energetic_primary["energy"] = -np.inf

    for flat_particle in flat_particles:
        if flat_particle["level"] > 0:
            continue

        particle = flat_particle["particle"]
        if particle["pdg_encoding"] not in class_abs_pdg_codes:
            continue

        if most_energetic_primary is None:
            most_energetic_primary = particle
        elif particle["energy"] > most_energetic_primary["energy"]:
            most_energetic_primary = particle

    assert most_energetic_primary["energy"] >= 0

    return most_energetic_primary


@numba.njit(fastmath=True, cache=True, error_model="numpy")
def get_most_energetic_primary_neutrino(flat_particles):
    """Get most energetic primary neutrino"""
    return get_most_energetic_primary(flat_particles, NEUTRINOS)


@numba.njit(fastmath=True, cache=True, error_model="numpy")
def get_most_energetic_primary_muon(flat_particles):
    """Get most energetic primary muon"""
    return get_most_energetic_primary(flat_particles, MUONS)
