#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Functions for working with i3cols versions of I3Particle (I3PARTICLE_T) and
I3MCTree (FLAT_PARTICLE_T).

See dataclasses/private/dataclasses/physics/I3MCTreePhysicsLibrary.cxx
and dataclasses/private/dataclasses/physics/I3Particle.cxx
"""


from __future__ import absolute_import, division, print_function

__author__ = "Justin L. Lanfranchi for the IceCube Collaboration"

__all__ = [
    "get_null_particle",
    "get_best_filter",
    "true_filter",
    "is_cascade",
    "is_neutrino",
    "is_nucleus",
    "is_track",
    "is_muon",
    "more_energetic",
    "get_most_energetic",
    "get_most_energetic_neutrino",
    "get_most_energetic_muon",
    "get_most_energetic_track",
]

import copy
import numba
import numpy as np

from i3cols import dtypes, enums


@numba.njit(cache=True, error_model="numpy")
def get_null_particle():
    """Get a null particle for use when an invalid / n/a result is desired.

    Returns
    -------
    null_particle : shape () ndarray of dtype I3PARTICLE_T

    """
    null_particle = np.empty(shape=1, dtype=dtypes.I3PARTICLE_T)[0]
    # TODO: set majorID, minorID to random values?
    null_particle["id"]["majorID"] = 0
    null_particle["id"]["minorID"] = 0
    null_particle["pdg_encoding"] = 0
    null_particle["shape"] = enums.ParticleShape.Null
    null_particle["pos"]["x"] = np.nan
    null_particle["pos"]["y"] = np.nan
    null_particle["pos"]["z"] = np.nan
    null_particle["dir"]["zenith"] = np.nan
    null_particle["dir"]["azimuth"] = np.nan
    null_particle["time"] = np.nan
    null_particle["energy"] = np.nan
    null_particle["length"] = np.nan
    null_particle["speed"] = np.nan
    null_particle["fit_status"] = enums.FitStatus.NotSet
    null_particle["location_type"] = enums.LocationType.Anywhere
    return null_particle


@numba.njit(error_model="numpy")
def get_best_filter(particles, filter_function, cmp_function):
    """Get best particle according to `cmp_function`, only looking at particles
    for which `filter_function` returns `True`. If no particle meeting these
    criteria is found, returns a copy of `NULL_I3PARTICLE`.

    See dataclasses/public/dataclasses/physics/I3MCTreeUtils.h

    Parameters
    ----------
    particles : ndarray of dtyppe I3PARTICLE_T
    filter_function : numba Callable(I3PARTICLE_T)
    cmp_function : numba Callable(I3PARTICLE_T, I3PARTICLE_T)

    Returns
    -------
    best_particle : shape () ndarray of dtype I3PARTICLE_T

    """
    best_particle = get_null_particle()

    for particle in particles:
        if filter_function(particle) and cmp_function(test=particle, ref=best_particle):
            best_particle = particle

    return best_particle


@numba.njit(cache=True, error_model="numpy")
def true_filter(test):  # pylint: disable=unused-argument
    """Simply return True regardless of the input.

    Designed to be used with `get_best_filter` where no filtering is desired;
    intended to have same effect as `IsParticle` function defined in
    dataclasses/private/dataclasses/physics/I3MCTreePhysicsLibrary.cxx

    Parameters
    ----------
    test

    Returns
    -------
    True : bool

    """
    return True


@numba.njit(cache=True, error_model="numpy")
def is_cascade(particle):
    """Test if particle is a cascade.

    See dataclasses/private/dataclasses/physics/I3Particle.cxx

    Parameters
    ----------
    particle : shape () ndarray of dtype I3PARTICLE_T

    Returns
    -------
    is_cascade : bool

    """
    return (
        particle["shape"]
        in (enums.ParticleShape.Cascade, enums.ParticleShape.CascadeSegment,)
        or particle["pdg_encoding"]
        in (
            enums.ParticleType.EPlus,
            enums.ParticleType.EMinus,
            enums.ParticleType.Brems,
            enums.ParticleType.DeltaE,
            enums.ParticleType.PairProd,
            enums.ParticleType.NuclInt,
            enums.ParticleType.Hadrons,
            enums.ParticleType.Pi0,
            enums.ParticleType.PiPlus,
            enums.ParticleType.PiMinus,
        )
        or (
            particle["shape"] != enums.ParticleShape.Primary
            and (
                is_nucleus(particle)
                or particle["pdg_encoding"]
                in (
                    enums.ParticleType.PPlus,
                    enums.ParticleType.PMinus,
                    enums.ParticleType.Gamma,
                )
            )
        )
    )


@numba.njit(cache=True, error_model="numpy")
def is_neutrino(particle):
    """Test if particle is a neutrino.

    See dataclasses/private/dataclasses/physics/I3Particle.cxx

    Parameters
    ----------
    particle : shape () ndarray of dtype I3PARTICLE_T

    Returns
    -------
    is_neutrino : bool

    """
    return particle["pdg_encoding"] in (
        enums.ParticleType.NuE,
        enums.ParticleType.NuEBar,
        enums.ParticleType.NuMu,
        enums.ParticleType.NuMuBar,
        enums.ParticleType.NuTau,
        enums.ParticleType.NuTauBar,
        enums.ParticleType.Nu,
    )


@numba.njit(cache=True, error_model="numpy")
def is_nucleus(particle):
    """Test if particle is a nucleus.

    See dataclasses/private/dataclasses/physics/I3Particle.cxx

    Parameters
    ----------
    particle : shape () ndarray of dtype I3PARTICLE_T

    Returns
    -------
    is_nucleus : bool

    """
    return 1000000000 <= abs(particle["pdg_encoding"]) <= 1099999999


@numba.njit(cache=True, error_model="numpy")
def is_track(particle):
    """Test if particle is a track.

    See dataclasses/private/dataclasses/physics/I3Particle.cxx

    Parameters
    ----------
    particle : shape () ndarray of dtype I3PARTICLE_T

    Returns
    -------
    is_track : bool

    """
    return (
        particle["shape"]
        in (
            enums.ParticleShape.InfiniteTrack,
            enums.ParticleShape.StartingTrack,
            enums.ParticleShape.StoppingTrack,
            enums.ParticleShape.ContainedTrack,
        )
        or particle["pdg_encoding"]
        in (
            enums.ParticleType.MuPlus,
            enums.ParticleType.MuMinus,
            enums.ParticleType.TauPlus,
            enums.ParticleType.TauMinus,
            enums.ParticleType.STauPlus,
            enums.ParticleType.STauMinus,
            enums.ParticleType.SMPPlus,
            enums.ParticleType.SMPMinus,
            enums.ParticleType.Monopole,
            enums.ParticleType.Qball,
        )
        or (
            particle["shape"] == enums.ParticleShape.Primary
            and (
                is_nucleus(particle)
                or particle["pdg_encoding"]
                in (
                    enums.ParticleType.PPlus,
                    enums.ParticleType.PMinus,
                    enums.ParticleType.Gamma,
                )
            )
        )
    )


@numba.njit(cache=True, error_model="numpy")
def is_muon(particle):
    """Test if particle is a muon.

    See dataclasses/private/dataclasses/physics/I3Particle.cxx

    Parameters
    ----------
    particle : shape () ndarray of dtype I3PARTICLE_T

    Returns
    -------
    is_muon : bool

    """
    return (
        particle["pdg_encoding"] == enums.ParticleType.MuPlus
        or particle["pdg_encoding"] == enums.ParticleType.MuMinus
    )


@numba.njit(cache=True, error_model="numpy")
def more_energetic(test, ref):
    """Is `test` particle more energetic than `ref` particle?

    Not if `test` energy is NaN, always returns False.

    Designed to be used with `get_best_filter`.

    See function `MoreEnergetic` in
    dataclasses/private/dataclasses/physics/I3MCTreePhysicsLibrary.cxx

    Parameters
    ----------
    test : I3PARTICLE_T
    ref : I3PARTICLE_T

    Returns
    -------
    is_most_energetic : bool

    """
    if np.isnan(test["energy"]):
        return False

    if np.isnan(ref["energy"]):
        return True

    return test["energy"] > ref["energy"]

    # return not np.isnan(test["energy"]) and (
    #    np.isnan(ref["energy"]) or test["energy"] > ref["energy"]
    # )


@numba.njit(error_model="numpy")
def get_most_energetic(particles):
    """Get most energetic particle. If no particle with a non-NaN energy is
    found, returns a copy of `NULL_I3PARTICLE`.

    Parameters
    ----------
    particles : ndarray of dtyppe I3PARTICLE_T

    Returns
    -------
    most_energetic : shape () ndarray of dtype I3PARTICLE_T

    """
    return get_best_filter(
        particles=particles, filter_function=true_filter, cmp_function=more_energetic,
    )


@numba.njit(error_model="numpy")
def get_most_energetic_neutrino(particles):
    """Get most energetic neutrino.

    Parameters
    ----------
    particles : ndarray of dtype I3PARTICLE_T

    Returns
    -------
    most_energetic_neutrino : shape () ndarray of dtype I3PARTICLE_T

    """
    return get_best_filter(
        particles=particles, filter_function=is_neutrino, cmp_function=more_energetic,
    )


@numba.njit(error_model="numpy")
def get_most_energetic_muon(particles):
    """Get most energetic muon.

    Parameters
    ----------
    particles : ndarray of dtype I3PARTICLE_T

    Returns
    -------
    most_energetic_muon : shape () ndarray of dtype I3PARTICLE_T

    """
    return get_best_filter(
        particles=particles, filter_function=is_muon, cmp_function=more_energetic,
    )


@numba.njit(error_model="numpy")
def get_most_energetic_track(particles):
    """Get most energetic track.

    Parameters
    ----------
    particles : ndarray of dtype I3PARTICLE_T

    Returns
    -------
    most_energetic_track : shape () ndarray of dtype I3PARTICLE_T

    """
    return get_best_filter(
        particles=particles, filter_function=is_track, cmp_function=more_energetic,
    )
