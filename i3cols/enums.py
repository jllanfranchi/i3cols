#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order, import-outside-toplevel


"""
Useful enums, some mapped directly from IceCube software
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
    "InteractionType",
    "OnOff",
    "ParticleType",
    "ParticleShape",
    "PulseFlags",
    "LocationType",
    "FitStatus",
    "TriggerConfigID",
    "TriggerTypeID",
    "TriggerSourceID",
    "TriggerSubtypeID",
    "ExtractionError",
    "OMType",
    "CableType",
    "DOMGain",
    "TrigMode",
    "LCMode",
    "ToroidType",
    "CramerRaoStatus",
]

import enum


class InteractionType(enum.IntEnum):
    """Neutrino interactions are either charged current (cc) or neutral current
    (nc); integer encodings are copied from the dominant IceCube software
    convention.
    """

    # pylint: disable=invalid-name
    undefined = 0
    CC = 1
    NC = 2


class OnOff(enum.IntEnum):
    """enum OnOff from public/dataclasses/status/I3DOMStatus.h

    Representable by np.int8.

    """

    # pylint: disable=invalid-name
    Unknown = -1
    Off = 0
    On = 1


class ParticleType(enum.IntEnum):
    """Particle types as found in an `I3Particle`.

    Only Requires int32 dtype for storage.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18;
    added (and so names might not be "standard"):
        K0, K0Bar, SigmaaCPP, SigmaCP

    """

    # pylint: disable=invalid-name

    # NB: These match the PDG codes. Keep it that way!
    unknown = 0
    Gamma = 22
    EPlus = -11
    EMinus = 11
    MuPlus = -13
    MuMinus = 13
    Pi0 = 111
    PiPlus = 211
    PiMinus = -211
    K0_Long = 130
    K0 = 311
    K0Bar = -311
    KPlus = 321
    KMinus = -321
    SigmaCPP = 4222  # charmed Sigma ++
    SigmaCP = 4212  # charmed Sigma +
    Neutron = 2112
    PPlus = 2212
    PMinus = -2212
    K0_Short = 310
    Eta = 221
    Lambda = 3122
    SigmaPlus = 3222
    Sigma0 = 3212
    SigmaMinus = 3112
    Xi0 = 3322
    XiMinus = 3312
    OmegaMinus = 3334
    NeutronBar = -2112
    LambdaBar = -3122
    SigmaMinusBar = -3222
    Sigma0Bar = -3212
    SigmaPlusBar = -3112
    Xi0Bar = -3322
    XiPlusBar = -3312
    OmegaPlusBar = -3334
    DPlus = 411
    DMinus = -411
    D0 = 421
    D0Bar = -421
    DsPlus = 431
    DsMinusBar = -431
    LambdacPlus = 4122
    WPlus = 24
    WMinus = -24
    Z0 = 23
    NuE = 12
    NuEBar = -12
    NuMu = 14
    NuMuBar = -14
    TauPlus = -15
    TauMinus = 15
    NuTau = 16
    NuTauBar = -16

    # Nuclei
    He3Nucleus = 1000020030
    He4Nucleus = 1000020040
    Li6Nucleus = 1000030060
    Li7Nucleus = 1000030070
    Be9Nucleus = 1000040090
    B10Nucleus = 1000050100
    B11Nucleus = 1000050110
    C12Nucleus = 1000060120
    C13Nucleus = 1000060130
    N14Nucleus = 1000070140
    N15Nucleus = 1000070150
    O16Nucleus = 1000080160
    O17Nucleus = 1000080170
    O18Nucleus = 1000080180
    F19Nucleus = 1000090190
    Ne20Nucleus = 1000100200
    Ne21Nucleus = 1000100210
    Ne22Nucleus = 1000100220
    Na23Nucleus = 1000110230
    Mg24Nucleus = 1000120240
    Mg25Nucleus = 1000120250
    Mg26Nucleus = 1000120260
    Al26Nucleus = 1000130260
    Al27Nucleus = 1000130270
    Si28Nucleus = 1000140280
    Si29Nucleus = 1000140290
    Si30Nucleus = 1000140300
    Si31Nucleus = 1000140310
    Si32Nucleus = 1000140320
    P31Nucleus = 1000150310
    P32Nucleus = 1000150320
    P33Nucleus = 1000150330
    S32Nucleus = 1000160320
    S33Nucleus = 1000160330
    S34Nucleus = 1000160340
    S35Nucleus = 1000160350
    S36Nucleus = 1000160360
    Cl35Nucleus = 1000170350
    Cl36Nucleus = 1000170360
    Cl37Nucleus = 1000170370
    Ar36Nucleus = 1000180360
    Ar37Nucleus = 1000180370
    Ar38Nucleus = 1000180380
    Ar39Nucleus = 1000180390
    Ar40Nucleus = 1000180400
    Ar41Nucleus = 1000180410
    Ar42Nucleus = 1000180420
    K39Nucleus = 1000190390
    K40Nucleus = 1000190400
    K41Nucleus = 1000190410
    Ca40Nucleus = 1000200400
    Ca41Nucleus = 1000200410
    Ca42Nucleus = 1000200420
    Ca43Nucleus = 1000200430
    Ca44Nucleus = 1000200440
    Ca45Nucleus = 1000200450
    Ca46Nucleus = 1000200460
    Ca47Nucleus = 1000200470
    Ca48Nucleus = 1000200480
    Sc44Nucleus = 1000210440
    Sc45Nucleus = 1000210450
    Sc46Nucleus = 1000210460
    Sc47Nucleus = 1000210470
    Sc48Nucleus = 1000210480
    Ti44Nucleus = 1000220440
    Ti45Nucleus = 1000220450
    Ti46Nucleus = 1000220460
    Ti47Nucleus = 1000220470
    Ti48Nucleus = 1000220480
    Ti49Nucleus = 1000220490
    Ti50Nucleus = 1000220500
    V48Nucleus = 1000230480
    V49Nucleus = 1000230490
    V50Nucleus = 1000230500
    V51Nucleus = 1000230510
    Cr50Nucleus = 1000240500
    Cr51Nucleus = 1000240510
    Cr52Nucleus = 1000240520
    Cr53Nucleus = 1000240530
    Cr54Nucleus = 1000240540
    Mn52Nucleus = 1000250520
    Mn53Nucleus = 1000250530
    Mn54Nucleus = 1000250540
    Mn55Nucleus = 1000250550
    Fe54Nucleus = 1000260540
    Fe55Nucleus = 1000260550
    Fe56Nucleus = 1000260560
    Fe57Nucleus = 1000260570
    Fe58Nucleus = 1000260580

    # The following are fake particles used in Icetray and have no official codes
    # The section abs(code) > 2000000000 is reserved for this kind of use
    CherenkovPhoton = 2000009900
    Nu = -2000000004
    Monopole = -2000000041
    Brems = -2000001001
    DeltaE = -2000001002
    PairProd = -2000001003
    NuclInt = -2000001004
    MuPair = -2000001005
    Hadrons = -2000001006
    ContinuousEnergyLoss = -2000001111
    FiberLaser = -2000002100
    N2Laser = -2000002101
    YAGLaser = -2000002201
    STauPlus = -2000009131
    STauMinus = -2000009132
    SMPPlus = -2000009500
    SMPMinus = -2000009501


class ParticleShape(enum.IntEnum):
    """`I3Particle` property `shape`.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18
    """

    Null = 0
    Primary = 10
    TopShower = 20
    Cascade = 30
    CascadeSegment = 31
    InfiniteTrack = 40
    StartingTrack = 50
    StoppingTrack = 60
    ContainedTrack = 70
    MCTrack = 80
    Dark = 90


class PulseFlags(enum.IntEnum):
    """Pulse flags.

    Values corresponding with even powers of 2

        LC = 1
        ATWD = 2
        FADC = 4

    can be used to define a bit mask (i.e., multiple of {LC, ATWD, FADC} can be
    simultaneously true).

    If the LC bit is true, then the pulse comes from a hard local coincidence
    (HLC) hit; otherwise, the hits are soft local coincidence (SLC). .. ::

        is_hlc = (pulses["flags"] & PulseFlags.LC).astype(bool)
        is_slc = np.logical_not((pulses["flags"] & PulseFlags.LC).astype(bool))

    Scraped from dataclasses/public/dataclasses/physics/I3RecoPulse.h, 2020-02-18
    """

    # pylint: disable=invalid-name
    LC = 1
    ATWD = 2
    LC_ATWD = 3
    FADC = 4
    LC_FADC = 5
    ATWD_FADC = 6
    LC_ATWD_FADC = 7


class LocationType(enum.IntEnum):
    """`I3Particle` property `location`.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18
    """

    # pylint: disable=invalid-name
    Anywhere = 0
    IceTop = 10
    InIce = 20
    InActiveVolume = 30


class FitStatus(enum.IntEnum):
    """`I3Particle` property `fit_status`.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18
    """

    # pylint: disable=invalid-name
    NotSet = -1
    OK = 0
    PositiveLLH = 1  # NOT present in IceCube / icetray software
    Skipped = 2  # NOT present in IceCube / icetray software
    GeneralFailure = 10
    InsufficientHits = 20
    FailedToConverge = 30
    MissingSeed = 40
    InsufficientQuality = 50


class TriggerConfigID(enum.IntEnum):
    """Trigger common names mapped into TriggerConfigID (or config_id) in i3
    files.

    Note that this seems to be a really unique ID for a trigger, subsuming
    TriggerTypeID and SourceID.

    See docs at ::

      http://software.icecube.wisc.edu/documentation/projects/trigger-sim/trigger_config_ids.html

    script to dump enumerated values & details of each is at ::

      http://code.icecube.wisc.edu/svn/projects/trigger-sim/trunk/resources/scripts/print_trigger_configuration.py

    run via .. ::

      $I3_SRC/trigger-sim/resources/scripts/print_trigger_configuration.py -g GCDFILE

    """

    # I added "NONE" (code -1) since GCD file(s) were found with
    # TriggerKey.source == 40 (GLOBAL), TriggerKey.type == 30 (THROUGHPUT), and
    # TriggerKey.subtype == 0 (NO_SUBTYPE) have TriggerKey.config_id of None
    NONE = -1

    SMT8_IN_ICE = 1006
    SMT3_DeepCore = 1011
    SMT6_ICE_TOP = 102
    SLOP = 24002
    Cluster = 1007
    Cylinder = 21001

    # Added for 2016-2017
    Cylinder_ICE_TOP = 21002

    # Unique to 2011
    SLOP_2011 = 22005

    # Unique to IC79
    SMT3_DeepCore_IC79 = 1010


class TriggerTypeID(enum.IntEnum):
    """Trigger TypeID:  Enumeration describing what "algorithm" issued a
    trigger. More details about a specific trigger can be stored in the
    I3TriggerStatus maps as part of the detector status.

    See docs at ::

      http://software.icecube.wisc.edu/documentation/projects/trigger-sim/trigger_config_ids.html

    and enumerated values (and more comments on each type) are defined in ::

      http://code.icecube.wisc.edu/svn/projects/dataclasses/trunk/public/dataclasses/TriggerKey.h

    """

    SIMPLE_MULTIPLICITY = 0
    CALIBRATION = 10
    MIN_BIAS = 20
    THROUGHPUT = 30
    TWO_COINCIDENCE = 40
    THREE_COINCIDENCE = 50
    MERGED = 70
    SLOW_PARTICLE = 80
    FRAGMENT_MULTIPLICITY = 105
    STRING = 120
    VOLUME = 125
    SPHERE = 127
    UNBIASED = 129
    SPASE_2 = 170
    UNKNOWN_TYPE = 180


class TriggerSourceID(enum.IntEnum):
    """Trigger SourceID: Enumeration describing what "subdetector" issued a trigger.

    See docs at ::

      http://software.icecube.wisc.edu/documentation/projects/trigger-sim/trigger_config_ids.html

    and enumerated values are defined in ::

      http://code.icecube.wisc.edu/svn/projects/dataclasses/trunk/public/dataclasses/TriggerKey.h

    """

    IN_ICE = 0
    ICE_TOP = 10
    AMANDA_TWR_DAQ = 20
    EXTERNAL = 30
    GLOBAL = 40
    AMANDA_MUON_DAQ = 50
    SPASE = 70
    UNKNOWN_SOURCE = 80


class TriggerSubtypeID(enum.IntEnum):
    """Trigger SubtypeID: Enumeration describing how a software trigger was
    orginally "configured" within the TWR DAQ trigger system.

    Enumerated values are defined in ::

      http://code.icecube.wisc.edu/svn/projects/dataclasses/trunk/public/dataclasses/TriggerKey.h

    """

    # pylint: disable=invalid-name
    NO_SUBTYPE = 0
    M18 = 50
    M24 = 100
    T0 = 150
    LASER = 200
    UNKNOWN_SUBTYPE = 250


class I3WaveformSource(enum.IntEnum):
    """dataclasses/public/dataclasses/physics/I3Waveform.h"""

    ATWD = 0
    FADC = 1
    TWR_ELECTRICAL = 2
    TWR_OPTICAL = 3
    ETC = 4
    SLC = 5


class I3WaveformStatus(enum.IntEnum):
    """dataclasses/public/dataclasses/physics/I3Waveform.h"""

    VIRGINAL = 0
    # NB: 1 is sometimes used as a flag by other modules.
    COMBINED = 1 << 1
    SATURATED = 1 << 2
    UNDERSHOT = 1 << 3


class ExtractionError(enum.IntEnum):
    """Error codes that can be set by retro/i3processing/extract_events.py"""

    NO_ERROR = 0
    NU_CC_LEPTON_SECONDARY_MISSING = 1
    NU_NC_OUTOING_NU_MISSING = 2


class OMType(enum.IntEnum):
    """`OMType` enum

    Note this currently requires only uint8, i.e., [0, 255], for storage.

    Scraped from dataclasses/public/dataclasses/geometry/I3OMGeo.h, 2019-06-26
    SVN rev 167541
    """

    # pylint: disable=invalid-name
    UnknownType = 0
    AMANDA = 10
    IceCube = 20
    IceTop = 30
    Scintillator = 40
    IceAct = 50
    # OMType > 100 are Gen2 R&D optical modules
    PDOM = 110
    DEgg = 120
    mDOM = 130
    WOM = 140
    FOM = 150


class CableType(enum.IntEnum):
    """icecube.dataclasses.CableType"""

    # pylint: disable=invalid-name
    UnknownCableType = -1
    Terminated = 0
    Unterminated = 1


class DOMGain(enum.IntEnum):
    """icecube.dataclasses.DOMGain"""

    # pylint: disable=invalid-name
    UnknownGainType = -1
    High = 0
    Low = 1


class TrigMode(enum.IntEnum):
    """icecube.dataclasses.TrigMode"""

    # pylint: disable=invalid-name
    UnknownTrigMode = -1
    TestPattern = 0
    CPU = 1
    SPE = 2
    Flasher = 3
    MPE = 4


class LCMode(enum.IntEnum):
    """icecube.dataclasses.LCMode"""

    # pylint: disable=invalid-name
    UnknownLCMode = -1
    LCOff = 0
    UpOrDown = 1
    Up = 2
    Down = 3
    UpAndDown = 4
    SoftLC = 5


class ToroidType(enum.IntEnum):
    """icecube.dataclasses.ToroidType"""

    # pylint: disable=invalid-name
    OLD_TOROID = 0
    NEW_TOROID = 1


class CramerRaoStatus(enum.IntEnum):
    """icecube.recclasses.CramerRaoStatus; see
    recclasses/public/recclasses/CramerRaoParams.h"""

    # pylint: disable=invalid-name
    NotSet = -1
    OK = 0
    MissingInput = 10
    SingularMatrix = 20
    InsufficientHits = 30
    OneStringEvent = 40
    OtherProblems = 50
