# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position


"""
Structured dtypes for `index` column and representing IceCube data
"""


from __future__ import absolute_import, division, print_function

__all__ = [
    "OMKEY_T",
    "I3POSITION_T",
    "I3DIRECTION_T",
    "I3OMGEO_T",
    "I3DOMCALIBRATION_T",
    "PULSE_T",
    "PHOTON_T",
    "HITS_SUMMARY_T",
    "EVT_HIT_INFO_T",
    "TRIGGER_T",
    "TRIGGERKEY_T",
    "I3TRIGGERREADOUTCONFIG_T",
    "I3TIME_T",
    "I3PARTICLEID_T",
    "I3PARTICLE_T",
    "FLAT_PARTICLE_T",
    "CRAMERRAOPARAMS_T",
    "DOMCALVERSION_T",
    "DSTPOSITION_T",
    "FLAT_I3DOMCALIBRATION_T",
    "FLAT_I3VEMCALIBRATION_T",
    "FLAT_PULSE_T",
    "FLAT_TRIGGER_T",
    "I3CLASTFITPARAMS_T",
    "I3CSCDLLHFITPARAMS_T",
    "I3DIPOLEFITPARAMS_T",
    "I3DIRECTHITSVALUES_T",
    "I3DOMSTATUS_T",
    "I3DST16_T",
    "I3EVENTHEADER_T",
    "I3FILLRATIOINFO_T",
    "I3FILTERRESULT_T",
    "I3FINITECUTS_T",
    "I3GENIERESULTDICT_SCALARS_T",
    "I3HITMULTIPLICITYVALUES_T",
    "I3HITSTATISTICSVALUES_T",
    "I3LINEFITPARAMS_T",
    "I3LOGLIKELIHOODFITPARAMS_T",
    "I3PORTIAEVENT_T",
    "I3RUSAGE_T",
    "I3SANTAFITPARAMS_T",
    "I3STARTSTOPPARAMS_T",
    "I3SUPERDSTTRIGGER_T",
    "I3TENSOROFINERTIAFITPARAMS_T",
    "I3TIMECHARACTERISTICSVALUES_T",
    "I3TIMEWINDOW_T",
    "I3TRACKCHARACTERISTICSVALUES_T",
    "I3VEMCALIBRATION_T",
    "I3VETO_T",
    "LINEARFIT_T",
    "SPECHARGEDISTRIBUTION_T",
    "START_STOP_T",
    "TAUPARAM_T",
]


import numpy as np


OMKEY_T = np.dtype([("string", np.int32), ("om", np.uint32), ("pmt", np.uint8),])
"""icetray/public/icetray/OMKey.h"""


I3POSITION_T = np.dtype([("x", np.float64), ("y", np.float64), ("z", np.float64),])
"""dataclasses/public/dataclasses/I3Position.h"""


I3DIRECTION_T = np.dtype([("zenith", np.float64), ("azimuth", np.float64),])
"""dataclasses/public/dataclasses/I3Direction.h"""


I3OMGEO_T = np.dtype(
    [
        ("position", I3POSITION_T),
        ("direction", I3DIRECTION_T),
        ("omtype", np.uint8),
        ("omkey", OMKEY_T),
        ("area", np.float64),
    ]
)
"""dataclasses/public/dataclasses/geometry/I3OMGeo.h"""


DOMCALVERSION_T = np.dtype([("major", np.int8), ("minor", np.int8), ("rev", np.int8),])


LINEARFIT_T = np.dtype([("slope", np.float64), ("intercept", np.float64),])
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


# TODO: not sure of names as they would appear in Python
# QUADRATICFIT_T = np.dtype(
#    [
#        ('quadFitA', np.float64),
#        ('quadFitB', np.float64),
#        ('quadFitC', np.float64),
#    ]
# )
# """dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


SPECHARGEDISTRIBUTION_T = np.dtype(
    [
        ("exp1_amp", np.float64),
        ("exp1_width", np.float64),
        ("exp2_amp", np.float64),
        ("exp2_width", np.float64),
        ("gaus_amp", np.float64),
        ("gaus_mean", np.float64),
        ("gaus_width", np.float64),
        ("compensation_factor", np.float64),
        ("slc_gaus_mean", np.float64),
        ("is_valid", np.bool8),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


TAUPARAM_T = np.dtype(
    [
        ("p0", np.float64),
        ("p1", np.float64),
        ("p2", np.float64),
        ("p3", np.float64),
        ("p4", np.float64),
        ("p5", np.float64),
        ("tau_frac", np.float64),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


I3DOMCALIBRATION_T = np.dtype(
    [
        # ('omkey', OMKEY_T),
        ("temperature", np.float64),  # Kelvin
        ("fadc_gain", np.float64),
        ("fadc_baseline_fit", LINEARFIT_T),
        ("fadc_beacon_baseline", np.float64),
        ("fadc_delta_t", np.float64),
        ("front_end_impedance", np.float64),
        ("tau_parameters", TAUPARAM_T),
        # where are ampGains_[3]?
        # ('atwd_gain', <icecube.dataclasses._atwd_gain_proxy>),
        # ('atwd_delta_t', <icecube.dataclasses._atwd_gain_proxy>),
        # ('atwd_freq_fit', <icecube.dataclasses._atwd_freq_fit_proxy>),
        # ('atwd_bin_calib_slope', <icecube.dataclasses._atwd_bin_calib_slope_proxy>),
        ("transit_time", LINEARFIT_T),
        ("hv_gain_fit", LINEARFIT_T),
        ("dom_cal_version", DOMCALVERSION_T),
        # ('atwd_beacon_baseline', <icecube.dataclasses._atwd_beacon_baseline_proxy>),
        ("spe_disc_calib", LINEARFIT_T),
        ("mpe_disc_calib", LINEARFIT_T),
        ("pmt_disc_calib", LINEARFIT_T),
        ("relative_dom_eff", np.float64),
        ("dom_noise_rate", np.float64),  # 1/ns
        ("dom_noise_thermal_rate", np.float64),
        ("dom_noise_decay_rate", np.float64),
        ("dom_noise_scintillation_mean", np.float64),
        ("dom_noise_scintillation_sigma", np.float64),
        ("dom_noise_scintillation_hits", np.float64),
        ("combined_spe_charge_distribution", SPECHARGEDISTRIBUTION_T),
        ("mean_atwd_charge", np.float64),
        ("mean_fadc_charge", np.float64),
        ("toroid_type", np.uint8),  # icecube.dataclasses.ToroidType.NEW_TOROID),
        ("is_mean_atwd_charge_valid", np.bool8),
        ("is_mean_fadc_charge_valid", np.bool8),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


FLAT_I3DOMCALIBRATION_T = np.dtype(
    [("omkey", OMKEY_T), ("dom_cal", I3DOMCALIBRATION_T),]
)


I3VEMCALIBRATION_T = np.dtype(
    [
        ("pe_per_vem", np.float64),
        ("mu_peak_width", np.float64),
        ("hglg_cross_over", np.float64),
        ("corr_factor", np.float64),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3VEMCalibration.h"""


FLAT_I3VEMCALIBRATION_T = np.dtype(
    [("omkey", OMKEY_T), ("vem_cal", I3VEMCALIBRATION_T),]
)


# TODO: I3CALIBRATION_T = np.dtype(
#     [
#         ('start_time', I3TIME_T),
#         ('end_time', I3TIME_T),
#         ('dom_cal', I3DOMCALIBRATIONMAP_T),
#         ('vem_cal', I3VEMCALIBRATIONMAP_T),
#     ]
# )
# """dataclasses/public/dataclasses/calibration/I3Calibration.h"""


I3DOMSTATUS_T = np.dtype(
    [
        ("omkey", OMKEY_T),
        ("cable_type", np.int8),  # icecube.dataclasses.CableType
        ("dac_fadc_ref", np.float64),
        ("dac_trigger_bias_0", np.float64),
        ("dac_trigger_bias_1", np.float64),
        ("delta_compress", np.int8),  # icecube.dataclasses.OnOff
        ("dom_gain_type", np.int8),  # icecube.dataclasses.DOMGain
        ("fe_pedestal", np.float64),
        # ('identity', <bound method I3DOMStatus.identity>) ... ???,
        ("lc_mode", np.int8),  # icecube.dataclasses.LCMode
        ("lc_span", np.uint32),
        ("lc_window_post", np.float64),
        ("lc_window_pre", np.float64),
        ("mpe_threshold", np.float64),
        ("n_bins_atwd_0", np.uint32),
        ("n_bins_atwd_1", np.uint32),
        ("n_bins_atwd_2", np.uint32),
        ("n_bins_atwd_3", np.uint32),
        ("n_bins_fadc", np.uint32),
        ("pmt_hv", np.float64),
        ("slc_active", np.bool8),
        ("spe_threshold", np.float64),
        ("status_atwd_a", np.int8),  # icecube.dataclasses.OnOff
        ("status_atwd_b", np.int8),  # icecube.dataclasses.OnOff
        ("status_fadc", np.int8),  # icecube.dataclasses.OnOff
        ("trig_mode", np.int8),  # icecube.dataclasses.TrigMode
        ("tx_mode", np.int8),  # icecube.dataclasses.LCMode
    ]
)


PULSE_T = np.dtype(
    [
        ("time", np.float64),
        ("charge", np.float32),
        ("width", np.float32),
        ("flags", np.uint8),  # icecube.dataclasses.I3RecoPulse.PulseFlags
    ]
)
"""dataclasses/public/dataclasses/physics/I3RecoPulse.h"""


FLAT_PULSE_T = np.dtype([("key", OMKEY_T), ("pulse", PULSE_T)])
"""Both omkey and pulse all in one (whereas in pulse series, omkey is a key and
value is a sequence of N pulses)"""


I3TIMEWINDOW_T = np.dtype([("start", np.float64), ("stop", np.float64)])
"""dataclasses/public/dataclasses/I3TimeWindow.h"""


PHOTON_T = np.dtype(
    [
        ("time", np.float32),
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("coszen", np.float32),
        ("azimuth", np.float32),
        ("wavelength", np.float32),
    ]
)


HITS_SUMMARY_T = np.dtype(
    [
        ("earliest_hit_time", np.float32),
        ("latest_hit_time", np.float32),
        ("average_hit_time", np.float32),
        ("total_charge", np.float32),
        ("num_hits", np.uint32),
        ("num_doms_hit", np.uint32),
        ("time_window_start", np.float32),
        ("time_window_stop", np.float32),
    ]
)


EVT_HIT_INFO_T = np.dtype(
    [("time", np.float32), ("charge", np.float32), ("event_dom_idx", np.uint32),]
)


TRIGGERKEY_T = np.dtype(
    [
        ("source", np.uint8),
        ("type", np.uint8),
        ("subtype", np.uint8),
        ("config_id", np.int32),
    ]
)
"""dataclasses/public/dataclasses/physics/TriggerKey.h"""


TRIGGER_T = np.dtype(
    [
        ("time", np.float32),
        ("length", np.float32),
        ("fired", np.bool),
        ("key", TRIGGERKEY_T),
    ]
)
"""dataclasses/public/dataclasses/physics/I3Trigger.h"""


FLAT_TRIGGER_T = np.dtype(
    [("level", np.uint8), ("parent_idx", np.int8), ("trigger", TRIGGER_T),]
)


I3TRIGGERREADOUTCONFIG_T = np.dtype(
    [
        ("readout_time_minus", np.float64),
        ("readout_time_plus", np.float64),
        ("readout_time_offset", np.float64),
    ]
)
"""icecube.dataclasses.I3TriggerReadoutConfig"""


I3TIME_T = np.dtype([("utc_year", np.int32), ("utc_daq_time", np.int64)])
"""I3Time is defined internally by the year and daqTime (tenths of ns since the
beginning of the year). See dataclasses/public/dataclasses/I3Time.h"""


I3EVENTHEADER_T = np.dtype(
    [
        ("run_id", np.uint32),
        ("sub_run_id", np.uint32),
        ("event_id", np.uint32),
        ("sub_event_id", np.uint32),
        ("sub_event_stream", np.dtype("S20")),
        ("state", np.uint8),
        ("start_time", I3TIME_T),
        ("end_time", I3TIME_T),
    ]
)
"""See: dataclasses/public/dataclasses/physics/I3EventHeader.h"""


I3RUSAGE_T = np.dtype(
    [
        ("SystemTime", np.float64),
        ("UserTime", np.float64),
        ("WallClockTime", np.float64),
    ]
)
"""icetray/public/icetray/I3PhysicsTimer.h"""


I3PARTICLEID_T = np.dtype([("majorID", np.uint64), ("minorID", np.int32),])
"""dataclasses/public/dataclasses/physics/I3ParticleID.h"""


I3PARTICLE_T = np.dtype(
    [
        ("id", I3PARTICLEID_T),
        ("pdg_encoding", np.int32),
        ("shape", np.uint8),
        ("pos", I3POSITION_T),
        ("dir", I3DIRECTION_T),
        ("time", np.float64),
        ("energy", np.float64),
        ("length", np.float64),
        ("speed", np.float64),
        ("fit_status", np.int8),
        ("location_type", np.uint8),
    ]
)
"""dataclasses/public/dataclasses/physics/I3Particle.h"""


I3SUPERDSTTRIGGER_T = np.dtype([("time", np.float64), ("length", np.float64)])
"""dataclasses/public/dataclasses/payload/I3SuperDSTTrigger.h"""


I3DIPOLEFITPARAMS_T = np.dtype(
    [
        ("Magnet", np.float64),
        ("MagnetX", np.float64),
        ("MagnetY", np.float64),
        ("MagnetZ", np.float64),
        ("AmpSum", np.float64),
        ("NHits", np.int32),
        ("NPairs", np.int32),
        ("MaxAmp", np.float64),
    ]
)
"""recclasses/public/recclasses/I3DipoleFitParams.h"""


FLAT_PARTICLE_T = np.dtype(
    [("level", np.uint8), ("parent_idx", np.int16), ("particle", I3PARTICLE_T),]
)


I3GENIERESULTDICT_SCALARS_T = np.dtype(
    [
        ("iev", np.int32),
        ("neu", np.int32),
        ("tgt", np.int32),
        ("Z", np.int32),
        ("A", np.int32),
        ("hitnuc", np.int32),
        ("hitqrk", np.int32),
        ("resid", np.bool8),
        ("sea", np.bool8),
        ("qel", np.bool8),
        ("res", np.bool8),
        ("dis", np.bool8),
        ("coh", np.bool8),
        ("dfr", np.bool8),
        ("imd", np.bool8),
        ("nuel", np.bool8),
        ("em", np.bool8),
        ("cc", np.bool8),
        ("nc", np.bool8),
        ("charm", np.bool8),
        ("neut_code", np.int32),
        ("nuance_code", np.int32),
        ("wght", np.float64),
        ("xs", np.float64),
        ("ys", np.float64),
        ("ts", np.float64),
        ("Q2s", np.float64),
        ("Ws", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("t", np.float64),
        ("Q2", np.float64),
        ("W", np.float64),
        ("Ev", np.float64),
        ("pxv", np.float64),
        ("pyv", np.float64),
        ("pzv", np.float64),
        ("En", np.float64),
        ("pxn", np.float64),
        ("pyn", np.float64),
        ("pzn", np.float64),
        ("pdgl", np.int32),
        ("El", np.float64),
        ("KEl", np.float64),
        ("pxl", np.float64),
        ("pyl", np.float64),
        ("pzl", np.float64),
        ("nfp", np.int32),
        ("nfn", np.int32),
        ("nfpip", np.int32),
        ("nfpim", np.int32),
        ("nfpi0", np.int32),
        ("nfkp", np.int32),
        ("nfkm", np.int32),
        ("nfk0", np.int32),
        ("nfem", np.int32),
        ("nfother", np.int32),
        ("nip", np.int32),
        ("nin", np.int32),
        ("nipip", np.int32),
        ("nipim", np.int32),
        ("nipi0", np.int32),
        ("nikp", np.int32),
        ("nikm", np.int32),
        ("nik0", np.int32),
        ("niem", np.int32),
        ("niother", np.int32),
        ("ni", np.int32),
        # Variable-length arrays are omitted
        # ("pdgi", <class 'icecube.dataclasses.ListInt'>),
        # ("resc", <class 'icecube.dataclasses.ListInt'>),
        # ("Ei", <class 'icecube.dataclasses.ListDouble'>),
        # ("pxi", <class 'icecube.dataclasses.ListDouble'>),
        # ("pyi", <class 'icecube.dataclasses.ListDouble'>),
        # ("pzi", <class 'icecube.dataclasses.ListDouble'>),
        ("nf", np.int32),
        # More variable-length arrays
        # ("pdgf", <class 'icecube.dataclasses.ListInt'>),
        # ("Ef", <class 'icecube.dataclasses.ListDouble'>),
        # ("KEf", <class 'icecube.dataclasses.ListDouble'>),
        # ("pxf", <class 'icecube.dataclasses.ListDouble'>),
        # ("pyf", <class 'icecube.dataclasses.ListDouble'>),
        # ("pzf", <class 'icecube.dataclasses.ListDouble'>),
        ("vtxx", np.float64),
        ("vtxy", np.float64),
        ("vtxz", np.float64),
        ("vtxt", np.float64),
        ("calresp0", np.float64),
        ("xsec", np.float64),
        ("diffxsec", np.float64),
        ("prob", np.float64),
        ("tgtmass", np.float64),
        # All remaining items ("_*" and "rw_*") appear in L5, L7 sim I3 files
        # but I can't find them in ConvertToGST.cxx...
        ("_azimax", np.float64),
        ("_azimin", np.float64),
        ("_elogmax", np.float64),
        ("_elogmin", np.float64),
        ("_glbprbscale", np.float64),
        ("_gvold", np.float64),
        ("_gvoll", np.float64),
        ("_gvolr", np.float64),
        ("_ngennu", np.float64),
        ("_plawind", np.float64),
        ("_zenmax", np.float64),
        ("_zenmin", np.float64),
        # GENIE systmatics exclusively contain 4 elements, ergo still "scalars"
        ("rw_AhtBY", (np.float64, 4)),
        ("rw_BhtBY", (np.float64, 4)),
        ("rw_CV1uBY", (np.float64, 4)),
        ("rw_CV2uBY", (np.float64, 4)),
        ("rw_MaCCQE", (np.float64, 4)),
        ("rw_MaCCRES", (np.float64, 4)),
        ("rw_MaCOHpi", (np.float64, 4)),
        ("rw_MaNCEL", (np.float64, 4)),
        ("rw_MaNCRES", (np.float64, 4)),
    ]
)
"""genie-icetray/private/genie-icetray/ConvertToGST.cxx
Reading from I3 files Python requires ``from icecube imoprt genie_icetray``"""


I3LINEFITPARAMS_T = np.dtype(
    [
        ("LFVel", np.float64),
        ("LFVelX", np.float64),
        ("LFVelY", np.float64),
        ("LFVelZ", np.float64),
        ("NHits", np.int32),
    ]
)
"""recclasses/public/recclasses/I3LineFitParams.h"""

I3FILLRATIOINFO_T = np.dtype(
    [
        ("mean_distance", np.float64),
        ("rms_distance", np.float64),
        ("nch_distance", np.float64),
        ("energy_distance", np.float64),
        ("fill_radius_from_rms", np.float64),
        ("fill_radius_from_mean", np.float64),
        ("fill_radius_from_mean_plus_rms", np.float64),
        ("fillradius_from_nch", np.float64),
        ("fill_radius_from_energy", np.float64),
        ("fill_ratio_from_rms", np.float64),
        ("fill_ratio_from_mean", np.float64),
        ("fill_ratio_from_mean_plus_rms", np.float64),
        ("fillratio_from_nch", np.float64),
        ("fill_ratio_from_energy", np.float64),
        ("hit_count", np.int32),
    ]
)
"""recclasses/public/recclasses/I3FillRatioInfo.h"""


I3FINITECUTS_T = np.dtype(
    [
        ("Length", np.float64),
        ("endFraction", np.float64),
        ("startFraction", np.float64),
        ("Sdet", np.float64),
        ("finiteCut", np.float64),
        ("DetectorLength", np.float64),
    ]
)
"""recclasses/public/recclasses/I3FiniteCuts.h"""


CRAMERRAOPARAMS_T = np.dtype(
    [
        ("cramer_rao_theta", np.float64),
        ("cramer_rao_phi", np.float64),
        ("variance_theta", np.float64),
        ("variance_phi", np.float64),
        ("variance_x", np.float64),
        ("variance_y", np.float64),
        ("covariance_theta_phi", np.float64),
        ("covariance_theta_x", np.float64),
        ("covariance_theta_y", np.float64),
        ("covariance_phi_x", np.float64),
        ("covariance_phi_y", np.float64),
        ("covariance_x_y", np.float64),
        # ('cramer_rao_theta_corr', nan, float),  # obsolete
        # ('cramer_rao_phi_corr', nan, float),  # obsolete
        # ('llh_est', nan, float),  # obsolete
        # ('rllh_est', nan, float),  # obsolete
        ("status", np.int8),  # enum CramerRaoStatus
    ]
)
"""recclasses/public/recclasses/CramerRaoParams.h"""


I3DIRECTHITSVALUES_T = np.dtype(
    [
        ("n_dir_strings", np.uint32),
        ("n_dir_doms", np.uint32),
        ("n_dir_pulses", np.uint64),
        ("q_dir_pulses", np.float64),
        ("n_early_strings", np.uint32),
        ("n_early_doms", np.uint32),
        ("n_early_pulses", np.uint64),
        ("q_early_pulses", np.float64),
        ("n_late_strings", np.uint32),
        ("n_late_doms", np.uint32),
        ("n_late_pulses", np.uint64),
        ("q_late_pulses", np.float64),
        ("dir_track_length", np.float64),
        ("dir_track_hit_distribution_smoothness", np.float64),
    ]
)
"""recclasses/public/recclasses/I3DirectHitsValues.h"""


I3HITSTATISTICSVALUES_T = np.dtype(
    [
        ("cog", I3POSITION_T),
        ("cog_z_sigma", np.float64),
        ("min_pulse_time", np.float64),
        ("max_pulse_time", np.float64),
        ("q_max_doms", np.float64),
        ("q_tot_pulses", np.float64),
        ("z_min", np.float64),
        ("z_max", np.float64),
        ("z_mean", np.float64),
        ("z_sigma", np.float64),
        ("z_travel", np.float64),
    ]
)
"""recclasses/public/recclasses/I3HitStatisticsValues.h"""


I3HITMULTIPLICITYVALUES_T = np.dtype(
    [
        ("n_hit_strings", np.uint32),
        ("n_hit_doms", np.uint32),
        ("n_hit_doms_one_pulse", np.uint32),
        ("n_pulses", np.uint64),
    ]
)
"""recclasses/public/recclasses/I3HitMultiplicityValues.h"""


I3CLASTFITPARAMS_T = I3TENSOROFINERTIAFITPARAMS_T = np.dtype(
    [
        ("mineval", np.float64),
        ("evalratio", np.float64),
        ("eval2", np.float64),
        ("eval3", np.float64),
    ]
)
"""recclasses/public/recclasses/I3TensorOfInertiaFitParams.h
and recclasses/public/recclasses/I3CLastFitParams.h"""


I3VETO_T = np.dtype(
    [
        ("nUnhitTopLayers", np.int16),
        ("nLayer", np.int16),
        ("earliestLayer", np.int16),
        ("earliestOM", np.int16),
        ("earliestContainment", np.int16),
        ("latestLayer", np.int16),
        ("latestOM", np.int16),
        ("latestContainment", np.int16),
        ("mostOuterLayer", np.int16),
        ("depthHighestHit", np.float64),
        ("depthFirstHit", np.float64),
        ("maxDomChargeLayer", np.int16),
        ("maxDomChargeString", np.int16),
        ("maxDomChargeOM", np.int16),
        ("nDomsBeforeMaxDOM", np.int16),
        ("maxDomChargeLayer_xy", np.int16),
        ("maxDomChargeLayer_z", np.int16),
        ("maxDomChargeContainment", np.int16),
    ]
)
"""recclasses/public/recclasses/I3Veto.h"""


I3CSCDLLHFITPARAMS_T = np.dtype(
    [
        ("HitCount", np.int32),
        ("HitOmCount", np.int32),
        ("UnhitOmCount", np.int32),
        ("Status", np.int32),
        ("ErrT", np.float64),
        ("ErrX", np.float64),
        ("ErrY", np.float64),
        ("ErrZ", np.float64),
        ("ErrTheta", np.float64),
        ("ErrPhi", np.float64),
        ("ErrEnergy", np.float64),
        ("NegLlh", np.float64),
        ("ReducedLlh", np.float64),
    ]
)
"""recclasses/public/recclasses/I3CscdLlhFitParams.h"""


I3PORTIAEVENT_T = np.dtype(
    [
        ("TotalBestNPE", np.float64),
        ("TotalAtwdNPE", np.float64),
        ("TotalFadcNPE", np.float64),
        ("TotalNch", np.int32),
        ("AtwdNch", np.int32),
        ("FadcNch", np.int32),
        ("TotalBestNPEbtw", np.float64),
        ("TotalAtwdNPEbtw", np.float64),
        ("TotalFadcNPEbtw", np.float64),
        ("TotalNchbtw", np.int32),
        ("AtwdNchbtw", np.int32),
        ("FadcNchbtw", np.int32),
        ("FirstPulseOMKey", OMKEY_T),
        ("LastPulseOMKey", OMKEY_T),
        ("LargestNPEOMKey", OMKEY_T),
        ("FirstPulseOMKeybtw", OMKEY_T),
        ("LastPulseOMKeybtw", OMKEY_T),
    ]
)
"""recclasses/public/recclasses/I3PortiaEvent.h"""


I3STARTSTOPPARAMS_T = np.dtype(
    [
        ("LLHStartingTrack", np.float64),
        ("LLHStoppingTrack", np.float64),
        ("LLHInfTrack", np.float64),
    ]
)
"""recclasses/public/recclasses/I3StartStopParams.h"""


I3TRACKCHARACTERISTICSVALUES_T = np.dtype(
    [
        ("avg_dom_dist_q_tot_dom", np.float64),
        ("empty_hits_track_length", np.float64),
        ("track_hits_separation_length", np.float64),
        ("track_hits_distribution_smoothness", np.float64),
    ]
)
"""recclasses/public/recclasses/I3TrackCharacteristicsValues.h"""


I3FILTERRESULT_T = np.dtype(
    [("condition_passed", np.bool8), ("prescale_passed", np.bool8)]
)
"""dataclasses/public/dataclasses/physics/I3FilterResult.h"""


DSTPOSITION_T = np.dtype([("x", np.int8), ("y", np.int8), ("z", np.int8)])
"""recclasses/public/recclasses/I3DST.h"""


I3DST16_T = np.dtype(
    [
        ("n_string", np.uint8),
        ("cog", DSTPOSITION_T),
        ("n_dir", np.uint8),
        ("ndom", np.uint16),  # `nchannel_` in .h file
        ("reco_label", np.uint8),
        ("time", np.uint64),
        ("trigger_tag", np.uint16),
    ]
)
"""recclasses/public/recclasses/I3DST16.h"""


I3SANTAFITPARAMS_T = np.dtype(
    [
        ("zc", np.float64),
        ("tc", np.float64),
        ("dc", np.float64),
        ("uz", np.float64),
        ("chi2", np.float64),
        ("chi2_simple", np.float64),
        ("zc_error", np.float64),
        ("tc_error", np.float64),
        ("dc_error", np.float64),
        ("uz_error", np.float64),
        ("dof", np.int32),
        ("string", np.int32),
        ("n_calls", np.int32),
        ("fit_time", np.float64),
        ("fit_status", np.int8),
        ("fit_type", np.int8),
        # ('zenith', nan, float),  # derived from uz: u_z = -cos(zenith)
    ]
)
"""oscNext_meta/trunk/santa/public/santa/I3SantaFitParams.h"""


I3TIMECHARACTERISTICSVALUES_T = np.dtype(
    [
        ("timelength_fwhm", np.float64),
        ("timelength_last_first", np.int32),
        ("timelength_maxgap", np.int32),
        ("zpattern", np.int32),
    ]
)
"""recclasses/public/recclasses/I3TimeCharacteristicsValues.h"""


I3LOGLIKELIHOODFITPARAMS_T = np.dtype(
    [
        ("logl", np.float64),
        ("rlogl", np.float64),
        ("ndof", np.int32),
        ("nmini", np.int32),
    ]
)
"""gulliver.I3LogLikelihoodFitParams"""


START_STOP_T = np.dtype([("start", np.uint64), ("stop", np.uint64),])
