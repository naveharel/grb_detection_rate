"""Python mirror of the app's survey presets.

SINGLE SOURCE OF TRUTH: these values mirror ``PRESETS`` in ``web/app.js``. If you
change a preset there (or here), update both. Presets only set survey parameters;
the physics parameters use the engine defaults, exactly as the app does.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurveyPreset:
    """Survey-strategy parameters for a named instrument preset.

    Mirrors the fields the app preset sets: i, f_live, A_log, omega_exp, t_oh,
    omega_srv, optical. Physics is left at engine defaults.
    """

    key: str
    label: str               # display name for figures, e.g. "ZTF"
    i_det: int               # required detections per cadence cycle
    f_live: float            # live fraction
    A_log: float             # log10(F_lim_ref / Jy)
    omega_exp_deg2: float    # single-exposure field of view [deg^2]
    t_overhead_s: float      # per-exposure overhead [s]
    omega_srv_deg2: float    # total survey footprint [deg^2]
    optical: bool            # optical-survey mode (sub-night cadence) in the app
    s_fade: float = 0.0      # identification cut: min fading rate [mag/day]
    s_rise: float = 0.0      # identification cut: min rise rate [mag/day]


# Values verified against web/app.js PRESETS (2026-07 — split into the two
# real ZTF observing modes; i_det, f_live and footprints match the detected
# samples of Ho et al. 2022 / Andreoni et al. 2021). The rise/fade identification
# cuts default off (s_fade = s_rise = 0): they are free user knobs in the app, not
# part of a preset's identity.
ZTF_PUBLIC = SurveyPreset(
    key="ztf_public", label="ZTF public", i_det=2, f_live=0.08, A_log=-4.68,
    omega_exp_deg2=47.0, t_overhead_s=15.0, omega_srv_deg2=15000.0, optical=True,
)

ZTF_HC = SurveyPreset(
    key="ztf_hc", label="ZTF high-cadence", i_det=6, f_live=0.17, A_log=-4.68,
    omega_exp_deg2=47.0, t_overhead_s=15.0, omega_srv_deg2=2500.0, optical=True,
)

# Backwards-compatible alias: existing figures that referenced the single
# "ZTF" preset now get the public all-sky mode.
ZTF = ZTF_PUBLIC

RUBIN = SurveyPreset(
    key="rubin", label="Rubin", i_det=2, f_live=0.7, A_log=-7.0,
    omega_exp_deg2=9.6, t_overhead_s=30.0, omega_srv_deg2=18000.0, optical=True,
)

PRESETS = {p.key: p for p in (ZTF_PUBLIC, ZTF_HC, RUBIN)}
