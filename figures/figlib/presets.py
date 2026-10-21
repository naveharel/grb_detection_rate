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
    omega_srv_deg2: float    # footprint per observing window [deg^2]
    optical: bool            # optical-survey mode (sub-night cadence) in the app
    s_fade: float = 0.0      # identification cut: min fading rate [mag/day]
    s_rise: float = 0.0      # identification cut: min rise rate [mag/day]
    t_night_h: float | None = None  # effective window set by the ZTF presets
    visits_per_night: int | None = None


# Observing-block defaults (2026-10-21), mirrored from web/app.js. Both require
# two detections. Public alternates halves of its full 15,000 deg^2 footprint;
# no additional alternate-night multiplier is applied to its 7,500 deg^2 area.
# Rise/fade cuts remain free user choices rather than preset identity fields.
ZTF_PUBLIC = SurveyPreset(
    key="ztf_public", label="ZTF public", i_det=2, f_live=0.166, A_log=-4.68,
    omega_exp_deg2=47.0, t_overhead_s=15.0, omega_srv_deg2=7500.0, optical=True,
    t_night_h=4.0, visits_per_night=2,
)

ZTF_HC = SurveyPreset(
    key="ztf_hc", label="ZTF high-cadence", i_det=2, f_live=0.166, A_log=-4.68,
    omega_exp_deg2=47.0, t_overhead_s=15.0, omega_srv_deg2=2500.0, optical=True,
    t_night_h=6.0, visits_per_night=6,
)

# Historical two-day configuration for existing paper figures. Keep it explicit
# so updating app presets cannot silently change their scientific setup.
ZTF_PUBLIC_LEGACY = SurveyPreset(
    key="ztf_public_legacy", label="ZTF public (legacy)", i_det=2, f_live=0.08,
    A_log=-4.68, omega_exp_deg2=47.0, t_overhead_s=15.0,
    omega_srv_deg2=15000.0, optical=True,
)

# Alias for the current public preset; historical figures use the named legacy.
ZTF = ZTF_PUBLIC

RUBIN = SurveyPreset(
    key="rubin", label="Rubin", i_det=2, f_live=0.7, A_log=-7.0,
    omega_exp_deg2=9.6, t_overhead_s=30.0, omega_srv_deg2=18000.0, optical=True,
)

PRESETS = {p.key: p for p in (ZTF_PUBLIC, ZTF_HC, RUBIN)}
