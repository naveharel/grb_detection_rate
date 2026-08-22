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

    Mirrors the fields the app preset sets: i, f_eff, n_v, dt_v_h, A_log,
    omega_exp, t_oh, plus the figures-side omega_srv / optical. Physics is
    left at engine defaults.

    ``f_eff`` is the usable fraction of the night window (weather, moon,
    engineering, time share); the wall-clock live fraction the engine consumes
    is f_live = f_eff * f_night with f_night = t_night / 24 h.  ``n_v`` and
    ``dt_v_h`` are the night schedule (visits per visit-night and their
    spacing); the identification requirement ``i_det`` is a pipeline property,
    decoupled from n_v.
    """

    key: str
    label: str               # display name for figures, e.g. "ZTF"
    i_det: int               # required detection epochs (pipeline property)
    f_eff: float             # usable fraction of the night window
    n_v: int                 # visits per visit-night (night schedule)
    dt_v_h: float            # intra-night visit spacing [hr]
    A_log: float             # log10(F_lim_ref / Jy)
    omega_exp_deg2: float    # single-exposure field of view [deg^2]
    t_overhead_s: float      # per-exposure overhead [s]
    omega_srv_deg2: float    # total survey footprint [deg^2]
    optical: bool            # optical-survey mode (integer-day cadences) in the app
    s_fade: float = 0.0      # identification cut: min fading rate [mag/day]
    s_rise: float = 0.0      # identification cut: min rise rate [mag/day]


# Values verified against web/app.js PRESETS (2026-08 — night-schedule
# parameterization: i decoupled from the visits offered, f_live replaced by
# f_eff; both ZTF modes take ~319 exposures/night on the same telescope, so
# both give f_eff ~= 0.40, matching ZTF's ~40% public time share). The
# rise/fade identification cuts default off (s_fade = s_rise = 0): they are
# free user knobs in the app, not part of a preset's identity.
ZTF_PUBLIC = SurveyPreset(
    key="ztf_public", label="ZTF public", i_det=2, f_eff=0.40, n_v=2,
    dt_v_h=2.0, A_log=-4.68, omega_exp_deg2=47.0, t_overhead_s=15.0,
    omega_srv_deg2=15000.0, optical=True,
)

ZTF_HC = SurveyPreset(
    key="ztf_hc", label="ZTF high-cadence", i_det=2, f_eff=0.40, n_v=6,
    dt_v_h=1.5, A_log=-4.68, omega_exp_deg2=47.0, t_overhead_s=15.0,
    omega_srv_deg2=2500.0, optical=True,
)

# Backwards-compatible alias: existing figures that referenced the single
# "ZTF" preset now get the public all-sky mode.
ZTF = ZTF_PUBLIC

RUBIN = SurveyPreset(
    key="rubin", label="Rubin", i_det=2, f_eff=0.7, n_v=2, dt_v_h=0.5,
    A_log=-7.0, omega_exp_deg2=9.6, t_overhead_s=30.0,
    omega_srv_deg2=18000.0, optical=True,
)

PRESETS = {p.key: p for p in (ZTF_PUBLIC, ZTF_HC, RUBIN)}
