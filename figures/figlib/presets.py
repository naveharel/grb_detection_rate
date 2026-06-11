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


# Values verified against web/app.js PRESETS (2026-05).
ZTF = SurveyPreset(
    key="ztf", label="ZTF", i_det=10, f_live=0.2, A_log=-4.68,
    omega_exp_deg2=47.0, t_overhead_s=15.0, omega_srv_deg2=27500.0, optical=True,
)

RUBIN = SurveyPreset(
    key="rubin", label="Rubin", i_det=10, f_live=0.7, A_log=-7.0,
    omega_exp_deg2=9.6, t_overhead_s=30.0, omega_srv_deg2=18000.0, optical=True,
)

PRESETS = {p.key: p for p in (ZTF, RUBIN)}
