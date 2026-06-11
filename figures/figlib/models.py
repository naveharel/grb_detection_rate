"""Build engine models from presets and reproduce the app's strategy points.

This is the "reproduce the app/engine" layer. It mirrors how ``standalone_bridge.py``
wires a preset into ``make_rate_model`` so figures match the app, while staying a thin,
read-only consumer of the engine.

For full end-to-end app parity (whole surfaces, slices, the q/D views) a script can
also call ``standalone_bridge.compute_all(params)`` directly. Use this module for the
common case of building a model and locating named strategy points; use the
``overrides`` module to push past the public API.
"""

from __future__ import annotations

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.core import make_rate_model
from grb_detect.params import SurveyDesignParams

from . import presets as _presets


def build_model_from_preset(preset, *, toh_approx: bool = False, **physics_overrides):
    """Build the (day) ``DetectionRateModel`` for a ``SurveyPreset``.

    Mirrors the app's preset wiring: survey params from the preset, physics from the
    engine defaults unless overridden via ``physics_overrides`` (same kwargs as
    ``make_rate_model``: ``p, E_kiso_log10, n0_log10, epsilon_e_log10,
    epsilon_B_log10, theta_j_rad, gamma0_log10, nu_log10, D_euc_gpc, rho_grb_log10``).

    Returns the *day* model. This is correct for strategy points at cadence >= 1 day
    (e.g. the ZTF 2-day reference) and for strategy-surface figures. The app's sub-day
    optical branch (a separate night model + ``f_night`` factor) is intentionally not
    reproduced here; add a dedicated builder if a figure needs it.
    """
    design = SurveyDesignParams(omega_survey_max_sr=preset.omega_srv_deg2 * DEG2_TO_SR)
    t_oh = 0.0 if toh_approx else preset.t_overhead_s
    return make_rate_model(
        A_log=preset.A_log,
        f_live=preset.f_live,
        t_overhead_s=t_oh,
        omega_exp_deg2=preset.omega_exp_deg2,
        design=design,
        **physics_overrides,
    )


def n_exp_max(model) -> float:
    """Maximum number of exposures = survey footprint / single-exposure FOV."""
    return model.instrument.omega_survey_max_sr / model.instrument.omega_exp_sr


def ztf_strategy_point(model, *, i_det: int = _presets.ZTF.i_det):
    """(N_exp, t_cad_s, i_det) for the ZTF reference point, matching the bridge.

    The bridge uses ``N_ztf = min(omega_srv/omega_exp, N_exp_max)`` (which equals
    ``N_exp_max`` for the ZTF footprint) and a fixed cadence of 2 days.
    """
    return (n_exp_max(model), 2.0 * DAY_S, int(i_det))
