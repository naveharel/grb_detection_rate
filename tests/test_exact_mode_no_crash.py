"""Enabling exact rate mode must never surface a Python error to the UI.

`web/app.js`'s runUpdate() renders `compute_all()`'s `error` field (or a raw
JS exception) as red text in #metric-status. `compute_all` wraps its entire
body in one try/except and returns {"error": traceback} on any exception, so
this test exercises exactly that path across the state space a real user can
reach from the sidebar: every combination of exact-mode + its 3 sub-toggles,
each built-in preset, optical on/off, and the filter sliders (q_min, D_min,
s_fade, s_rise) pushed to their real slider maxima (build_standalone.py
_SLIDERS) — the combination most likely to make `grid_search_optimum` raise
"All strategies ... unphysical (A0 is empty)".

Historical bug being guarded against: a red-text error appearing when turning
on "Exact rate mode" in the live app.

Run with::

    .venv/Scripts/python -m pytest tests/test_exact_mode_no_crash.py -v
"""
from __future__ import annotations

import pytest

import standalone_bridge as sb

GPC_TO_CM = 3.085677581491367e27

# compute_all's grid resolution has no params override hook (unlike nx/ny
# elsewhere) — shrink the module-level grid constants for this file's ~100
# exact-mode calls (the expensive branch) so the suite stays fast; full
# resolution is irrelevant to whether compute_all raises.
_SMALL_GRIDS = dict(NX_REGIME=48, NY_REGIME=60, NX_DEFAULT=40, NY_DEFAULT=50)


@pytest.fixture(scope="module", autouse=True)
def _small_grids():
    saved = {k: getattr(sb, k) for k in _SMALL_GRIDS}
    for k, v in _SMALL_GRIDS.items():
        setattr(sb, k, v)
    yield
    for k, v in saved.items():
        setattr(sb, k, v)


BASE: dict = {
    "i_det": 2,
    "A_log": -4.68,
    "f_live": 0.2,
    "t_overhead_s": 0.0,
    "omega_exp_deg2": 47.0,
    "omega_srv_deg2": 41253.0,
    "t_night_h": 10.0,
    "optical_survey": False,
    "color_regimes": False,
    "full_integral": False,
    "qmin": 0.0,
    "Dmin_cm": 0.0,
    "s_fade": 0.0,
    "s_rise": 0.0,
    "p": 2.2,
    "nu_log10": 14.7,
    "E_kiso_log10": 53.0,
    "n0_log10": 0.0,
    "epsilon_e_log10": -1.0,
    "epsilon_B_log10": -4.0,
    "theta_j_rad": 0.1,
    "gamma0_log10": 2.5,
    "D_euc_gpc": 4.55,
    "rho_grb_log10": 2.415,
}

# PRESETS (web/app.js).
PRESETS = {
    "default":    {},
    "ztf_public": {"optical_survey": True, "i_det": 2, "f_live": 0.08,
                   "A_log": -4.68, "omega_exp_deg2": 47.0, "t_overhead_s": 15.0},
    "ztf_hc":     {"optical_survey": True, "i_det": 6, "f_live": 0.17,
                   "A_log": -4.68, "omega_exp_deg2": 47.0, "t_overhead_s": 15.0},
    "rubin":      {"optical_survey": True, "i_det": 2, "f_live": 0.7,
                   "A_log": -7.0, "omega_exp_deg2": 9.6, "t_overhead_s": 30.0},
}

# Real slider maxima (build_standalone.py _SLIDERS) — the most aggressive
# filter combination a user can actually dial in from the sidebar.
EXTREME_FILTERS = {
    "qmin": 14.5,
    "Dmin_cm": 12.0 * GPC_TO_CM,
    "s_fade": 2.0,
    "s_rise": 2.0,
}


def _run(params) -> None:
    payload = sb.compute_all(params)
    assert payload.get("error") is None, (
        f"compute_all raised: {payload.get('error')}"
    )


@pytest.mark.parametrize("preset_name,preset", PRESETS.items(), ids=list(PRESETS.keys()))
@pytest.mark.parametrize("optical_on", [False, True], ids=["nonoptical", "optical"])
@pytest.mark.parametrize(
    "win_iminus1,win_tp,rise_rs,fade_rs",
    [(True, False, True, True), (False, True, False, False), (True, True, True, False)],
    ids=["subs-defaultish", "subs-alt-a", "subs-alt-b"],
)
def test_exact_mode_defaults_never_error(preset_name, preset, optical_on,
                                          win_iminus1, win_tp, rise_rs, fade_rs):
    """Exact mode + every sub-toggle combination, at default filter values,
    across every preset and optical on/off — the vanilla "just flip the
    switch" user flow."""
    params = {
        **BASE, **preset,
        "optical_survey": optical_on if not preset.get("optical_survey") else True,
        "full_integral": True,
        "win_iminus1": win_iminus1, "win_tp": win_tp,
        "rise_random_start": rise_rs, "fade_random_start": fade_rs,
    }
    _run(params)


@pytest.mark.parametrize("preset_name,preset", PRESETS.items(), ids=list(PRESETS.keys()))
@pytest.mark.parametrize("optical_on", [False, True], ids=["nonoptical", "optical"])
def test_exact_mode_with_extreme_filters_never_errors(preset_name, preset, optical_on):
    """Exact mode + all 3 sub-toggles + every filter slider pinned to its real
    maximum — the combination most likely to make the whole grid unphysical
    and hit grid_search_optimum's RuntimeError path."""
    params = {
        **BASE, **preset,
        "optical_survey": optical_on if not preset.get("optical_survey") else True,
        "full_integral": True,
        "win_iminus1": True, "win_tp": True,
        "rise_random_start": True, "fade_random_start": True,
        **EXTREME_FILTERS,
    }
    _run(params)


@pytest.mark.parametrize("preset_name,preset", PRESETS.items(), ids=list(PRESETS.keys()))
def test_toggling_exact_mode_on_and_off_never_errors(preset_name, preset):
    """Simulates the literal user action: load a preset, then click the
    exact-mode switch on, then off again."""
    base_params = {**BASE, **preset, "optical_survey": True}
    _run({**base_params, "full_integral": False})
    _run({**base_params, "full_integral": True,
          "win_iminus1": True, "win_tp": True,
          "rise_random_start": True, "fade_random_start": True})
    _run({**base_params, "full_integral": False})
