"""Markers must lie on the displayed graph.

Invariant: every point drawn on the 3D plot (the grid optimum, and the ZTF
public / ZTF high-cadence reference points) is a *sample of the function that
plot displays* — the same shared sidebar model that computes the surface, not
some other model or a stale evaluation. A marker's authenticity as "the real
ZTF mode" comes from loading that mode's preset (which moves the sidebar
sliders to match); drift away from a preset is communicated by dimming the
marker (see `paramsMatchPreset()` in web/app.js), never by letting the point
drift off the surface it's drawn over.

Also pinned here: the display-floor gate. `compute_surface`'s grid masks any
cell with log10(R) below `ZMIN_DISPLAY_LOG10` (drawn as a gap). A marker whose
own rate falls below that floor must be hidden (None in the payload) rather
than drawn as a point floating over what is, for every other purpose, an empty
hole in the surface — this is exactly the "mode point doesn't line up with the
rest of the graph" symptom reported from manual testing (reproduced at
A_log=-2.0, a value inside the slider's real range: R_ztf's log10 ≈ -2.13,
just below the -2.0 floor).

Run with::

    .venv/Scripts/python -m pytest tests/test_marker_on_graph.py -v
"""
from __future__ import annotations

import math

import pytest

import standalone_bridge as sb
from grb_detect.core import ZMIN_DISPLAY_LOG10

GPC_TO_CM = 3.085677581491367e27

# compute_all's grid resolution has no params override hook — shrink the
# module-level grid constants so this file's many compute_all() calls stay
# fast; the marker-vs-surface invariant tested here doesn't depend on grid
# density (it re-evaluates the marker's own point directly, not off the grid).
_SMALL_GRIDS = dict(NX_REGIME=48, NY_REGIME=60, NX_DEFAULT=40, NY_DEFAULT=50)


@pytest.fixture(scope="module", autouse=True)
def _small_grids():
    saved = {k: getattr(sb, k) for k in _SMALL_GRIDS}
    for k, v in _SMALL_GRIDS.items():
        setattr(sb, k, v)
    yield
    for k, v in saved.items():
        setattr(sb, k, v)


# UI slider defaults (see build_standalone.py _SLIDERS / web/template.html).
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
    "N_v": 1,
    "dt_v_h": 2.0,
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

CONFIGS = {
    "nonoptical_default":  {},
    "optical_default":     {"optical_survey": True},
    "optical_drifted":     {"optical_survey": True, "f_live": 0.55, "N_v": 3,
                            "dt_v_h": 1.0},
    "ztf_public_preset":   {"optical_survey": True, "i_det": 2, "f_live": 0.08,
                            "N_v": 2, "dt_v_h": 2.0, "A_log": -4.68,
                            "omega_exp_deg2": 47.0, "t_overhead_s": 15.0},
    "ztf_hc_preset":       {"optical_survey": True, "i_det": 2, "f_live": 0.17,
                            "N_v": 6, "dt_v_h": 1.0, "A_log": -4.68,
                            "omega_exp_deg2": 47.0, "t_overhead_s": 15.0},
    "rubin_preset":        {"optical_survey": True, "i_det": 2, "f_live": 0.7,
                            "N_v": 1, "dt_v_h": 2.0, "A_log": -7.0,
                            "omega_exp_deg2": 9.6, "t_overhead_s": 30.0},
    "exact_mode":          {"optical_survey": True, "full_integral": True,
                            "N_v": 2, "dt_v_h": 2.0, "s_fade": 0.3, "s_rise": 0.5},
    "filters":             {"qmin": 1.0, "Dmin_cm": 1.0 * GPC_TO_CM},
}

MARKERS = [
    ("opt",    "N_opt",    "t_cad_opt_s",
     ("R_opt", "t_exp_opt_s", "q_med_opt", "D_med_Gpc_opt")),
    ("ztf",    "N_ztf",    "t_cad_ztf_s",
     ("R_ztf", "t_exp_ztf_s", "q_med_ztf", "D_med_Gpc_ztf")),
    ("ztf_hc", "N_ztf_hc", "t_cad_ztf_hc_s",
     ("R_ztf_hc", "t_exp_ztf_hc_s", "q_med_ztf_hc", "D_med_Gpc_ztf_hc")),
]


def _reeval_on_sidebar_model(params: dict, N: float, t_cad_s: float):
    """The marker point as the displayed (sidebar) model gives it."""
    state = sb._build_models(params)
    return sb._eval_point(
        N, t_cad_s, state["i_det"], state["model"],
        state["toh_approx"], state["t_overhead_s"],
        full_integral=state["full_on"],
        q_min=state["q_min"], D_min_cm=state["D_min_cm"],
        s_fade=state["s_fade"], s_rise=state["s_rise"], s_mode=state["s_mode"],
        rise_random_start=state["rise_random_start"],
        fade_random_start=state["fade_random_start"],
    )


def _apply_display_gate(point):
    R = point[0]
    if not (math.isfinite(R) and R > 0.0
            and math.log10(R) >= float(ZMIN_DISPLAY_LOG10)):
        return (math.nan,) * len(point)
    return point


@pytest.mark.parametrize("name,overrides", CONFIGS.items(), ids=list(CONFIGS.keys()))
def test_markers_are_samples_of_the_displayed_model(name, overrides):
    """Payload marker values must equal the sidebar-model evaluation at the
    marker's coordinates — the on-graph invariant. Fails the moment any
    marker is evaluated through a divergent path (e.g. an aux model), or
    when a sub-floor marker leaks through without being hidden."""
    params = {**BASE, **overrides}
    payload = sb.compute_all(params)
    assert payload.get("error") is None, payload.get("error")

    checked = 0
    for label, N_key, t_key, value_keys in MARKERS:
        N_m = payload[N_key]
        t_m = payload[t_key]
        if N_m is None or t_m is None:
            continue
        expected = _apply_display_gate(
            _reeval_on_sidebar_model(params, float(N_m), float(t_m)))
        for key, exp in zip(value_keys, expected):
            got = payload[key]
            if math.isnan(exp):
                assert got is None, (
                    f"[{name}] {label}: {key} should be hidden (None), "
                    f"payload has {got!r}")
            else:
                assert got is not None, (
                    f"[{name}] {label}: {key} is None but the displayed "
                    f"model gives {exp!r}")
                assert got == pytest.approx(exp, rel=1e-12), (
                    f"[{name}] {label}: {key} = {got!r} differs from the "
                    f"displayed model's {exp!r} — marker off the graph")
        checked += 1
    assert checked >= 2, f"[{name}] too few markers checked ({checked})"


def test_ztf_markers_present_at_optical_defaults():
    """Sanity: at the optical defaults both ZTF points are drawn."""
    payload = sb.compute_all({**BASE, "optical_survey": True})
    assert payload["R_ztf"] is not None
    assert payload["R_ztf_hc"] is not None


def test_marker_follows_the_sidebar_schedule():
    """The marker moves WITH the surface when the schedule sliders move —
    guards against re-hard-coding a schedule into the marker evaluation."""
    p1 = {**BASE, "optical_survey": True}                       # N_v = 1
    p2 = {**BASE, "optical_survey": True, "N_v": 2, "dt_v_h": 2.0}
    R1 = sb.compute_all(p1)["R_ztf"]
    R2 = sb.compute_all(p2)["R_ztf"]
    assert R1 is not None and R2 is not None
    assert abs(R1 - R2) > 1e-6 * max(R1, R2), (
        "R_ztf did not respond to the sidebar N_v — marker evaluated on a "
        "model other than the displayed one?"
    )


def test_display_floor_gate_hides_dim_markers():
    """A marker whose rate falls below the surface's display floor must be
    hidden, not drawn floating over the masked surface hole.

    Reproduces the live bug found on this branch: A_log=-2.0 (well inside the
    slider's [-12, -2] range) put log10(R_ztf) just under ZMIN_DISPLAY_LOG10,
    yet R_ztf came back as a real float before the fix.
    """
    params = {**BASE, "optical_survey": True, "A_log": -2.0}
    payload = sb.compute_all(params)
    assert payload.get("error") is None, payload.get("error")

    # N_ztf/t_cad_ztf_s are always present (real footprint/cadence) even when
    # the resulting rate is gated to None — only the value fields are hidden.
    R_raw = _reeval_on_sidebar_model(
        params, float(payload["N_ztf"]), float(payload["t_cad_ztf_s"]))[0]
    assert math.isfinite(R_raw) and R_raw > 0.0, (
        "fixture no longer produces a finite sub-floor rate — adjust A_log"
    )
    assert math.log10(R_raw) < float(ZMIN_DISPLAY_LOG10), (
        "fixture rate not below the display floor — adjust A_log"
    )
    assert payload["R_ztf"] is None, (
        "sub-floor marker leaked into the payload (would float over the "
        "masked surface)"
    )
    assert payload["t_exp_ztf_s"] is None and payload["q_med_ztf"] is None
