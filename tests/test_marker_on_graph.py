"""Markers must lie on the displayed graph.

Invariant (root-caused 2026-08-23; regression introduced by commit 363a571):
every point drawn on a plot is a sample of the function that plot displays.
The optimum and both ZTF reference points must therefore be evaluated with the
SAME sidebar model that computes the surface, the day-line overlay, and the
slice curves — never with an aux model carrying a different schedule or
parameters.  A marker's authenticity as "the real ZTF mode" comes from loading
that mode's preset (which sets the sidebar sliders to the mode's schedule);
slider drift is communicated by dimming, not by letting the point leave the
surface.

Also pinned here: the display-floor gate — the surface masks cells below
ZMIN_DISPLAY_LOG10, so a marker whose rate falls below the floor is hidden
(None in the payload) rather than drawn floating over the masked hole.

Run with::

    .venv/Scripts/python -m pytest tests/test_marker_on_graph.py -v
"""
from __future__ import annotations

import math

import pytest

import standalone_bridge as sb
from grb_detect.core import ZMIN_DISPLAY_LOG10
from make_parity_snapshots import BASE, SNAP_GRIDS

GPC_TO_CM = 3.085677581491367e27

CONFIGS = {
    "optical_default":  {"optical_survey": True},
    "optical_drifted":  {"optical_survey": True, "f_eff": 0.55, "N_v": 3,
                         "dt_v_h": 1.0},
    "ztf_public_preset": {"optical_survey": True, "i_det": 2, "f_eff": 0.40,
                          "N_v": 2, "dt_v_h": 2.0, "t_overhead_s": 15.0},
    "exact_cuts_sched": {"optical_survey": True, "full_integral": True,
                         "N_v": 2, "dt_v_h": 2.0, "s_fade": 0.3, "s_rise": 0.5,
                         "win_tp": True},
    "fdec_override":    {"optical_survey": True, "N_v": 2, "dt_v_h": 2.0,
                         "F_dec_override_Jy": 0.02},
    "nonoptical":       {},
}

MARKERS = [
    ("opt", "N_opt", "t_cad_opt_s",
     ("R_opt", "t_exp_opt_s", "q_med_opt", "D_med_Gpc_opt")),
    ("ztf", "N_ztf", "t_cad_ztf_s",
     ("R_ztf", "t_exp_ztf_s", "q_med_ztf", "D_med_Gpc_ztf")),
    ("ztf_hc", "N_ztf_hc", "t_cad_ztf_hc_s",
     ("R_ztf_hc", "t_exp_ztf_hc_s", "q_med_ztf_hc", "D_med_Gpc_ztf_hc")),
]


@pytest.fixture(scope="module", autouse=True)
def _small_grids():
    saved = {k: getattr(sb, k) for k in SNAP_GRIDS}
    for k, v in SNAP_GRIDS.items():
        setattr(sb, k, v)
    yield
    for k, v in saved.items():
        setattr(sb, k, v)


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


@pytest.mark.parametrize("name,overrides", CONFIGS.items(),
                         ids=list(CONFIGS.keys()))
def test_markers_are_samples_of_the_displayed_model(name, overrides):
    """Payload marker values must equal the sidebar-model evaluation at the
    marker's coordinates — the on-graph invariant.  Fails the moment any
    marker is evaluated through a divergent path (e.g. an aux model)."""
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


def test_ztf_markers_present_at_defaults():
    """Sanity: at the optical defaults both ZTF points are drawn."""
    payload = sb.compute_all({**BASE, "optical_survey": True})
    assert payload["R_ztf"] is not None
    assert payload["R_ztf_hc"] is not None


def test_marker_follows_the_sidebar_schedule():
    """The marker moves WITH the surface when the schedule sliders move —
    guards against re-hard-coding a schedule into the marker evaluation."""
    p1 = {**BASE, "optical_survey": True}                      # N_v = 1
    p2 = {**BASE, "optical_survey": True, "N_v": 2, "dt_v_h": 2.0}
    R1 = sb.compute_all(p1)["R_ztf"]
    R2 = sb.compute_all(p2)["R_ztf"]
    assert R1 is not None and R2 is not None
    assert abs(R1 - R2) > 1e-6 * max(R1, R2), (
        "R_ztf did not respond to the sidebar N_v — marker evaluated on a "
        "model other than the displayed one?")


def test_display_floor_gate_hides_dim_markers():
    """A marker whose rate falls below the surface's display floor must be
    hidden, not drawn floating over the masked surface hole."""
    # A very shallow instrument: rates drop ~4 dex below the defaults.
    params = {**BASE, "optical_survey": True, "A_log": -2.0}
    payload = sb.compute_all(params)
    assert payload.get("error") is None, payload.get("error")

    R_raw = _reeval_on_sidebar_model(
        params, float(payload["N_ztf"]), float(payload["t_cad_ztf_s"]))[0]
    assert math.isfinite(R_raw) and R_raw > 0.0, (
        "fixture no longer produces a finite sub-floor rate — adjust A_log")
    assert math.log10(R_raw) < float(ZMIN_DISPLAY_LOG10), (
        "fixture rate not below the display floor — adjust A_log")
    assert payload["R_ztf"] is None, (
        "sub-floor marker leaked into the payload (would float over the "
        "masked surface)")
    assert payload["t_exp_ztf_s"] is None and payload["q_med_ztf"] is None
