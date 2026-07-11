"""Regression tests: every drawn point must carry finite hover extras.

The 3D-surface hover tooltip references per-point customdata (t_exp, q_med,
D_med). Plotly renders the raw ``%{customdata[i]}`` token literally when a
referenced cell is null, so any cell that is *drawn* (finite plotted rate) but
has NaN/None extras shows template garbage in the app.

Root cause of the historical bug: after model_night gained
``instrument.f_live = f_live / f_night``, ``compute_surface`` and
``_eval_point`` kept computing the extras with model_day on sub-day optical
cells — where model_day's ``t_exp = f_live * t_cad / N_exp - t_OH`` can be
<= 0 while model_night's rate is finite, yielding NaN medians on drawn cells.

These tests pin the invariant at three depths (engine grid, single-point
evaluator, full JSON payload) so any future change that reintroduces a
drawn-cell / NaN-extras mismatch fails here.

Run with::

    .venv/Scripts/python -m pytest tests/test_hover_customdata.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import standalone_bridge as sb
from grb_detect.constants import DAY_S
from grb_detect.core import compute_surface
from grb_detect.params import GPC_TO_CM


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

# UI default slider values (web/template.html), except t_overhead_s: a large
# t_OH is what makes model_day's t_exp go negative on the sub-day branch, so
# the historical failure mode is exercised.
def _optical_params(**overrides) -> dict:
    params = {
        "i_det": 10,
        "A_log": -4.68,
        "f_live": 0.2,
        "t_overhead_s": 20.0,
        "omega_exp_deg2": 47.0,
        "omega_srv_deg2": 27500.0,
        "t_night_h": 10.0,
        "optical_survey": True,
        "color_regimes": False,
        "full_integral": False,
        "qmin": 0.0,
        "Dmin_cm": 0.0,
        "s_fade": 0.0,
        "s_rise": 0.0,
        "s_mode": "discrete",
        "toh_approx": False,
        # Physics defaults (UI slider defaults)
        "p": 2.5,
        "nu_log10": 14.7,
        "E_kiso_log10": 53.0,
        "n0_log10": 0.0,
        "epsilon_e_log10": -1.0,
        "epsilon_B_log10": -2.0,
        "theta_j_rad": 0.1,
        "gamma0_log10": 2.5,
        "D_euc_gpc": 5.28,
        "rho_grb_log10": 2.415,
    }
    params.update(overrides)
    return params


@pytest.fixture(scope="module")
def state() -> dict:
    return sb._build_models(_optical_params())


# --------------------------------------------------------------------------- #
# 1. Engine grid invariant: finite Z_plot => finite extras                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "q_min,D_min_cm",
    [(0.0, 0.0), (1.0, 1.0 * GPC_TO_CM)],
    ids=["no-filters", "qmin-dmin-filters"],
)
def test_surface_drawn_cells_have_finite_extras(state, q_min, D_min_cm):
    """Every cell with a finite plotted rate must have finite t_exp / q_med /
    D_med — the exact property whose violation renders literal hover
    templates. Covers both cadence branches (model_night sub-day vs model_day
    multi-day) in optical-survey mode with t_OH > 0."""
    X, Y_s, Z_plot, Z_raw, rid, t_exp_g, q_med_g, D_med_Gpc_g = compute_surface(
        state["model_day"], state["model_night"], state["i_det"],
        optical_survey=True, color_regimes=False,
        t_night_s=state["t_night_s"],
        nx=60, ny=90,
        q_min=q_min, D_min_cm=D_min_cm,
    )
    drawn = np.isfinite(Z_plot)
    assert drawn.any(), "no drawn cells — fixture params no longer produce a surface"

    # The historical bug lived on the sub-day branch: make sure it is sampled.
    subday_drawn = drawn & (Y_s < DAY_S)
    assert subday_drawn.any(), "no drawn sub-day cells — invariant not exercised"

    for name, arr in [("t_exp", t_exp_g), ("q_med", q_med_g), ("D_med_Gpc", D_med_Gpc_g)]:
        bad = drawn & ~np.isfinite(arr)
        assert not bad.any(), (
            f"{bad.sum()} drawn cell(s) with non-finite {name} "
            f"(e.g. N_exp={X[bad][0]:.3g}, t_cad={Y_s[bad][0]:.3g} s) — "
            "these render literal %{customdata[...]} hover text"
        )


# --------------------------------------------------------------------------- #
# 2. Point evaluator: screenshot repro (sub-day optical, t_OH > 0)            #
# --------------------------------------------------------------------------- #


def test_eval_point_subday_extras_match_night_model(state):
    """Repro of the reported point (N_exp ~ 10, t_cad = 0.174 h): finite rate
    must come with finite extras, and t_exp must follow the *night* formula
    (f_live/f_night) * t_cad / N_exp - t_OH."""
    N_exp, t_cad_s = 10.0, 0.174 * 3600.0

    # Precondition that made the old code fail: model_day's t_exp is invalid
    # (t_exp_s returns NaN where f_live*t_cad/N_exp - t_OH <= 0) while the
    # sub-day rate (model_night) is finite.
    t_exp_day = float(state["model_day"].t_exp_s(np.array([N_exp]), np.array([t_cad_s]))[0])
    assert not math.isfinite(t_exp_day), (
        "fixture no longer reproduces the failure precondition "
        f"(model_day t_exp = {t_exp_day:.3g} s is valid) — adjust t_overhead_s/f_live"
    )

    R, t_exp, q_med, D_med_Gpc = sb._eval_point(
        N_exp, t_cad_s, state["i_det"],
        state["model_day"], state["model_night"],
        state["f_live"], state["f_live_night"], state["f_night"],
        optical_on=True, approx_on=False,
        t_overhead_s=state["t_overhead_s"],
        full_integral=False,
    )
    assert math.isfinite(R) and R > 0, f"rate not finite at repro point (R={R})"
    assert math.isfinite(t_exp), "t_exp is NaN on a point with finite rate"
    assert math.isfinite(q_med), "q_med is NaN on a point with finite rate"
    assert math.isfinite(D_med_Gpc), "D_med is NaN on a point with finite rate"

    expected_t_exp = state["f_live_night"] * t_cad_s / N_exp - state["t_overhead_s"]
    assert t_exp == pytest.approx(expected_t_exp, rel=1e-9), (
        "sub-day t_exp does not follow the night-model formula"
    )


# --------------------------------------------------------------------------- #
# 3. Full payload: no None extras wherever a value is drawn                   #
# --------------------------------------------------------------------------- #


def _assert_no_null_extras(drawn_key, extras_keys, payload):
    drawn_vals = payload[drawn_key]
    for name in extras_keys:
        extras = payload[name]
        assert len(extras) == len(drawn_vals), f"{name} length mismatch vs {drawn_key}"
        bad = [i for i, (z, e) in enumerate(zip(drawn_vals, extras))
               if z is not None and e is None]
        assert not bad, (
            f"{len(bad)} drawn point(s) in {drawn_key} with null {name} "
            f"(first at flat index {bad[0]}) — literal hover templates in the app"
        )


def test_compute_all_payload_extras_not_null_on_drawn_points():
    """End-to-end guard over the exact JSON the JS layer consumes: the surface,
    the discrete-day overlay, and the optimum marker scalars."""
    payload = sb.compute_all(_optical_params())

    _assert_no_null_extras(
        "Z_flat", ["t_exp_flat", "q_med_flat", "D_med_Gpc_flat"], payload,
    )
    if payload.get("day_line_shape", [0, 0])[0] > 0:
        _assert_no_null_extras(
            "day_line_R_flat",
            ["day_line_t_exp_flat", "day_line_q_med_flat", "day_line_D_med_Gpc_flat"],
            payload,
        )

    # Optimum marker scalars: a finite optimum rate must carry finite extras.
    if payload.get("R_opt") is not None:
        for key in ("t_exp_opt_s", "q_med_opt", "D_med_Gpc_opt"):
            assert payload.get(key) is not None, (
                f"optimum has finite R_opt but null {key}"
            )
