"""Regression tests: every drawn point must carry finite hover extras.

The 3D-surface hover tooltip references per-point customdata (t_exp, q_med,
D_med). Plotly renders the raw ``%{customdata[i]}`` token literally when a
referenced cell is null, so any cell that is *drawn* (finite plotted rate) but
has NaN/None extras shows template garbage in the app.

Root cause of the historical bug (pre-schedule two-model era): the extras
were computed with a differently-parameterized model than the rate, so cells
whose rate was finite could carry NaN medians.  The single-model schedule
architecture removes the dispatch, but the failure class survives wherever
the budget can go negative on drawn-adjacent cells — e.g. at N_v > 1 the
budget t_exp = f_live·t_cad/(N_exp·N_v) − t_OH turns negative at N_v-times
smaller N_exp, so the invariant is exercised with both N_v = 1 and N_v = 6.

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
    "q_min,D_min_cm,N_v,dt_v_h",
    [
        (0.0, 0.0, 1, 2.0),
        (1.0, 1.0 * GPC_TO_CM, 1, 2.0),
        (0.0, 0.0, 6, 1.5),
    ],
    ids=["no-filters", "qmin-dmin-filters", "schedule-Nv6"],
)
def test_surface_drawn_cells_have_finite_extras(q_min, D_min_cm, N_v, dt_v_h):
    """Every cell with a finite plotted rate must have finite t_exp / q_med /
    D_med — the exact property whose violation renders literal hover
    templates. Exercised with t_OH > 0 at N_v = 1 and N_v = 6 (the schedule
    divides the budget, moving the t_exp <= 0 boundary to smaller N_exp)."""
    st = sb._build_models(_optical_params(N_v=N_v, dt_v_h=dt_v_h))
    X, Y_s, Z_plot, Z_raw, rid, t_exp_g, q_med_g, D_med_Gpc_g = compute_surface(
        st["model"], st["i_det"],
        optical_survey=True, color_regimes=False,
        t_night_s=st["t_night_s"],
        nx=60, ny=90,
        q_min=q_min, D_min_cm=D_min_cm,
    )
    drawn = np.isfinite(Z_plot)
    assert drawn.any(), "no drawn cells — fixture params no longer produce a surface"

    for name, arr in [("t_exp", t_exp_g), ("q_med", q_med_g), ("D_med_Gpc", D_med_Gpc_g)]:
        bad = drawn & ~np.isfinite(arr)
        assert not bad.any(), (
            f"{bad.sum()} drawn cell(s) with non-finite {name} "
            f"(e.g. N_exp={X[bad][0]:.3g}, t_cad={Y_s[bad][0]:.3g} s) — "
            "these render literal %{customdata[...]} hover text"
        )


# --------------------------------------------------------------------------- #
# 2. Point evaluator: budget boundary under the N_v > 1 schedule              #
# --------------------------------------------------------------------------- #


def test_eval_point_schedule_budget_boundary():
    """At N_v = 6 the budget t_exp = f_live·t_cad/(N_exp·N_v) − t_OH turns
    negative at 6× smaller N_exp: a point valid at N_v = 1 must return the
    all-NaN tuple at N_v = 6 (no finite rate with NaN extras), and a valid
    schedule point's t_exp must follow the schedule formula."""
    st1 = sb._build_models(_optical_params())
    st6 = sb._build_models(_optical_params(N_v=6, dt_v_h=1.5))
    t_cad_s = 2.0 * DAY_S

    # Pick N_exp so that the N_v=1 budget is positive but the N_v=6 one is not.
    f_live, t_oh = st1["f_live"], st1["t_overhead_s"]
    N_boundary_1 = f_live * t_cad_s / t_oh
    N_exp = N_boundary_1 / 3.0          # valid at N_v=1, invalid at N_v=6

    common = dict(full_integral=False)
    R1, t_exp1, q1, d1 = sb._eval_point(
        N_exp, t_cad_s, st1["i_det"], st1["model"], False, t_oh, **common)
    assert math.isfinite(R1) and math.isfinite(t_exp1) and math.isfinite(q1)
    assert t_exp1 == pytest.approx(f_live * t_cad_s / N_exp - t_oh, rel=1e-9)

    R6, t_exp6, q6, d6 = sb._eval_point(
        N_exp, t_cad_s, st6["i_det"], st6["model"], False, t_oh, **common)
    assert not math.isfinite(R6), "negative-budget schedule point returned a rate"
    assert not math.isfinite(t_exp6) and not math.isfinite(q6) and not math.isfinite(d6)

    # A valid N_v=6 point follows the schedule budget formula.
    N_ok = N_boundary_1 / 20.0
    R6b, t_exp6b, q6b, d6b = sb._eval_point(
        N_ok, t_cad_s, st6["i_det"], st6["model"], False, t_oh, **common)
    assert math.isfinite(R6b) and math.isfinite(q6b) and math.isfinite(d6b)
    assert t_exp6b == pytest.approx(f_live * t_cad_s / (N_ok * 6) - t_oh, rel=1e-9)


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
