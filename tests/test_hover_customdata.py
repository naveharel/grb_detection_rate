"""Regression tests: every drawn point must carry finite hover extras.

The 3D-surface hover tooltip references per-point customdata (t_exp, q_med,
D_med). Plotly renders the raw ``%{customdata[i]}`` token literally when a
referenced cell is null, so any cell that is *drawn* (finite plotted rate) but
has NaN/None extras shows template garbage in the app.

Historical bug (pre dtv-window): with the old dual model_day/model_night
architecture, after model_night gained ``instrument.f_live = f_live /
f_night``, ``compute_surface`` and ``_eval_point`` could compute the extras
with the wrong model on sub-day optical cells, yielding NaN medians on drawn
cells. Optical mode is now discrete-day-only and a single model serves the
whole surface, which makes that specific mismatch structurally impossible —
but the same class of bug can reappear wherever a *different* model is used
for a marker's rate vs its extras, which is exactly the situation for the
ZTF public/HC markers (each evaluated on its own aux model carrying its own
N_v/dt_v_s schedule; see `compute_all`).

These tests pin the invariant (finite rate ⇒ finite extras, from the same
model) at three depths (engine grid, single-point evaluator, full JSON
payload) so any future change that reintroduces a drawn-cell / NaN-extras
mismatch fails here.

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
from grb_detect.detection_rate import DetectionRateModel
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
    templates. Optical-survey mode (discrete day-multiple cadences only),
    single model, t_OH > 0."""
    X, Y_s, Z_plot, Z_raw, rid, t_exp_g, q_med_g, D_med_Gpc_g = compute_surface(
        state["model"], state["i_det"],
        optical_survey=True, color_regimes=False,
        t_night_s=state["t_night_s"],
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
# 2. Point evaluator: a ZTF marker's aux model (its own N_v/dt_v_s)           #
# --------------------------------------------------------------------------- #


def test_eval_point_aux_model_extras_match_own_t_exp(state):
    """A marker evaluated on its own aux model (N_v, dt_v_s baked in at
    construction — see compute_all's model_ztf_public/model_ztf_hc) must get
    its rate, t_exp and medians from that SAME aux model — the modern
    incarnation of the hover-consistency invariant, now guarding the
    aux-model divergence point instead of the retired model_day/model_night
    one."""
    model = state["model"]
    N_v, dt_v_s = 6, 1.0 * 3600.0  # ZTF HC-style schedule
    aux = DetectionRateModel(
        phys=model.phys, instrument=model.instrument, micro=model.micro,
        pls=model.pls, win_i_minus_one=model.win_i_minus_one,
        win_from_peak=model.win_from_peak, N_v=N_v, dt_v_s=dt_v_s,
    )
    N_exp, t_cad_s = 50.0, float(DAY_S)  # nightly revisit, HC-style

    R, t_exp, q_med, D_med_Gpc = sb._eval_point(
        N_exp, t_cad_s, state["i_det"], aux,
        approx_on=False, t_overhead_s=state["t_overhead_s"],
        full_integral=False,
    )
    assert math.isfinite(R) and R > 0, f"rate not finite at repro point (R={R})"
    assert math.isfinite(t_exp), "t_exp is NaN on a point with finite rate"
    assert math.isfinite(q_med), "q_med is NaN on a point with finite rate"
    assert math.isfinite(D_med_Gpc), "D_med is NaN on a point with finite rate"

    expected_t_exp = float(aux.t_exp_s(np.array([N_exp]), np.array([t_cad_s]))[0])
    assert t_exp == pytest.approx(expected_t_exp, rel=1e-9), (
        "t_exp does not follow the aux model's own (N_v-aware) budget formula"
    )
    # N_v must actually be affecting the budget (not silently ignored).
    t_exp_no_schedule = float(model.t_exp_s(np.array([N_exp]), np.array([t_cad_s]))[0])
    assert t_exp != pytest.approx(t_exp_no_schedule, rel=1e-6), (
        "aux model's N_v is not affecting t_exp — schedule not applied"
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
