"""Verification tests for the R(q) and R(D) view bridge.

Covers:
  1. Cumulative endpoint matches `rate_log10` / `rate_log10_full_integral`
     under the same cross-filter.
  2. Differential trapezoidally integrates back to the cumulative at x=0
     (within ~1%).  This is the load-bearing test for the (3/2) factor in
     dR/dD and the dominant-term derivative formula for dR/dq.
  3. t_OH-approximation invalidity returns an all-None payload.

Run with::

    .venv/Scripts/python -m pytest tests/test_qd_views.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import standalone_bridge as bridge
from grb_detect.detection_rate import DetectionRateModel
from grb_detect.params import (
    AfterglowPhysicalParams,
    GPC_TO_CM,
    SurveyInstrumentParams,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _default_params() -> dict:
    """Mirror the dict shape JS sends to compute_all / compute_qdview."""
    return dict(
        i_det=10,
        A_log=-4.68,
        f_live=0.2,
        t_overhead_s=0.0,
        omega_exp_deg2=47.0,
        omega_srv_deg2=27500.0,
        t_night_h=10.0,
        p=2.5,
        nu_log10=14.7,
        E_kiso_log10=53.0,
        n0_log10=0.0,
        epsilon_e_log10=-1.0,
        epsilon_B_log10=-2.0,
        theta_j_rad=0.1,
        gamma0_log10=2.5,
        D_euc_gpc=5.28,
        rho_grb_log10=2.415,
        optical_survey=False,
        color_regimes=False,
        full_integral=False,
        qmin=0.0,
        Dmin_cm=0.0,
        toh_approx=False,
    )


@pytest.fixture
def default_model() -> DetectionRateModel:
    return DetectionRateModel(AfterglowPhysicalParams(), SurveyInstrumentParams())


# --------------------------------------------------------------------------- #
# 1. Model-method tests: dR_dq_full_integral and dR_dD_full_integral          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N_exp,t_cad_s", [(100.0, 86400.0), (50.0, 3600.0), (200.0, 43200.0)])
def test_dRdq_integrates_to_total(default_model, N_exp, t_cad_s):
    """∫_{0}^{q_nr} (dR/dq) dq must equal the full-integral total rate."""
    model = default_model
    q_vals, dRdq = model.dR_dq_full_integral(
        10, N_exp, t_cad_s, q_min=0.0, D_min_cm=0.0, N_q=500,
    )
    integ = float(np.trapezoid(dRdq, q_vals))
    log10R = float(model.rate_log10_full_integral(
        10, np.array([N_exp]), np.array([t_cad_s]),
        q_min=0.0, D_min_cm=0.0, N_q=500,
    )[0])
    R_total = 10.0 ** log10R
    assert R_total > 0
    # Both routes use the same N_q=500 grid → should agree to ≤ 0.5%.
    assert math.isclose(integ, R_total, rel_tol=5e-3)


@pytest.mark.parametrize("N_exp,t_cad_s", [(100.0, 86400.0), (50.0, 3600.0), (200.0, 43200.0)])
def test_dRdD_integrates_to_total(default_model, N_exp, t_cad_s):
    """∫_{0}^{D_Euc} (dR/dD) dD must equal the full-integral total rate.

    This is the load-bearing test for the (3/2) factor in dR/dD.
    """
    model = default_model
    D_cm, dRdD = model.dR_dD_full_integral(
        10, N_exp, t_cad_s, q_min=0.0, D_min_cm=0.0, N_q=500, N_D=400,
    )
    integ = float(np.trapezoid(dRdD, D_cm))
    log10R = float(model.rate_log10_full_integral(
        10, np.array([N_exp]), np.array([t_cad_s]),
        q_min=0.0, D_min_cm=0.0, N_q=500,
    )[0])
    R_total = 10.0 ** log10R
    assert R_total > 0
    # D-marginal uses an independent level-set inversion of D_eff(q); both
    # numerical paths should agree to ≤ 2% at N_D=400.
    assert math.isclose(integ, R_total, rel_tol=2e-2)


def test_dRdq_respects_q_min(default_model):
    """When q_min > 0, dR/dq must be exactly zero below q_min."""
    q_min = 1.5
    q_vals, dRdq = default_model.dR_dq_full_integral(
        10, 100.0, 86400.0, q_min=q_min, D_min_cm=0.0, N_q=500,
    )
    below = q_vals < q_min
    assert np.all(dRdq[below] == 0.0)
    # Above q_min the integrand has at least some positive support.
    assert np.any(dRdq[~below] > 0.0)


def test_dRdD_respects_D_min(default_model):
    """When D_min > 0, dR/dD must be exactly zero below D_min."""
    D_min_cm = 1.0e27  # ~0.32 Gpc
    D_cm, dRdD = default_model.dR_dD_full_integral(
        10, 100.0, 86400.0, q_min=0.0, D_min_cm=D_min_cm, N_q=500, N_D=200,
    )
    below = D_cm < D_min_cm
    assert np.all(dRdD[below] == 0.0)


# --------------------------------------------------------------------------- #
# 2. Bridge end-to-end: compute_qdview                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("full_on", [False, True])
def test_qdview_cumulative_endpoint_matches_rate(full_on):
    """Rq_cum[0] should match rate_log10(q_min=0, D_min=D_sidebar) at the
    strategy point. Same for RD_cum[0] with q_min=q_sidebar, D_min=0."""
    params = _default_params()
    params["full_integral"] = full_on
    params["qmin"] = 0.5            # cross-filters R(D)
    params["Dmin_cm"] = 0.5 * GPC_TO_CM   # cross-filters R(q)

    N_fix, t_cad_fix_s = 100.0, 86400.0
    payload = bridge.compute_qdview(params, N_fix, t_cad_fix_s)
    assert payload.get("error") is None

    state = bridge._build_models(params)
    model = state["model_day"]  # default_params has optical=False
    rate_fn = (model.rate_log10_full_integral if full_on else model.rate_log10)

    # Rq_cum at q=0: rate filtered to q≥0, D≥D_sidebar
    expected_q = 10.0 ** float(rate_fn(
        10, np.array([N_fix]), np.array([t_cad_fix_s]),
        q_min=0.0, D_min_cm=params["Dmin_cm"],
    )[0])
    Rq_cum_0 = payload["qdview_Rq_cum_flat"][0]
    assert Rq_cum_0 is not None and math.isclose(Rq_cum_0, expected_q, rel_tol=5e-3)

    # RD_cum at D=0: rate filtered to q≥q_sidebar, D≥0.
    # In full-integral mode, RD_cum is built by integrating dR/dD on a 200-pt
    # D-grid via level-set inversion of D_eff(q); rate_log10_full_integral
    # trapezoids over q at N_q=500. The two routes agree analytically but
    # diverge ~1% at N_D=200 due to discretization across the q_dec/q_j
    # transitions in D_eff(q).
    expected_D = 10.0 ** float(rate_fn(
        10, np.array([N_fix]), np.array([t_cad_fix_s]),
        q_min=params["qmin"], D_min_cm=0.0,
    )[0])
    RD_cum_0 = payload["qdview_RD_cum_flat"][0]
    tol_D = 1.5e-2 if full_on else 5e-3
    assert RD_cum_0 is not None and math.isclose(RD_cum_0, expected_D, rel_tol=tol_D)


@pytest.mark.parametrize("full_on", [False, True])
def test_qdview_differential_integrates_to_cumulative(full_on):
    """np.trapz(Rq_diff, q_grid) ≈ Rq_cum[0]; same for D-view. Load-bearing
    for the (3/2) prefactor in dR/dD and the dominant-term derivative."""
    params = _default_params()
    params["full_integral"] = full_on
    N_fix, t_cad_fix_s = 100.0, 86400.0
    payload = bridge.compute_qdview(params, N_fix, t_cad_fix_s)
    assert payload.get("error") is None

    q_grid  = np.asarray(payload["qdview_q_grid_flat"], dtype=float)
    Rq_diff = np.asarray(payload["qdview_Rq_diff_flat"], dtype=float)
    Rq_cum0 = float(payload["qdview_Rq_cum_flat"][0])
    integ_q = float(np.trapezoid(Rq_diff, q_grid))
    # Coarser tolerance for dominant-term: the 150-point view grid can miss
    # the q_max cliff. 2% is enough to catch the (3/2) factor; the cliff
    # rounding adds extra noise to dominant-term.
    tol = 5e-3 if full_on else 2e-2
    assert math.isclose(integ_q, Rq_cum0, rel_tol=tol)

    D_grid_Gpc = np.asarray(payload["qdview_D_grid_Gpc_flat"], dtype=float)
    RD_diff    = np.asarray(payload["qdview_RD_diff_flat"], dtype=float)
    RD_cum0    = float(payload["qdview_RD_cum_flat"][0])
    integ_D = float(np.trapezoid(RD_diff, D_grid_Gpc))
    tol_D = 2e-2 if full_on else 3e-2
    assert math.isclose(integ_D, RD_cum0, rel_tol=tol_D)


def test_qdview_toh_invalidity_returns_all_none():
    """A strategy that fails t_OH approximation must yield an all-None payload."""
    params = _default_params()
    params["toh_approx"] = True
    params["t_overhead_s"] = 30.0
    # Force f_live·t_cad/N_exp ≤ t_overhead: f_live=0.2, t_cad=100s, N_exp=10
    # → 0.2 · 100 / 10 = 2 s < 30 s. Invalid.
    payload = bridge.compute_qdview(params, N_exp_fix=10.0, t_cad_fix_s=100.0)
    assert payload.get("error") is None
    assert all(v is None for v in payload["qdview_Rq_cum_flat"])
    assert all(v is None for v in payload["qdview_Rq_diff_flat"])
    assert all(v is None for v in payload["qdview_RD_cum_flat"])
    assert all(v is None for v in payload["qdview_RD_diff_flat"])
    assert payload["qdview_total_rate_q"] is None
    assert payload["qdview_total_rate_D"] is None
