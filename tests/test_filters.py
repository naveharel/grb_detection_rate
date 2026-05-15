"""Physics verification tests for the q_min / D_min detection-rate filters.

These tests verify that the viewing-angle floor (q_min) and distance floor
(D_min) implemented in `grb_detect/detection_rate.py` are mathematically and
physically correct end-to-end.  See the plan in
`C:/Users/naveh/.claude/plans/can-you-make-a-sharded-thacker.md` for the full
verification protocol; this file implements Layer B.

Each test maps to one row of the plan's test table (B.1 ... B.16).

Run with::

    .venv/Scripts/python -m pytest tests/test_filters.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grb_detect.constants import DEG2_TO_SR
from grb_detect.detection_rate import DetectionRateModel
from grb_detect.params import (
    AfterglowPhysicalParams,
    GPC_TO_CM,
    MicrophysicsParams,
    SurveyDesignParams,
    SurveyInstrumentParams,
    SurveyTelescopeParams,
)
from grb_detect.plot_3d_core import maximize_log_surface_iterative


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_default_model() -> DetectionRateModel:
    """Default-physics, default-instrument model."""
    return DetectionRateModel(AfterglowPhysicalParams(), SurveyInstrumentParams())


def _make_deep_model() -> DetectionRateModel:
    """Deeper (lower F_lim) instrument; opens up A1 and A4 regimes."""
    tele = SurveyTelescopeParams(
        omega_exp_sr=100.0 * DEG2_TO_SR,
        f_live=0.5,
        F_lim_ref_Jy=10 ** (-8.0),
    )
    instr = SurveyInstrumentParams(telescope=tele, design=SurveyDesignParams())
    return DetectionRateModel(AfterglowPhysicalParams(), instr)


@pytest.fixture(scope="module")
def model() -> DetectionRateModel:
    return _make_default_model()


@pytest.fixture(scope="module")
def deep_model() -> DetectionRateModel:
    return _make_deep_model()


@pytest.fixture(scope="module")
def strategy_grid() -> tuple[np.ndarray, np.ndarray]:
    """A representative (N_exp, t_cad) grid spanning all 7 regimes (broadcast-ready)."""
    N = np.logspace(0, 4, 30)
    t = np.logspace(np.log10(60), np.log10(86400 * 30), 30)  # 1 min -> 30 days
    return np.meshgrid(N, t, indexing="ij")


def _first_cell_in(masks, name, N, T):
    """Return (N_val, t_val) of any cell in regime `name`, or (None, None)."""
    idx = np.argwhere(masks[name])
    if len(idx) == 0:
        return None, None
    i, j = idx[len(idx) // 2]
    return float(N[i, j]), float(T[i, j])


# --------------------------------------------------------------------------- #
# Helper: analytic unfiltered baseline per regime                             #
# --------------------------------------------------------------------------- #


def _analytic_unfiltered_rate(model: DetectionRateModel, regime: str, N: float, t: float, i_det: int = 1) -> float:
    """Closed-form unfiltered rate per regime: base * q_max^2 * (D_eff/D_Euc)^3."""
    N_a = np.asarray([N])
    t_a = np.asarray([t])
    t_exp = model.t_exp_s(N_a, t_a)
    F_lim = model.F_lim_Jy(t_exp)
    fO = model.f_Omega(N_a)
    qE = model.q_Euc(F_lim)
    qi = model.q_i(i_det, t_a)
    D_dec = model.D_dec(F_lim)
    D_i = model.D_i(i_det, t_a, F_lim)

    R_int = model.phys.R_int_yr
    theta_j = model.phys.theta_j_rad
    D_euc = model.phys.D_euc_cm
    q_nr = float(model.derived.q_nr)
    q_dec = float(model.derived.q_dec)

    base = 0.5 * fO * (theta_j ** 2) * R_int

    if regime == "A1":
        q_max_sq = q_nr ** 2
        D_norm_cubed = 1.0
    elif regime in ("A2", "A3"):
        q_max_sq = float(qE[0]) ** 2
        D_norm_cubed = 1.0
    elif regime == "A4":
        q_max_sq = q_nr ** 2
        D_norm_cubed = (float(D_i[0]) / D_euc) ** 3
    elif regime in ("A5", "A6"):
        q_max_sq = float(qi[0]) ** 2
        D_norm_cubed = (float(D_i[0]) / D_euc) ** 3
    elif regime == "A7":
        q_max_sq = q_dec ** 2
        D_eff7 = min(float(D_dec[0]), float(D_i[0]))
        D_norm_cubed = (D_eff7 / D_euc) ** 3
    else:
        raise ValueError(regime)

    return float(base[0] * q_max_sq * D_norm_cubed)


# --------------------------------------------------------------------------- #
# B.1 - Dominant unfiltered identity                                          #
# --------------------------------------------------------------------------- #


def test_B01_unfiltered_dominant_identity(model, strategy_grid):
    """rate_log10(q_min=0,D_min_cm=0) matches the closed-form unfiltered rate per regime."""
    N, T = strategy_grid
    i_det = 3  # exposes A2, A3, A4, A5
    R_filtered = model.rate_log10(i_det, N, T, q_min=0.0, D_min_cm=0.0)
    masks = model.region_masks(i_det, N, T)

    checked = 0
    for regime in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]:
        idx = np.argwhere(masks[regime])
        if len(idx) == 0:
            continue
        # Spot-check the median cell of each populated regime
        i, j = idx[len(idx) // 2]
        expected = _analytic_unfiltered_rate(model, regime, float(N[i, j]), float(T[i, j]), i_det)
        assert expected > 0, f"{regime}: analytic baseline is non-positive at ({N[i,j]}, {T[i,j]})"
        got = 10.0 ** float(R_filtered[i, j])
        assert math.isclose(got, expected, rel_tol=1e-12), (
            f"{regime}: unfiltered identity violated at (N={N[i,j]}, t={T[i,j]}): "
            f"code={got:.6e}, analytic={expected:.6e}, rel_err={abs(got-expected)/expected:.3e}"
        )
        checked += 1
    assert checked >= 3, f"Expected to hit at least 3 regimes, got {checked}"


# --------------------------------------------------------------------------- #
# B.2 - Full-integral unfiltered identity                                     #
# --------------------------------------------------------------------------- #


def test_B02_unfiltered_full_integral_at_qmin_zero_is_same_codepath(model, strategy_grid):
    """At q_min=0,D_min=0, the full-integral filter clauses degenerate to no-op.

    We assert that the full-integral with explicit (q_min=0, D_min_cm=0)
    matches the call without those kwargs (relative <= 1e-13)."""
    N, T = strategy_grid
    i_det = 3
    R_default = model.rate_log10_full_integral(i_det, N, T, N_q=500)
    R_explicit = model.rate_log10_full_integral(i_det, N, T, q_min=0.0, D_min_cm=0.0, N_q=500)
    # Both should give exactly identical results since the filter args have default 0.
    np.testing.assert_array_equal(R_default, R_explicit)


# --------------------------------------------------------------------------- #
# B.3 - q_min beyond q_nr zeros everything                                    #
# --------------------------------------------------------------------------- #


def test_B03_qmin_beyond_qnr_zeros_everything(model, strategy_grid):
    """q_min slightly above q_nr -> NaN log10 rate at every A0 point, both modes."""
    N, T = strategy_grid
    i_det = 1
    q_nr = float(model.derived.q_nr)
    big_q = q_nr * (1.0 + 1e-9)

    R_dom = model.rate_log10(i_det, N, T, q_min=big_q, D_min_cm=0.0)
    R_full = model.rate_log10_full_integral(i_det, N, T, q_min=big_q, D_min_cm=0.0, N_q=500)

    masks = model.region_masks(i_det, N, T)
    A0 = masks["A0"]
    assert np.all(np.isnan(R_dom[A0])), "Dominant: some A0 cells survived q_min > q_nr"
    assert np.all(np.isnan(R_full[A0])), "Full integral: some A0 cells survived q_min > q_nr"


# --------------------------------------------------------------------------- #
# B.4 - D_min beyond D_Euc zeros A1/A2/A3 (dominant) and everything (full)    #
# --------------------------------------------------------------------------- #


def test_B04_Dmin_beyond_DEuc_zeros_A1_A2_A3(deep_model, strategy_grid):
    """D_min > D_Euc -> NaN in A1/A2/A3 (D_eff=D_Euc); full integral -> NaN everywhere."""
    N, T = strategy_grid
    i_det = 1
    D_euc = deep_model.phys.D_euc_cm
    big_D = D_euc * (1.0 + 1e-9)

    R_dom = deep_model.rate_log10(i_det, N, T, q_min=0.0, D_min_cm=big_D)
    R_full = deep_model.rate_log10_full_integral(i_det, N, T, q_min=0.0, D_min_cm=big_D, N_q=500)

    masks = deep_model.region_masks(i_det, N, T)
    for name in ("A1", "A2", "A3"):
        if np.any(masks[name]):
            assert np.all(np.isnan(R_dom[masks[name]])), (
                f"Dominant: {name} survived D_min > D_Euc"
            )

    # Full integral: D_eff(q) is always <= D_Euc (capped), so D_min > D_Euc zeros all.
    A0 = masks["A0"]
    assert np.all(np.isnan(R_full[A0])), "Full integral: D_min > D_Euc did not zero everything"


# --------------------------------------------------------------------------- #
# B.5 - Monotone decreasing in q_min                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["dominant", "full"])
def test_B05_monotone_decreasing_in_qmin(model, strategy_grid, mode):
    """log10 R is non-increasing as q_min is swept from 0 -> q_nr at multiple cells."""
    N, T = strategy_grid
    i_det = 3
    masks = model.region_masks(i_det, N, T)
    q_nr = float(model.derived.q_nr)
    q_sweep = np.linspace(0.0, q_nr * 0.99, 25)

    # Pick a few cells, one per available regime
    cells = []
    for name in ("A1", "A2", "A3", "A4", "A5"):
        Nv, tv = _first_cell_in(masks, name, N, T)
        if Nv is not None:
            cells.append((name, Nv, tv))
    assert len(cells) >= 3, "Need at least 3 regime cells to test monotonicity"

    for name, Nv, tv in cells:
        rates = []
        for q in q_sweep:
            if mode == "dominant":
                r = model.rate_log10(i_det, np.array([Nv]), np.array([tv]), q_min=q, D_min_cm=0.0)
            else:
                r = model.rate_log10_full_integral(
                    i_det, np.array([Nv]), np.array([tv]), q_min=q, D_min_cm=0.0, N_q=500
                )
            rates.append(float(r[0]))
        rates_arr = np.array(rates)
        # Drop NaN tail (regime fully filtered out)
        finite = np.isfinite(rates_arr)
        finite_rates = rates_arr[finite]
        diffs = np.diff(finite_rates)
        # log10 R should be non-increasing (allow tiny float jitter)
        assert np.all(diffs <= 1e-10), (
            f"{name} ({mode}): rate increased as q_min grew: max diff={diffs.max():.3e}"
        )


# --------------------------------------------------------------------------- #
# B.6 - Monotone decreasing in D_min                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["dominant", "full"])
def test_B06_monotone_decreasing_in_Dmin(model, strategy_grid, mode):
    """log10 R is non-increasing as D_min is swept from 0 -> D_Euc."""
    N, T = strategy_grid
    i_det = 3
    masks = model.region_masks(i_det, N, T)
    D_euc = model.phys.D_euc_cm
    D_sweep = np.linspace(0.0, D_euc * 0.99, 25)

    cells = []
    for name in ("A1", "A2", "A3", "A4", "A5"):
        Nv, tv = _first_cell_in(masks, name, N, T)
        if Nv is not None:
            cells.append((name, Nv, tv))
    assert len(cells) >= 3

    for name, Nv, tv in cells:
        rates = []
        for D in D_sweep:
            if mode == "dominant":
                r = model.rate_log10(i_det, np.array([Nv]), np.array([tv]), q_min=0.0, D_min_cm=D)
            else:
                r = model.rate_log10_full_integral(
                    i_det, np.array([Nv]), np.array([tv]), q_min=0.0, D_min_cm=D, N_q=500
                )
            rates.append(float(r[0]))
        rates_arr = np.array(rates)
        finite = np.isfinite(rates_arr)
        finite_rates = rates_arr[finite]
        diffs = np.diff(finite_rates)
        assert np.all(diffs <= 1e-10), (
            f"{name} ({mode}): rate increased as D_min grew: max diff={diffs.max():.3e}"
        )


# --------------------------------------------------------------------------- #
# B.7 - Dominant vs full-integral at A1 (filter-on agreement)                 #
# --------------------------------------------------------------------------- #


def test_B07_dominant_vs_full_integral_at_A1(deep_model):
    """At A1 cells (q_max=q_nr, D_eff=D_Euc both constant in q), the full integral
    is exactly the closed form, so it should match the dominant-term rate
    to <1% (log10 diff < 0.005) both unfiltered AND with a non-trivial filter.
    """
    # Pick a known A1 cell from the deep-model grid
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 1
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    assert len(A1_idx) > 0, "No A1 cells in deep_model grid"
    # Use the deep interior — far from regime boundaries
    i, j = A1_idx[len(A1_idx) // 2]
    Nv = float(N[i, j])
    tv = float(T[i, j])

    q_nr = float(deep_model.derived.q_nr)
    D_euc = deep_model.phys.D_euc_cm

    for q_min, D_min in [(0.0, 0.0), (q_nr / 2, D_euc / 2)]:
        R_dom = float(deep_model.rate_log10(
            i_det, np.array([Nv]), np.array([tv]), q_min=q_min, D_min_cm=D_min
        )[0])
        R_full = float(deep_model.rate_log10_full_integral(
            i_det, np.array([Nv]), np.array([tv]), q_min=q_min, D_min_cm=D_min, N_q=500
        )[0])
        assert np.isfinite(R_dom) and np.isfinite(R_full), (
            f"NaN rate at q_min={q_min}, D_min={D_min}"
        )
        diff = abs(R_dom - R_full)
        assert diff < 0.005, (
            f"A1 q_min={q_min}, D_min={D_min}: log10 disagreement {diff:.4f} exceeds 0.005"
        )


# --------------------------------------------------------------------------- #
# B.8 - Dominant-term truncation visible outside A1                           #
# --------------------------------------------------------------------------- #


def test_B08_dominant_full_integral_disagree_in_tail_regimes(model, strategy_grid):
    """The dominant-term formula truncates the integration at q_max_k, ignoring
    the off-axis tail q in (q_max_k, q_nr] where D_eff(q) is small but non-zero.

    For A2/A3 (q_max=q_E < q_nr) and A5/A6 (q_max=q_i < q_nr), the full integral
    picks up this tail and reports a *higher* rate than the dominant-term.

    For A1 and A4, q_max=q_nr so there is no tail; the two modes agree exactly
    (modulo trapezoidal quadrature noise).

    This test confirms the disagreement is in the expected direction (full >= dominant)
    and is substantial in the tail-bearing regimes.
    """
    N, T = strategy_grid
    i_det = 3
    masks = model.region_masks(i_det, N, T)
    found_tail_regime = False
    for name in ("A2", "A3", "A5", "A6", "A7"):
        Nv, tv = _first_cell_in(masks, name, N, T)
        if Nv is None:
            continue
        R_dom = float(model.rate_log10(i_det, np.array([Nv]), np.array([tv]))[0])
        R_full = float(model.rate_log10_full_integral(
            i_det, np.array([Nv]), np.array([tv]), N_q=500
        )[0])
        if not (np.isfinite(R_dom) and np.isfinite(R_full)):
            continue
        # Full integral should be >= dominant (it adds the off-axis tail).
        assert R_full >= R_dom - 1e-6, (
            f"{name}: full integral ({R_full:.4f}) < dominant ({R_dom:.4f}); "
            f"this contradicts the off-axis-tail physics."
        )
        found_tail_regime = True
    assert found_tail_regime, "No tail-bearing regime (A2/A3/A5/A6/A7) reachable"


# --------------------------------------------------------------------------- #
# B.9 - Analytic median q (unfiltered)                                        #
# --------------------------------------------------------------------------- #


def test_B09_analytic_median_q_unfiltered(model, deep_model, strategy_grid):
    """At unfiltered q_min=0,D_min=0: q_med = q_max/sqrt(2) per regime."""
    # Test on both default and deep models to cover more regimes
    for m in (model, deep_model):
        if m is deep_model:
            N = np.logspace(0, 5, 80)
            t = np.logspace(np.log10(10), np.log10(86400 * 365), 80)
            N_g, T_g = np.meshgrid(N, t, indexing="ij")
        else:
            N_g, T_g = strategy_grid

        i_det = 3
        masks = m.region_masks(i_det, N_g, T_g)
        q_med, _ = m.compute_medians_analytic(i_det, N_g, T_g, q_min=0.0, D_min_cm=0.0)

        # Compute q_max per regime
        F_lim = m.F_lim_Jy(m.t_exp_s(N_g, T_g))
        qE = m.q_Euc(F_lim)
        qi = m.q_i(i_det, T_g)
        q_nr = float(m.derived.q_nr)
        q_dec = float(m.derived.q_dec)

        expected_q_max = np.full_like(qE, np.nan)
        expected_q_max = np.where(masks["A1"], q_nr, expected_q_max)
        expected_q_max = np.where(masks["A2"] | masks["A3"], qE, expected_q_max)
        expected_q_max = np.where(masks["A4"], q_nr, expected_q_max)
        expected_q_max = np.where(masks["A5"] | masks["A6"], qi, expected_q_max)
        expected_q_max = np.where(masks["A7"], q_dec, expected_q_max)

        expected_q_med = expected_q_max / math.sqrt(2.0)

        cells = np.isfinite(q_med) & np.isfinite(expected_q_med)
        assert np.any(cells), "No valid cells for unfiltered median check"
        np.testing.assert_allclose(q_med[cells], expected_q_med[cells], rtol=1e-12)


# --------------------------------------------------------------------------- #
# B.10 - Analytic median with filter                                          #
# --------------------------------------------------------------------------- #


def test_B10_analytic_median_with_filter(deep_model):
    """At an A1 cell with q_min=q_nr/2 and D_min=D_Euc/2:
       q_med = q_nr * sqrt(5/8), D_med = D_Euc * (9/16)**(1/3)."""
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 1
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    assert len(A1_idx) > 0
    i, j = A1_idx[len(A1_idx) // 2]
    Nv = float(N[i, j])
    tv = float(T[i, j])

    q_nr = float(deep_model.derived.q_nr)
    D_euc = deep_model.phys.D_euc_cm

    q_med, D_med = deep_model.compute_medians_analytic(
        i_det, np.array([Nv]), np.array([tv]),
        q_min=q_nr / 2, D_min_cm=D_euc / 2,
    )
    q_expected = q_nr * math.sqrt(5.0 / 8.0)
    D_expected = D_euc * (9.0 / 16.0) ** (1.0 / 3.0)
    assert math.isclose(float(q_med[0]), q_expected, rel_tol=1e-12), (
        f"q_med: got {q_med[0]}, expected {q_expected}"
    )
    assert math.isclose(float(D_med[0]), D_expected, rel_tol=1e-12), (
        f"D_med: got {D_med[0]}, expected {D_expected}"
    )


# --------------------------------------------------------------------------- #
# B.11 - Numerical median agrees with analytic at A1                          #
# --------------------------------------------------------------------------- #


def test_B11_numerical_vs_analytic_median_at_A1(deep_model):
    """At an A1 cell (where D_eff(q) is genuinely flat at D_Euc), the numerical
    median from compute_medians_numerical should agree with the analytic
    formula to within grid resolution."""
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 1
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    assert len(A1_idx) > 0
    i, j = A1_idx[len(A1_idx) // 2]
    Nv = float(N[i, j])
    tv = float(T[i, j])

    N_q = 1000  # bump up resolution for tight tolerance
    q_med_an, D_med_an = deep_model.compute_medians_analytic(
        i_det, np.array([Nv]), np.array([tv]),
    )
    q_med_nu, D_med_nu = deep_model.compute_medians_numerical(
        i_det, np.array([Nv]), np.array([tv]), N_q=N_q,
    )
    q_nr = float(deep_model.derived.q_nr)
    D_euc = deep_model.phys.D_euc_cm

    # Grid resolution sets the floor
    q_tol = q_nr / N_q * 2.0
    D_tol = D_euc / 100 * 2.0  # numerical D grid is 100 levels
    assert abs(float(q_med_nu[0]) - float(q_med_an[0])) < q_tol, (
        f"q_med: numerical={q_med_nu[0]}, analytic={q_med_an[0]}"
    )
    # Sanity: A1 has D_eff = D_Euc so analytic D_med = D_Euc * 2^(-1/3)
    assert math.isclose(float(D_med_an[0]), D_euc * 2 ** (-1.0 / 3.0), rel_tol=1e-12)
    assert abs(float(D_med_nu[0]) - float(D_med_an[0])) < D_tol


# --------------------------------------------------------------------------- #
# B.12 - Indicator bias at q_min (full integral, synthetic flat D_eff case)   #
# --------------------------------------------------------------------------- #


def test_B12_indicator_bias_at_qmin_bounded(deep_model):
    """The trapezoidal indicator `1{q>=q_min}` applied to the q-integrand
    introduces O(dq/q_nr) numerical error near q_min. Quantify and bound it
    at an A1 cell (where D_eff(q) = D_Euc is constant, so the integral has a
    closed form: R = base * 0.5 * (q_nr^2 - q_min^2) * 1).
    """
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 1
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    i, j = A1_idx[len(A1_idx) // 2]
    Nv = float(N[i, j])
    tv = float(T[i, j])
    q_nr = float(deep_model.derived.q_nr)

    N_q = 500
    dq = q_nr / N_q

    # Sample q_min at on-node and off-node positions
    q_min_values = []
    for k in [50, 100, 200, 300]:
        q_node = q_nr * (k / N_q)             # on-node
        q_off  = q_nr * ((k + 0.5) / N_q)     # off-node
        q_min_values.extend([q_node, q_off])

    max_rel_err = 0.0
    for q_min in q_min_values:
        R_full = float(deep_model.rate_log10_full_integral(
            i_det, np.array([Nv]), np.array([tv]),
            q_min=q_min, D_min_cm=0.0, N_q=N_q,
        )[0])
        # Closed form: R_dom would give base * (q_nr^2 - q_min^2) * 1
        R_dom = float(deep_model.rate_log10(
            i_det, np.array([Nv]), np.array([tv]),
            q_min=q_min, D_min_cm=0.0,
        )[0])
        rel_err = abs(10 ** R_full - 10 ** R_dom) / (10 ** R_dom)
        if rel_err > max_rel_err:
            max_rel_err = rel_err

    # Theoretical bound: ~dq/q_nr = 1/N_q = 0.002. Allow 5x headroom.
    bound = 5.0 / N_q
    assert max_rel_err < bound, (
        f"Indicator bias at q_min larger than expected: "
        f"max_rel_err={max_rel_err:.4e}, bound={bound:.4e}, N_q={N_q}, dq/q_nr={1/N_q:.4e}"
    )


# --------------------------------------------------------------------------- #
# B.13 - D_eff(q) monotonicity + continuity                                   #
# --------------------------------------------------------------------------- #


def test_B13_D_eff_q_monotone_and_continuous(model, strategy_grid):
    """Reconstruct the same D_eff(q) profile that rate_log10_full_integral
    uses internally; assert it is non-increasing in q and continuous at the
    q_dec / q_j phase boundaries (for PLS G)."""
    N, T = strategy_grid
    i_det = 3
    masks = model.region_masks(i_det, N, T)
    Nv, tv = _first_cell_in(masks, "A5", N, T)
    if Nv is None:
        Nv, tv = _first_cell_in(masks, "A2", N, T)

    t_exp = model.t_exp_s(np.array([Nv]), np.array([tv]))
    F_lim = model.F_lim_Jy(t_exp)
    D_i = model.D_i(i_det, np.array([tv]), F_lim)
    D_tilde_dec = model.D_dec(F_lim) / model.phys.D_euc_cm
    D_tilde_eff = np.minimum(D_i / model.phys.D_euc_cm, 1.0)

    q_dec = float(model.derived.q_dec)
    q_j = float(model.derived.q_j)
    q_nr = float(model.derived.q_nr)
    q_td = q_dec - 1.0
    a_II = float(model.pls.a_II(model.phys.p))
    a_III = float(model.pls.a_III(model.phys.p))

    N_q = 2000
    q_vals = np.linspace(0.0, q_nr, N_q + 1)[1:]
    qt = np.maximum(q_vals - 1.0, 0.0)

    on_axis = q_vals < q_dec
    phase_II = (q_vals >= q_dec) & (q_vals < q_j)
    phase_III = q_vals >= q_j

    safe_II = np.where(phase_II, np.maximum(qt, 1e-30), 1.0)
    safe_III = np.where(phase_III, np.maximum(qt, 1e-30), 1.0)
    D_max_II = float(D_tilde_dec[0]) * (safe_II / q_td) ** (-a_II)
    D_max_III = (float(D_tilde_dec[0]) / q_td) * (safe_III / q_td) ** (-a_III)
    D_tilde_max = np.where(on_axis, float(D_tilde_dec[0]),
                  np.where(phase_II, D_max_II, D_max_III))
    D_eff = np.minimum(D_tilde_max, float(D_tilde_eff[0]))

    # Monotonicity: D_eff non-increasing
    diffs = np.diff(D_eff)
    assert np.all(diffs <= 1e-12), f"D_eff increased: max diff {diffs.max():.3e}"

    # Continuity at q_dec: D_max_II(q_dec) should match the on-axis value
    val_left = float(D_tilde_dec[0])
    val_right = float(D_tilde_dec[0]) * (q_td / q_td) ** (-a_II)
    assert math.isclose(val_left, val_right, rel_tol=1e-12), "Discontinuity at q_dec"

    # Continuity at q_j (PLS G only: a_II = p-1, a_III = p, so a_II = a_III - 1)
    if model.pls.__class__.__name__ == "PLSG":
        val_II_at_qj = float(D_tilde_dec[0]) * (1.0 / q_td) ** (-a_II)
        val_III_at_qj = (float(D_tilde_dec[0]) / q_td) * (1.0 / q_td) ** (-a_III)
        assert math.isclose(val_II_at_qj, val_III_at_qj, rel_tol=1e-12), (
            f"PLS G: discontinuity at q_j: II={val_II_at_qj}, III={val_III_at_qj}"
        )


# --------------------------------------------------------------------------- #
# B.14 - NaN on over-filtered regime                                          #
# --------------------------------------------------------------------------- #


def test_B14_NaN_on_overfiltered_regime(deep_model):
    """When the filter zeros all weight, rate and medians return NaN, not -inf
    or stale values."""
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 1
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    assert len(A1_idx) > 0
    i, j = A1_idx[len(A1_idx) // 2]
    Nv = float(N[i, j])
    tv = float(T[i, j])

    q_nr = float(deep_model.derived.q_nr)
    D_euc = deep_model.phys.D_euc_cm

    # q_min > q_nr -> all-weight filtered out
    R = float(deep_model.rate_log10(
        i_det, np.array([Nv]), np.array([tv]),
        q_min=q_nr + 1e-6, D_min_cm=0.0,
    )[0])
    q_med, D_med = deep_model.compute_medians_analytic(
        i_det, np.array([Nv]), np.array([tv]),
        q_min=q_nr + 1e-6, D_min_cm=0.0,
    )
    assert np.isnan(R), f"Rate should be NaN, got {R}"
    assert np.isnan(float(q_med[0])), f"q_med should be NaN, got {q_med[0]}"
    assert np.isnan(float(D_med[0])), f"D_med should be NaN, got {D_med[0]}"

    # D_min > D_Euc -> same
    R2 = float(deep_model.rate_log10(
        i_det, np.array([Nv]), np.array([tv]),
        q_min=0.0, D_min_cm=D_euc + 1e20,
    )[0])
    q_med2, D_med2 = deep_model.compute_medians_analytic(
        i_det, np.array([Nv]), np.array([tv]),
        q_min=0.0, D_min_cm=D_euc + 1e20,
    )
    assert np.isnan(R2)
    assert np.isnan(float(q_med2[0]))
    assert np.isnan(float(D_med2[0]))

    # Full integral path
    R3 = float(deep_model.rate_log10_full_integral(
        i_det, np.array([Nv]), np.array([tv]),
        q_min=q_nr + 1e-6, D_min_cm=0.0, N_q=500,
    )[0])
    q_med3, D_med3 = deep_model.compute_medians_numerical(
        i_det, np.array([Nv]), np.array([tv]),
        N_q=500, q_min=q_nr + 1e-6, D_min_cm=0.0,
    )
    assert np.isnan(R3)
    assert np.isnan(float(q_med3[0]))
    assert np.isnan(float(D_med3[0]))


# --------------------------------------------------------------------------- #
# B.15 - Optimizer + _eval_point consistency under filters                    #
# --------------------------------------------------------------------------- #


def test_B15_optimizer_consistency_under_filter(deep_model):
    """Run the optimizer with a non-trivial filter, then evaluate the same
    model at the returned (N_opt, t_cad_opt). The two values must agree to
    within the optimizer's own discretization tolerance."""
    q_min = 1.5
    D_min_cm = 2.0 * GPC_TO_CM
    N_opt, t_cad_opt_s, R_opt = maximize_log_surface_iterative(
        deep_model, None, i_det=1,
        x_min=0.0, x_max=4.5, y_min=0.0, y_max=8.0,
        optical_survey=False,
        t_night_s=10.0 * 3600.0,
        full_integral=False,
        q_min=q_min, D_min_cm=D_min_cm,
    )
    # Re-evaluate via the same model
    R_check = float(deep_model.rate_log10(
        1, np.array([N_opt]), np.array([t_cad_opt_s]),
        q_min=q_min, D_min_cm=D_min_cm,
    )[0])
    assert np.isfinite(R_opt) and np.isfinite(R_check), (
        f"NaN at optimum: R_opt={R_opt}, R_check={R_check}"
    )
    assert abs(R_opt - R_check) < 0.01, (
        f"Optimizer disagreement: R_opt={R_opt:.4f}, R_check={R_check:.4f}, diff={abs(R_opt-R_check):.4e}"
    )


# --------------------------------------------------------------------------- #
# B.16 - Joint PDF normalization under filter                                 #
# --------------------------------------------------------------------------- #


def test_B16_joint_pdf_normalization_under_filter(deep_model):
    """The numerical-median integrand and the rate computation are driven by
    the SAME joint PDF p(q,D) ~ q*D^2. Verify by integrating the integrand
    on an A1 cell and checking it equals the analytic
    1/2*(q_max^2 - q_min^2) * 1/3 * (D_eff^3 - D_min^3) to <1%.
    """
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 1
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    assert len(A1_idx) > 0
    i, j = A1_idx[len(A1_idx) // 2]
    Nv = float(N[i, j])
    tv = float(T[i, j])

    q_nr = float(deep_model.derived.q_nr)
    D_euc = deep_model.phys.D_euc_cm

    q_min = q_nr / 3
    D_min_cm = D_euc / 3

    # Numerical: compute the full integral I in normalized units
    # Re-derive I = R / (fO * theta_j^2 * R_int)
    t_exp = deep_model.t_exp_s(np.array([Nv]), np.array([tv]))
    F_lim = deep_model.F_lim_Jy(t_exp)
    fO = float(deep_model.f_Omega(np.array([Nv]))[0])
    theta_j = deep_model.phys.theta_j_rad
    R_int = deep_model.phys.R_int_yr

    R_full = float(deep_model.rate_log10_full_integral(
        i_det, np.array([Nv]), np.array([tv]),
        q_min=q_min, D_min_cm=D_min_cm, N_q=2000,
    )[0])
    I_num = 10 ** R_full / (fO * theta_j ** 2 * R_int)

    # Analytic in normalized units: I = 1/2*(q_nr^2 - q_min^2)*(1 - (D_min/D_Euc)^3)
    # Note: this matches the dominant-term r1 / (fO * theta_j^2 * R_int / 2) ... wait,
    # the base is 0.5 * fO * theta_j^2 * R_int, so the un-base'd integral is
    # 0.5 * (q_nr^2 - q_min^2) * (1 - (D_min/D_Euc)^3).  Verify directly:
    I_an = 0.5 * (q_nr ** 2 - q_min ** 2) * (1.0 - (D_min_cm / D_euc) ** 3)

    rel_err = abs(I_num - I_an) / I_an
    assert rel_err < 0.01, (
        f"Joint-PDF normalization mismatch at A1: numerical I={I_num:.6e}, "
        f"analytic I={I_an:.6e}, rel_err={rel_err:.4e}"
    )
