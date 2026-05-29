"""Physics verification tests for the fading-rate filter (s_min, s_mode).

Mirrors `test_filters.py` in style.  The two filter modes are:
  * "discrete"   : Δm/Δt across the i detections (matches survey alert filters)
  * "continuous" : |dm/dt| at the burst peak (strategy-independent limit)

The continuous mode is the t_cad → 0 limit of the discrete mode (proved
analytically in the plan and in `docs/implementation_reference.tex`).  Both
reduce to the unfiltered baseline at s_min = 0.

Run with::

    .venv/Scripts/python -m pytest tests/test_fading_filter.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.detection_rate import DetectionRateModel
from grb_detect.params import (
    AfterglowPhysicalParams,
    MicrophysicsParams,
    SurveyDesignParams,
    SurveyInstrumentParams,
    SurveyTelescopeParams,
)


def _make_default_model() -> DetectionRateModel:
    return DetectionRateModel(AfterglowPhysicalParams(), SurveyInstrumentParams())


def _make_deep_model() -> DetectionRateModel:
    """Deeper instrument; reaches more regimes including A1."""
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
    N = np.logspace(0, 4, 30)
    t = np.logspace(np.log10(60), np.log10(86400 * 30), 30)
    return np.meshgrid(N, t, indexing="ij")


# --------------------------------------------------------------------------- #
# F.1 - s_min=0 in both modes is bit-identical to the unfiltered baseline     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_mode", ["discrete", "continuous"])
@pytest.mark.parametrize("full", [False, True])
def test_F01_smin_zero_matches_baseline(model, strategy_grid, s_mode, full):
    N, T = strategy_grid
    i_det = 3
    if full:
        R_base = model.rate_log10_full_integral(i_det, N, T, N_q=500)
        R_cut = model.rate_log10_full_integral(
            i_det, N, T, s_min=0.0, s_mode=s_mode, N_q=500,
        )
    else:
        R_base = model.rate_log10(i_det, N, T)
        R_cut = model.rate_log10(i_det, N, T, s_min=0.0, s_mode=s_mode)
    np.testing.assert_array_equal(R_base, R_cut)


def test_F01b_medians_baseline_match(model, strategy_grid):
    """compute_medians at s_min=0 matches the no-arg call exactly."""
    N, T = strategy_grid
    i_det = 3
    q_a, D_a = model.compute_medians(i_det, N, T, full_integral=False)
    q_b, D_b = model.compute_medians(
        i_det, N, T, full_integral=False, s_min=0.0, s_mode="discrete",
    )
    np.testing.assert_array_equal(q_a, q_b)
    np.testing.assert_array_equal(D_a, D_b)


# --------------------------------------------------------------------------- #
# F.2 - i_det = 1 in discrete mode bypasses the cut; continuous still applies #
# --------------------------------------------------------------------------- #


def test_F02_idet1_discrete_bypasses(model, strategy_grid):
    """At i_det=1 with s_mode='discrete', the cut is undefined (Δt=0) and bypassed.
    Rate must match the s_min=0 baseline.  Continuous mode is unaffected.
    """
    N, T = strategy_grid
    R_base = model.rate_log10(1, N, T)
    R_disc = model.rate_log10(1, N, T, s_min=1.0, s_mode="discrete")
    R_cont = model.rate_log10(1, N, T, s_min=1.0, s_mode="continuous")

    np.testing.assert_array_equal(R_base, R_disc)
    # Continuous at i=1 should still bite (more NaN cells than baseline).
    finite_base = np.isfinite(R_base).sum()
    finite_cont = np.isfinite(R_cont).sum()
    # Either the rate dropped or some cells got nan'd; in any case it can't be identical.
    assert not np.array_equal(R_base, R_cont) or finite_cont < finite_base, (
        "Continuous mode at i=1 should still apply the cut"
    )


# --------------------------------------------------------------------------- #
# F.3 - Monotonic in s_min                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_mode", ["discrete", "continuous"])
@pytest.mark.parametrize("full", [False, True])
def test_F03_monotone_in_smin(model, strategy_grid, s_mode, full):
    """log10 R is non-increasing as s_min sweeps from 0 to 2 mag/day."""
    N, T = strategy_grid
    i_det = 3
    s_sweep = np.linspace(0.0, 2.0, 11)

    # Pick a cell that we expect to be on the surface (cadence-limited regime).
    masks = model.region_masks(i_det, N, T)
    # Look for any populated regime away from the boundary.
    for name in ("A5", "A6", "A4", "A2", "A3"):
        idx = np.argwhere(masks[name])
        if len(idx):
            i, j = idx[len(idx) // 2]
            Nv = float(N[i, j])
            tv = float(T[i, j])
            break
    else:
        pytest.skip("No suitable regime cell in strategy_grid")

    rates = []
    for s in s_sweep:
        if full:
            r = model.rate_log10_full_integral(
                i_det, np.array([Nv]), np.array([tv]),
                s_min=float(s), s_mode=s_mode, N_q=500,
            )
        else:
            r = model.rate_log10(
                i_det, np.array([Nv]), np.array([tv]),
                s_min=float(s), s_mode=s_mode,
            )
        rates.append(float(r[0]))
    arr = np.array(rates)
    finite = arr[np.isfinite(arr)]
    diffs = np.diff(finite)
    assert np.all(diffs <= 1e-10), (
        f"s_mode={s_mode} full={full}: rate increased with s_min: "
        f"max diff={diffs.max():.3e}, rates={arr}"
    )


# --------------------------------------------------------------------------- #
# F.4 - Closed-form check: at q_s,phase the measured slope equals s_min       #
# --------------------------------------------------------------------------- #


def _alpha(model, phase):
    p = model.phys.p
    return (
        float(model.pls.alpha_II_temporal(p)) if phase == "II"
        else float(model.pls.alpha_III_temporal(p))
    )


def _t_p(model, q_tilde, phase):
    t_j = float(model.derived.t_j_s)
    return t_j * (q_tilde ** (8.0 / 3.0)) if phase == "II" else t_j * (q_tilde ** 2.0)


def _s_meas_discrete(model, q_tilde, phase, i_det, t_cad_s):
    """Reference implementation of the (★) finite-difference slope."""
    alpha = _alpha(model, phase)
    t_p = _t_p(model, q_tilde, phase)
    dt = (i_det - 1) * t_cad_s
    return (
        -2.5 * alpha * math.log10(1.0 + dt / t_p) * DAY_S / dt
    )


def _s_meas_continuous(model, q_tilde, phase):
    alpha = _alpha(model, phase)
    t_p = _t_p(model, q_tilde, phase)
    return -2.5 * alpha * DAY_S / (t_p * math.log(10.0))


@pytest.mark.parametrize("phase", ["II", "III"])
@pytest.mark.parametrize("s_min", [0.3, 0.7, 1.5])
def test_F04_discrete_qs_inverts_to_smin(model, phase, s_min):
    """Take q_s_phase returned by the helper, plug back into (★) → expect s_min."""
    i_det = 3
    t_cad_s = float(DAY_S)  # 1 day
    q_s_II, q_s_III = model._q_s_fading_caps(
        i_det, np.array([t_cad_s]), s_min, "discrete",
    )
    q_s = float(q_s_II[0]) if phase == "II" else float(q_s_III[0])
    # Skip if cap >= phase domain (cut doesn't bite that phase).
    if not math.isfinite(q_s) or q_s <= 1.0:
        pytest.skip(f"phase {phase} cap not in active range at s={s_min}")
    q_tilde = q_s - 1.0
    s_check = _s_meas_discrete(model, q_tilde, phase, i_det, t_cad_s)
    assert math.isclose(s_check, s_min, rel_tol=1e-10), (
        f"phase {phase}, s_min={s_min}: q_s_check={s_check}, expected {s_min}"
    )


@pytest.mark.parametrize("phase", ["II", "III"])
@pytest.mark.parametrize("s_min", [0.3, 0.7, 1.5])
def test_F04b_continuous_qs_inverts_to_smin(model, phase, s_min):
    """q_s for s_mode='continuous' should invert (B1) to s_min exactly."""
    i_det = 3
    t_cad_s = float(DAY_S)
    q_s_II, q_s_III = model._q_s_fading_caps(
        i_det, np.array([t_cad_s]), s_min, "continuous",
    )
    q_s = float(q_s_II[0]) if phase == "II" else float(q_s_III[0])
    if not math.isfinite(q_s) or q_s <= 1.0:
        pytest.skip(f"phase {phase} cap not in active range at s={s_min}")
    q_tilde = q_s - 1.0
    s_check = _s_meas_continuous(model, q_tilde, phase)
    assert math.isclose(s_check, s_min, rel_tol=1e-10), (
        f"phase {phase}, s_min={s_min}: q_s_check={s_check}, expected {s_min}"
    )


# --------------------------------------------------------------------------- #
# F.5 - Limit equivalence: discrete → continuous as t_cad → 0                  #
# --------------------------------------------------------------------------- #


def test_F05_limit_equivalence_at_small_tcad(deep_model):
    """At small t_cad, the discrete formula converges to the continuous formula.

    Mathematically (B1) is the t_cad → 0 limit of (A1).  Pick a cell where both
    modes produce finite, non-trivial rates and verify they agree.
    """
    # Walk t_cad down logarithmically; for each step, find a cell where both
    # discrete and continuous give finite rates and report the absolute log10
    # difference.  The smallest t_cad we use should give a difference well
    # below 0.005 (since (10^E − 1) − E·ln10 ≈ (E·ln10)²/2, the relative
    # discrepancy is ~E·ln10 / 2 ≈ s_min·dt/(DAY_S·2·|α|)).
    i_det = 3
    s_min = 0.5
    p = deep_model.phys.p
    abs_alpha = abs(deep_model.pls.alpha_III_temporal(p))
    # Pick t_cad such that E·ln10 < 0.01 → diff in log10 rate ≪ 0.005.
    # E = s_min·(i−1)·t_cad / (DAY_S·2.5·|α|);  set t_cad accordingly.
    t_cad_s = 0.001 * (DAY_S * 2.5 * abs_alpha) / (s_min * (i_det - 1))
    # Sweep N_exp to find a finite cell.
    diffs = []
    for N_exp in np.logspace(0, 5, 60):
        R_disc = float(deep_model.rate_log10(
            i_det, np.array([N_exp]), np.array([t_cad_s]),
            s_min=s_min, s_mode="discrete",
        )[0])
        R_cont = float(deep_model.rate_log10(
            i_det, np.array([N_exp]), np.array([t_cad_s]),
            s_min=s_min, s_mode="continuous",
        )[0])
        if math.isfinite(R_disc) and math.isfinite(R_cont):
            diffs.append(abs(R_disc - R_cont))
    assert diffs, "No finite cell on the N_exp sweep — check parameters."
    max_diff = max(diffs)
    # The limit-equivalence relative error scales like E·ln10 / 2 ≈ 0.01;
    # in log10 rate that translates to ~0.005.  Allow 3× headroom.
    assert max_diff < 0.02, (
        f"discrete/continuous disagreement {max_diff:.4f} at t_cad={t_cad_s:.3g}s "
        f"exceeds the (E·ln10)² limit bound."
    )


# --------------------------------------------------------------------------- #
# F.6 - Helper bypass: s_min == 0 → +inf caps                                 #
# --------------------------------------------------------------------------- #


def test_F06_helper_bypass_returns_inf(model):
    """At s_min=0 the helper returns +inf for both phase caps, regardless of mode."""
    for s_mode in ("discrete", "continuous"):
        q_s_II, q_s_III = model._q_s_fading_caps(
            3, np.array([DAY_S, DAY_S * 7]), 0.0, s_mode,
        )
        assert np.all(np.isposinf(q_s_II)), f"s_mode={s_mode}: q_s_II not +inf at s=0"
        assert np.all(np.isposinf(q_s_III)), f"s_mode={s_mode}: q_s_III not +inf at s=0"


# --------------------------------------------------------------------------- #
# F.7 - Helper bypass: i_det < 2 in discrete mode → +inf caps                 #
# --------------------------------------------------------------------------- #


def test_F07_discrete_idet1_bypass(model):
    q_s_II, q_s_III = model._q_s_fading_caps(
        1, np.array([DAY_S]), 1.0, "discrete",
    )
    assert np.all(np.isposinf(q_s_II))
    assert np.all(np.isposinf(q_s_III))


# --------------------------------------------------------------------------- #
# F.8 - Phase III has a more lenient cap than Phase II at the same s_min      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_mode", ["discrete", "continuous"])
def test_F08_phaseIII_cap_at_least_phaseII(model, s_mode):
    """Phase III has steeper temporal slope (α_III=-p, |α_II|<|α_III|), so for the
    same fading rate threshold the Phase III viewer space survives further.
    """
    s_min = 0.5
    i_det = 3
    t_cad_s = DAY_S
    q_s_II, q_s_III = model._q_s_fading_caps(
        i_det, np.array([t_cad_s]), s_min, s_mode,
    )
    qII = float(q_s_II[0])
    qIII = float(q_s_III[0])
    # If both are finite, expect qIII >= qII (Phase III tolerates larger q).
    if math.isfinite(qII) and math.isfinite(qIII):
        assert qIII >= qII - 1e-12, (
            f"s_mode={s_mode}: q_s_III ({qIII}) should be >= q_s_II ({qII})"
        )


# --------------------------------------------------------------------------- #
# F.9 - Full-integral consistency at A1 (no off-axis tail past the cap)        #
# --------------------------------------------------------------------------- #


def test_F09_full_vs_dominant_at_A1_with_cut(deep_model):
    """At A1, q_max=q_nr and D_eff is constant in q.  The fading cap reduces the
    effective q_max identically in both modes, so the dominant-term and the
    full-integral results should agree to <1% (log10 diff < 0.005) regardless of
    s_min."""
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 3
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    if len(A1_idx) == 0:
        pytest.skip("No A1 cells in deep_model grid")
    i, j = A1_idx[len(A1_idx) // 2]
    Nv = float(N[i, j])
    tv = float(T[i, j])

    for s_min in (0.3, 1.0):
        R_dom = float(deep_model.rate_log10(
            i_det, np.array([Nv]), np.array([tv]),
            s_min=s_min, s_mode="discrete",
        )[0])
        R_full = float(deep_model.rate_log10_full_integral(
            i_det, np.array([Nv]), np.array([tv]),
            s_min=s_min, s_mode="discrete", N_q=500,
        )[0])
        if not (math.isfinite(R_dom) and math.isfinite(R_full)):
            pytest.skip(f"NaN at A1 with s_min={s_min}")
        diff = abs(R_dom - R_full)
        assert diff < 0.005, (
            f"A1 s_min={s_min}: log10 disagreement {diff:.4f} > 0.005"
        )
