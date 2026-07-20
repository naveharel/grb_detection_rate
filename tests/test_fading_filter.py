"""Physics verification tests for the fading-rate filter (s_fade, s_mode).

Mirrors `test_filters.py` in style.  The two slope definitions are:
  * "discrete"   : Δm/Δt across the i detections (matches survey alert filters)
  * "continuous" : |dm/dt| at the first detection (strategy-independent limit)

The continuous mode is the t_cad → 0 limit of the discrete mode (proved
analytically in `docs/implementation_reference.tex`).  Both reduce to the
unfiltered baseline at s_fade = 0.

The two rate modes treat the observation start time differently:
  * Dominant-term (closed-form): best-case start at the peak, t_first ≈ t_p(q)
    → phase-mapped hard viewing-angle cap q ≤ q_s (`_q_s_fading_caps`).
  * Full-integral: uniform start, t_first = t_p + u with u ~ U(0, t_cad)
    → pointwise survival probability P(q) = clip((t_f,s − t_p)/t_cad, 0, 1)
    (`_fading_survival`) multiplying the q-integrand.
Since P ≤ 1{q ≤ q_s} pointwise, the full-integral rate sits below the
dominant-term rate whenever the cut is active — by design, not a bug.

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
# F.1 - s_fade=0 in both modes is bit-identical to the unfiltered baseline     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_mode", ["discrete", "continuous"])
@pytest.mark.parametrize("full", [False, True])
def test_F01_sfade_zero_matches_baseline(model, strategy_grid, s_mode, full):
    N, T = strategy_grid
    i_det = 3
    if full:
        R_base = model.rate_log10_full_integral(i_det, N, T, N_q=500)
        R_cut = model.rate_log10_full_integral(
            i_det, N, T, s_fade=0.0, s_mode=s_mode, N_q=500,
        )
    else:
        R_base = model.rate_log10(i_det, N, T)
        R_cut = model.rate_log10(i_det, N, T, s_fade=0.0, s_mode=s_mode)
    np.testing.assert_array_equal(R_base, R_cut)


def test_F01b_medians_baseline_match(model, strategy_grid):
    """compute_medians at s_fade=0 matches the no-arg call exactly."""
    N, T = strategy_grid
    i_det = 3
    q_a, D_a = model.compute_medians(i_det, N, T, full_integral=False)
    q_b, D_b = model.compute_medians(
        i_det, N, T, full_integral=False, s_fade=0.0, s_mode="discrete",
    )
    np.testing.assert_array_equal(q_a, q_b)
    np.testing.assert_array_equal(D_a, D_b)


# --------------------------------------------------------------------------- #
# F.2 - i_det = 1 in discrete mode bypasses the cut; continuous still applies #
# --------------------------------------------------------------------------- #


def test_F02_idet1_discrete_bypasses(model, strategy_grid):
    """At i_det=1 with s_mode='discrete', the cut is undefined (Δt=0) and bypassed.
    Rate must match the s_fade=0 baseline.  Continuous mode is unaffected.
    """
    N, T = strategy_grid
    R_base = model.rate_log10(1, N, T)
    R_disc = model.rate_log10(1, N, T, s_fade=1.0, s_mode="discrete")
    R_cont = model.rate_log10(1, N, T, s_fade=1.0, s_mode="continuous")

    np.testing.assert_array_equal(R_base, R_disc)
    # Continuous at i=1 should still bite (more NaN cells than baseline).
    finite_base = np.isfinite(R_base).sum()
    finite_cont = np.isfinite(R_cont).sum()
    # Either the rate dropped or some cells got nan'd; in any case it can't be identical.
    assert not np.array_equal(R_base, R_cont) or finite_cont < finite_base, (
        "Continuous mode at i=1 should still apply the cut"
    )


# --------------------------------------------------------------------------- #
# F.3 - Monotonic in s_fade                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_mode", ["discrete", "continuous"])
@pytest.mark.parametrize("full", [False, True])
def test_F03_monotone_in_sfade(model, strategy_grid, s_mode, full):
    """log10 R is non-increasing as s_fade sweeps from 0 to 2 mag/day."""
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
                s_fade=float(s), s_mode=s_mode, N_q=500,
            )
        else:
            r = model.rate_log10(
                i_det, np.array([Nv]), np.array([tv]),
                s_fade=float(s), s_mode=s_mode,
            )
        rates.append(float(r[0]))
    arr = np.array(rates)
    finite = arr[np.isfinite(arr)]
    diffs = np.diff(finite)
    assert np.all(diffs <= 1e-10), (
        f"s_mode={s_mode} full={full}: rate increased with s_fade: "
        f"max diff={diffs.max():.3e}, rates={arr}"
    )


# --------------------------------------------------------------------------- #
# F.4 - Closed-form check: at q_s,phase the measured slope equals s_fade       #
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
@pytest.mark.parametrize("s_fade", [0.3, 0.7, 1.5])
def test_F04_discrete_qs_inverts_to_sfade(model, phase, s_fade):
    """Take q_s_phase returned by the helper, plug back into (★) → expect s_fade."""
    i_det = 3
    t_cad_s = float(DAY_S)  # 1 day
    q_s_II, q_s_III = model._q_s_fading_caps(
        i_det, np.array([t_cad_s]), s_fade, "discrete",
    )
    q_s = float(q_s_II[0]) if phase == "II" else float(q_s_III[0])
    # Skip if cap >= phase domain (cut doesn't bite that phase).
    if not math.isfinite(q_s) or q_s <= 1.0:
        pytest.skip(f"phase {phase} cap not in active range at s={s_fade}")
    q_tilde = q_s - 1.0
    s_check = _s_meas_discrete(model, q_tilde, phase, i_det, t_cad_s)
    assert math.isclose(s_check, s_fade, rel_tol=1e-10), (
        f"phase {phase}, s_fade={s_fade}: q_s_check={s_check}, expected {s_fade}"
    )


@pytest.mark.parametrize("phase", ["II", "III"])
@pytest.mark.parametrize("s_fade", [0.3, 0.7, 1.5])
def test_F04b_continuous_qs_inverts_to_sfade(model, phase, s_fade):
    """q_s for s_mode='continuous' should invert (B1) to s_fade exactly."""
    i_det = 3
    t_cad_s = float(DAY_S)
    q_s_II, q_s_III = model._q_s_fading_caps(
        i_det, np.array([t_cad_s]), s_fade, "continuous",
    )
    q_s = float(q_s_II[0]) if phase == "II" else float(q_s_III[0])
    if not math.isfinite(q_s) or q_s <= 1.0:
        pytest.skip(f"phase {phase} cap not in active range at s={s_fade}")
    q_tilde = q_s - 1.0
    s_check = _s_meas_continuous(model, q_tilde, phase)
    assert math.isclose(s_check, s_fade, rel_tol=1e-10), (
        f"phase {phase}, s_fade={s_fade}: q_s_check={s_check}, expected {s_fade}"
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
    # discrepancy is ~E·ln10 / 2 ≈ s_fade·dt/(DAY_S·2·|α|)).
    i_det = 3
    s_fade = 0.5
    p = deep_model.phys.p
    abs_alpha = abs(deep_model.pls.alpha_III_temporal(p))
    # Pick t_cad such that E·ln10 < 0.01 → diff in log10 rate ≪ 0.005.
    # E = s_fade·(i−1)·t_cad / (DAY_S·2.5·|α|);  set t_cad accordingly.
    t_cad_s = 0.001 * (DAY_S * 2.5 * abs_alpha) / (s_fade * (i_det - 1))
    # Sweep N_exp to find a finite cell.
    diffs = []
    for N_exp in np.logspace(0, 5, 60):
        R_disc = float(deep_model.rate_log10(
            i_det, np.array([N_exp]), np.array([t_cad_s]),
            s_fade=s_fade, s_mode="discrete",
        )[0])
        R_cont = float(deep_model.rate_log10(
            i_det, np.array([N_exp]), np.array([t_cad_s]),
            s_fade=s_fade, s_mode="continuous",
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
# F.6 - Helper bypass: s_fade == 0 → +inf caps                                 #
# --------------------------------------------------------------------------- #


def test_F06_helper_bypass_returns_inf(model):
    """At s_fade=0 the helper returns +inf for both phase caps, regardless of mode."""
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
# F.8 - Phase III has a more lenient cap than Phase II at the same s_fade      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_mode", ["discrete", "continuous"])
def test_F08_phaseIII_cap_at_least_phaseII(model, s_mode):
    """Phase III has steeper temporal slope (α_III=-p, |α_II|<|α_III|), so for the
    same fading rate threshold the Phase III viewer space survives further.
    """
    s_fade = 0.5
    i_det = 3
    t_cad_s = DAY_S
    q_s_II, q_s_III = model._q_s_fading_caps(
        i_det, np.array([t_cad_s]), s_fade, s_mode,
    )
    qII = float(q_s_II[0])
    qIII = float(q_s_III[0])
    # If both are finite, expect qIII >= qII (Phase III tolerates larger q).
    if math.isfinite(qII) and math.isfinite(qIII):
        assert qIII >= qII - 1e-12, (
            f"s_mode={s_mode}: q_s_III ({qIII}) should be >= q_s_II ({qII})"
        )


# --------------------------------------------------------------------------- #
# F.9 - Full-integral sits below dominant-term under an active cut             #
# --------------------------------------------------------------------------- #


def test_F09_full_below_dominant_with_cut(deep_model):
    """The dominant-term mode keeps the best-case hard cap (first detection at
    the peak) while the full-integral mode uses the uniform-start survival
    probability P(q) ≤ 1{q ≤ q_s}.  Under an active cut the full-integral rate
    must therefore sit at or below the dominant-term rate (up to the small
    baseline full-vs-dominant tail difference, which raises the full integral
    by design — hence the tolerance from the s_fade=0 offset)."""
    N_grid = np.logspace(0, 5, 100)
    t_grid = np.logspace(np.log10(10), np.log10(86400 * 365), 100)
    N, T = np.meshgrid(N_grid, t_grid, indexing="ij")
    i_det = 3
    masks = deep_model.region_masks(i_det, N, T)
    A1_idx = np.argwhere(masks["A1"])
    if len(A1_idx) == 0:
        pytest.skip("No A1 cells in deep_model grid")
    # Pick the smallest-cadence A1 cell: at very long t_cad the discrete-mode
    # cut correctly annihilates the rate (with the t_dec floor even the
    # earliest possible start fades too slowly), which would NaN the check.
    k = int(np.argmin(T[A1_idx[:, 0], A1_idx[:, 1]]))
    i, j = A1_idx[k]
    Nv = float(N[i, j])
    tv = float(T[i, j])

    # Baseline offset: at A1 (constant D_eff) full vs dominant agree closely
    # at s_fade=0; use it to normalize the comparison.
    R_dom0 = float(deep_model.rate_log10(
        i_det, np.array([Nv]), np.array([tv]),
    )[0])
    R_full0 = float(deep_model.rate_log10_full_integral(
        i_det, np.array([Nv]), np.array([tv]), N_q=500,
    )[0])
    offset = R_full0 - R_dom0

    for s_fade in (0.3, 1.0):
        R_dom = float(deep_model.rate_log10(
            i_det, np.array([Nv]), np.array([tv]),
            s_fade=s_fade, s_mode="discrete",
        )[0])
        R_full = float(deep_model.rate_log10_full_integral(
            i_det, np.array([Nv]), np.array([tv]),
            s_fade=s_fade, s_mode="discrete", N_q=500,
        )[0])
        if not (math.isfinite(R_dom) and math.isfinite(R_full)):
            pytest.skip(f"NaN at A1 with s_fade={s_fade}")
        assert R_full - offset <= R_dom + 1e-9, (
            f"A1 s_fade={s_fade}: full-integral rate {R_full:.4f} (offset "
            f"{offset:.4f}) exceeds dominant-term {R_dom:.4f} — P(q) should "
            f"only remove weight relative to the hard cap"
        )


# --------------------------------------------------------------------------- #
# F.10 - Survival probability P(q): range, bypass, t_cad → 0 hard-cap limit    #
# --------------------------------------------------------------------------- #


def test_F10_survival_range_bypass_and_limit(model):
    q = np.linspace(0.0, float(model.derived.q_nr), 801)

    # Range: P ∈ [0, 1] for both modes at an active cut.
    for s_mode in ("discrete", "continuous"):
        P = model._fading_survival(q, 3, np.array([2.0 * DAY_S]), 0.5, s_mode)
        assert np.all(P >= 0.0) and np.all(P <= 1.0)

    # Bypass: exactly 1.0 everywhere (s_fade=0 any mode; i_det=1 discrete).
    P0 = model._fading_survival(q, 3, np.array([DAY_S]), 0.0, "discrete")
    P1 = model._fading_survival(q, 1, np.array([DAY_S]), 1.0, "discrete")
    assert np.all(P0 == 1.0)
    assert np.all(P1 == 1.0)

    # t_cad → 0 (continuous mode keeps t_f,s fixed): P → 1{q ≤ q_s}.
    t_tiny = np.array([1.0e-3])
    P = model._fading_survival(q, 3, t_tiny, 0.5, "continuous")
    q_s_II, q_s_III = model._q_s_fading_caps(3, t_tiny, 0.5, "continuous")
    q_j = float(model.derived.q_j)
    indicator = np.where(
        q < q_j, (q <= float(q_s_II[0])), (q <= float(q_s_III[0]))
    ).astype(float)
    # Only grid points inside the (vanishing) ramp may differ.
    n_diff = int(np.sum(P != indicator))
    assert n_diff <= 2, f"P differs from the hard-cap indicator at {n_diff} points"


# --------------------------------------------------------------------------- #
# F.11 - Ordering: R_full(s) ≤ R_full(0) and cut only removes weight           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_mode", ["discrete", "continuous"])
def test_F11_full_rate_ordering(model, strategy_grid, s_mode):
    N, T = strategy_grid
    i_det = 3
    R0 = model.rate_log10_full_integral(i_det, N, T, N_q=300)
    R1 = model.rate_log10_full_integral(
        i_det, N, T, s_fade=0.5, s_mode=s_mode, N_q=300,
    )
    both = np.isfinite(R0) & np.isfinite(R1)
    assert np.all(R1[both] <= R0[both] + 1e-12), (
        f"s_mode={s_mode}: full-integral rate increased under the fading cut"
    )


# --------------------------------------------------------------------------- #
# F.12 - Rate-level check against a first-principles P(q) reference            #
# --------------------------------------------------------------------------- #


def test_F12_full_rate_matches_first_principles_weight(model, strategy_grid):
    """Recompute the weighted rate independently: take the s_fade=0 integrand
    from dR_dq_full_integral, multiply by P(q) built from the closed-form
    formulas (not the helper), integrate, and compare to
    rate_log10_full_integral.  Also confirm the on-axis suppression bound
    ρ ≤ max(t_f,s)/t_cad when t_f,s < t_cad."""
    N, T = strategy_grid
    i_det = 3
    s_fade = 0.5
    masks = model.region_masks(i_det, N, T)
    for name in ("A7", "A6", "A5", "A3", "A2"):
        idx = np.argwhere(masks[name])
        if len(idx):
            i, j = idx[len(idx) // 2]
            Nv = float(N[i, j])
            tv = float(T[i, j])
            break
    else:
        pytest.skip("No populated regime cell in strategy_grid")

    # First-principles t_f,s (continuous mode) and P(q).
    p = model.phys.p
    t_j = float(model.derived.t_j_s)
    q_j = float(model.derived.q_j)
    LN10 = math.log(10.0)
    t_f_II = 2.5 * abs(float(model.pls.alpha_II_temporal(p))) * DAY_S / (s_fade * LN10)
    t_f_III = 2.5 * abs(float(model.pls.alpha_III_temporal(p))) * DAY_S / (s_fade * LN10)

    q_vals, dr0 = model.dR_dq_full_integral(i_det, Nv, tv, N_q=500)
    if not np.any(np.isfinite(dr0)):
        pytest.skip("t_exp-invalid cell")
    qt = np.maximum(q_vals - 1.0, 0.0)
    # Effective peak time carries the physical t_dec floor (exact on-axis:
    # t_j·q̃_dec^{8/3} = t_dec identically).
    t_dec = float(model.derived.t_dec_s)
    t_p_II = np.maximum(t_j * qt ** (8.0 / 3.0), t_dec)
    t_p_III = np.maximum(t_j * qt ** 2.0, t_dec)
    P_ref = np.where(
        q_vals < q_j,
        np.clip((t_f_II - t_p_II) / tv, 0.0, 1.0),
        np.clip((t_f_III - t_p_III) / tv, 0.0, 1.0),
    )
    R_ref = math.log10(np.trapezoid(dr0 * P_ref, q_vals))

    R_full = float(model.rate_log10_full_integral(
        i_det, np.array([Nv]), np.array([tv]),
        s_fade=s_fade, s_mode="continuous", N_q=500,
    )[0])
    assert abs(R_full - R_ref) < 0.01, (
        f"{name}: weighted rate {R_full:.4f} vs first-principles {R_ref:.4f}"
    )

    # On-axis suppression: when t_f,s < t_cad, P ≤ max(t_f,s)/t_cad everywhere,
    # so the linear suppression ratio obeys the same bound.
    if max(t_f_II, t_f_III) < tv:
        R0 = float(model.rate_log10_full_integral(
            i_det, np.array([Nv]), np.array([tv]), N_q=500,
        )[0])
        rho = 10.0 ** (R_full - R0)
        assert rho <= max(t_f_II, t_f_III) / tv * 1.001, (
            f"suppression ratio {rho:.4f} exceeds the uniform-start bound "
            f"{max(t_f_II, t_f_III) / tv:.4f}"
        )


# --------------------------------------------------------------------------- #
# F.13 - dR/dq and dR/dD carry the weight correctly                            #
# --------------------------------------------------------------------------- #


def test_F13_differential_views_carry_weight(model):
    i_det = 3
    Nv, tv = 30.0, 2.0 * DAY_S
    s_fade = 0.5

    # dR/dq: weighted output == unweighted output × P pointwise.  The identity
    # is exact mathematically; the two sides associate the triple product
    # (prefactor · q · D³) · P differently, so allow float-roundoff slack.
    q_vals, dr0 = model.dR_dq_full_integral(i_det, Nv, tv)
    _, dr1 = model.dR_dq_full_integral(
        i_det, Nv, tv, s_fade=s_fade, s_mode="discrete",
    )
    P = model._fading_survival(q_vals, i_det, np.array([tv]), s_fade, "discrete")
    np.testing.assert_allclose(dr1, dr0 * P, rtol=1e-13, atol=0.0)

    # dR/dD: P ≤ 1 in the inner q-integral ⇒ pointwise ≤ the unweighted curve.
    D_grid, dD0 = model.dR_dD_full_integral(i_det, Nv, tv)
    _, dD1 = model.dR_dD_full_integral(
        i_det, Nv, tv, s_fade=s_fade, s_mode="discrete",
    )
    both = np.isfinite(dD0) & np.isfinite(dD1)
    assert np.all(dD1[both] <= dD0[both] + 1e-300)


# --------------------------------------------------------------------------- #
# F.14 - Numerical medians under the weight                                    #
# --------------------------------------------------------------------------- #


def test_F14_numerical_medians_shift_and_nan(model):
    i_det = 3
    N = np.array([30.0])
    T = np.array([2.0 * DAY_S])

    # The weight removes late-peaking (large-q) bursts first → q_med decreases.
    q0, D0 = model.compute_medians(i_det, N, T, full_integral=True)
    q1, D1 = model.compute_medians(
        i_det, N, T, full_integral=True, s_fade=0.5, s_mode="discrete",
    )
    if not (np.isfinite(q0[0]) and np.isfinite(q1[0])):
        pytest.skip("t_exp-invalid cell")
    assert q1[0] <= q0[0] + 1e-12

    # Kill all weight: q_min above the zero-survival angle q_s → NaN medians.
    q_s_II, q_s_III = model._q_s_fading_caps(
        i_det, T, 2.0, "discrete",
    )
    q_kill = float(max(q_s_II[0], q_s_III[0])) + 0.5
    if not math.isfinite(q_kill):
        pytest.skip("cut bypassed — cannot construct zero-weight cell")
    q2, D2 = model.compute_medians(
        i_det, N, T, full_integral=True,
        q_min=q_kill, s_fade=2.0, s_mode="discrete",
    )
    assert np.isnan(q2[0]) and np.isnan(D2[0])
