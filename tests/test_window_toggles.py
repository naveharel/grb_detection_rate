"""Physics verification tests for the detection-window settings.

Two independent, off-by-default settings refine how the "≥ i detections
within the cadence" requirement is counted:

  * ``win_i_minus_one`` — the required duration is T_req = (i−1)·t_cad (the
    gaps between i detections) instead of i·t_cad, everywhere T_req enters
    (q_i, D_i, window-mode D_eff).  Exactly equivalent to the legacy mode
    evaluated at i−1.
  * ``win_from_peak`` — the window is measured from the light-curve turn-on
    t_on ≈ t_p(q) instead of from the burst: t_+(D) ≥ t_p,eff(q) + T_req.
    The cadence distance cap becomes q-dependent (no dominant-term
    rectangle), so the rate is evaluated with the q-integral internally.

Both on together give the corrected criterion
t_+(D) − t_p,eff(q) ≥ (i−1)·t_cad.

Run with::

    .venv/Scripts/python -m pytest tests/test_window_toggles.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from grb_detect.constants import DAY_S
from grb_detect.detection_rate import DetectionRateModel
from grb_detect.params import AfterglowPhysicalParams, SurveyInstrumentParams


def _model(**flags) -> DetectionRateModel:
    return DetectionRateModel(
        AfterglowPhysicalParams(), SurveyInstrumentParams(), **flags
    )


@pytest.fixture(scope="module")
def legacy() -> DetectionRateModel:
    return _model()


@pytest.fixture(scope="module")
def m_iminus1() -> DetectionRateModel:
    return _model(win_i_minus_one=True)


@pytest.fixture(scope="module")
def m_frompeak() -> DetectionRateModel:
    return _model(win_from_peak=True)


@pytest.fixture(scope="module")
def m_both() -> DetectionRateModel:
    return _model(win_i_minus_one=True, win_from_peak=True)


@pytest.fixture(scope="module")
def grid() -> tuple[np.ndarray, np.ndarray]:
    N = np.logspace(0, 3.5, 25)
    t = np.logspace(np.log10(600.0), np.log10(30 * DAY_S), 25)
    return np.meshgrid(N, t, indexing="ij")


def _nan_eq(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    m = np.isfinite(a)
    np.testing.assert_array_equal(a[m], b[m])


# ── Parity: both settings off is bit-identical to the default constructor ──

def test_flags_off_parity_dominant(legacy, grid):
    N, t = grid
    explicit = _model(win_i_minus_one=False, win_from_peak=False)
    _nan_eq(legacy.rate_log10(3, N, t), explicit.rate_log10(3, N, t))


def test_flags_off_parity_full_integral(legacy, grid):
    N, t = grid
    explicit = _model(win_i_minus_one=False, win_from_peak=False)
    _nan_eq(
        legacy.rate_log10_full_integral(3, N, t),
        explicit.rate_log10_full_integral(3, N, t),
    )


def test_flags_off_parity_medians(legacy, grid):
    N, t = grid
    explicit = _model(win_i_minus_one=False, win_from_peak=False)
    for full in (False, True):
        q0, D0 = legacy.compute_medians(3, N, t, full_integral=full)
        q1, D1 = explicit.compute_medians(3, N, t, full_integral=full)
        _nan_eq(q0, q1)
        _nan_eq(D0, D1)


# ── win_i_minus_one: exact shift identity and monotonicity ──────────────────

def test_iminus1_equals_legacy_at_i_minus_1(legacy, m_iminus1, grid):
    N, t = grid
    for i_det in (2, 3, 6):
        _nan_eq(
            m_iminus1.rate_log10(i_det, N, t),
            legacy.rate_log10(i_det - 1, N, t),
        )
        _nan_eq(
            m_iminus1.rate_log10_full_integral(i_det, N, t),
            legacy.rate_log10_full_integral(i_det - 1, N, t),
        )


def test_iminus1_qi_Di_shift(legacy, m_iminus1):
    t = np.logspace(2, 7, 40)
    F = np.full_like(t, 1e-5)
    np.testing.assert_array_equal(m_iminus1.q_i(4, t), legacy.q_i(3, t))
    np.testing.assert_array_equal(m_iminus1.D_i(4, t, F), legacy.D_i(3, t, F))


def test_iminus1_never_decreases_rate(legacy, m_iminus1, grid):
    N, t = grid
    r0 = legacy.rate_log10(3, N, t)
    r1 = m_iminus1.rate_log10(3, N, t)
    m = np.isfinite(r0) & np.isfinite(r1)
    assert np.all(r1[m] >= r0[m] - 1e-12)


def test_iminus1_i1_degenerate(m_iminus1):
    """i_det = 1 under the setting: T_req = 0 — cadence constraint vanishes."""
    t = np.logspace(2, 7, 10)
    F = np.full_like(t, 1e-5)
    np.testing.assert_array_equal(m_iminus1.q_i(1, t), np.ones_like(t))
    assert np.all(np.isinf(m_iminus1.D_i(1, t, F)))
    N = np.full_like(t, 100.0)
    assert np.all(np.isfinite(m_iminus1.rate_log10(1, N, t)))


# ── D_from_t_plus: the one relation behind both rectangle boundaries ────────

def test_D_from_t_plus_reproduces_D_i(legacy):
    """At t_+ = T_req the inversion equals D_i in both phases."""
    t_j = legacy.derived.t_j_s
    # Phase II (i·t_cad < t_j) and phase III (i·t_cad > t_j) required durations
    t_cad = np.array([0.05 * t_j / 3.0, 20.0 * t_j / 3.0])
    F = np.full_like(t_cad, 1e-5)
    T_req = 3.0 * t_cad
    np.testing.assert_allclose(
        legacy.D_from_t_plus(T_req, F),
        legacy.D_i(3, t_cad, F),
        rtol=1e-12,
    )


def test_D_from_t_plus_reproduces_D_max(legacy):
    """At t_+ = t_p(q) the inversion equals the flux-limited D_max(q)."""
    p = legacy.phys.p
    aII = float(legacy.pls.a_II(p))
    aIII = float(legacy.pls.a_III(p))
    q_dec = legacy.derived.q_dec
    qd_t = q_dec - 1.0
    q_j = legacy.derived.q_j

    q = np.array([1.5 * (q_dec - 1.0) + 1.0, 5.0])  # one phase-II, one phase-III
    F = np.full_like(q, 1e-5)
    D_dec = legacy.D_dec(F)
    qt = q - 1.0
    D_max = np.where(
        q < q_j,
        D_dec * (qt / qd_t) ** (-aII),
        D_dec * qd_t ** (-1.0) * (qt / qd_t) ** (-aIII),
    )
    np.testing.assert_allclose(
        legacy.D_from_t_plus(legacy._t_p_eff(q), F), D_max, rtol=1e-10
    )


def test_D_from_t_plus_edge_cases(legacy):
    F = np.array([1e-5, 1e-5, 1e-5])
    t_dec = legacy.derived.t_dec_s
    out = legacy.D_from_t_plus(np.array([0.0, t_dec, 10 * t_dec]), F)
    assert np.isinf(out[0])                      # no duration → no constraint
    D_dec = float(legacy.D_dec(np.array([1e-5]))[0])
    np.testing.assert_allclose(out[1], D_dec, rtol=1e-12)   # t_+ = t_dec endpoint
    assert out[2] < D_dec                        # longer window → closer cap


# ── win_from_peak: suppression, limits, routing ─────────────────────────────

def test_frompeak_never_increases_rate(legacy, m_frompeak, grid):
    N, t = grid
    r0 = legacy.rate_log10_full_integral(3, N, t)
    r1 = m_frompeak.rate_log10_full_integral(3, N, t)
    m = np.isfinite(r0) & np.isfinite(r1)
    assert np.all(r1[m] <= r0[m] + 1e-12)
    assert np.any(r1[m] < r0[m] - 1e-6)  # strictly suppressed somewhere


def test_frompeak_tcad_to_zero_limit(legacy, m_frompeak):
    """T_req → 0: the window cap D_w(q) → D_max(q) and both modes agree."""
    N = np.array([100.0])
    t = np.array([1e-3])
    r0 = legacy.rate_log10_full_integral(2, N, t)
    r1 = m_frompeak.rate_log10_full_integral(2, N, t)
    np.testing.assert_allclose(r1, r0, rtol=1e-6)


def test_frompeak_dominant_routes_to_full_integral(m_frompeak, grid):
    N, t = grid
    _nan_eq(
        m_frompeak.rate_log10(3, N, t),
        m_frompeak.rate_log10_full_integral(3, N, t),
    )


def test_frompeak_components_keep_rectangle_defs(legacy, m_frompeak):
    N = np.array([100.0])
    t = np.array([2.0 * DAY_S])
    logR, comps = m_frompeak.rate_log10(3, N, t, return_components=True)
    _, comps0 = legacy.rate_log10(3, N, t, return_components=True)
    np.testing.assert_array_equal(comps["q_i"], comps0["q_i"])
    np.testing.assert_array_equal(comps["D_i_cm"], comps0["D_i_cm"])
    np.testing.assert_array_equal(
        logR, m_frompeak.rate_log10_full_integral(3, N, t)
    )


def test_frompeak_medians_forced_numerical(m_frompeak):
    N = np.array([100.0])
    t = np.array([2.0 * DAY_S])
    q_a, D_a = m_frompeak.compute_medians(3, N, t, full_integral=False, N_q=200)
    q_n, D_n = m_frompeak.compute_medians_numerical(3, N, t, 200)
    _nan_eq(q_a, q_n)
    _nan_eq(D_a, D_n)


def test_frompeak_shifts_population_on_axis(legacy, m_frompeak):
    """The spurious off-axis corner (q ≈ q_i, window ≈ 0) is removed, so the
    detected-population median angle moves inward."""
    N = np.array([300.0])
    t = np.array([2.0 * DAY_S])
    q0, _ = legacy.compute_medians(2, N, t, full_integral=True, N_q=400)
    q1, _ = m_frompeak.compute_medians(2, N, t, full_integral=True, N_q=400)
    assert np.isfinite(q0[0]) and np.isfinite(q1[0])
    assert q1[0] <= q0[0] + 1e-9


def test_both_flags_equal_frompeak_at_i_minus_1(legacy, m_both, m_frompeak, grid):
    N, t = grid
    _nan_eq(
        m_both.rate_log10_full_integral(3, N, t),
        m_frompeak.rate_log10_full_integral(2, N, t),
    )


def test_frompeak_with_filters_composes(m_frompeak, grid):
    N, t = grid
    r_plain = m_frompeak.rate_log10_full_integral(3, N, t)
    r_filt = m_frompeak.rate_log10_full_integral(
        3, N, t, s_fade=0.3, s_rise=0.5
    )
    m = np.isfinite(r_plain) & np.isfinite(r_filt)
    assert np.all(r_filt[m] <= r_plain[m] + 1e-12)


def test_frompeak_dRdq_dRdD_run_and_integrate(m_frompeak):
    """dR/dq and dR/dD carry the q-dependent window cap and integrate to R."""
    N, t = 100.0, 2.0 * DAY_S
    logR = float(m_frompeak.rate_log10_full_integral(
        3, np.array([N]), np.array([t]))[0])
    q, dRdq = m_frompeak.dR_dq_full_integral(3, N, t, N_q=500)
    R_q = float(np.trapezoid(dRdq, q))
    D, dRdD = m_frompeak.dR_dD_full_integral(3, N, t, N_q=500, N_D=400)
    R_D = float(np.trapezoid(dRdD, D))
    R = 10.0 ** logR
    assert np.isfinite(R) and R > 0
    np.testing.assert_allclose(R_q, R, rtol=1e-2)
    np.testing.assert_allclose(R_D, R, rtol=2e-2)
