"""Engine tests for the intrinsic luminosity function (LF).

The LF replaces the single-luminosity assumption with a truncated power law
φ(L) ∝ L^α on [L_min, L_max], L ≡ νL_ν(1 day).  Everything below leans on the
pinned invariant that luminosity enters the engine only through F_lim/F_dec
(tests/test_fdec_override.py), so the brute-force reference is a dense log-s
trapezoid over the legacy single-L engine evaluated on scaled-derived copies.

Covers:
  - closed-form dominant-term LF rate vs. the dense reference (exact)
  - closed-form full-integral LF rate vs. the dense reference (exact),
    including the rise-cut ramp decomposition
  - α-singular exponents (α = −1, −5/2) via the expm1-stable power bracket
  - off-state structural parity and the degenerate L_min = L_max limit
  - invariance under the copy-on-write F_dec override while the LF is on
  - monotonicity in L_min / L_max
  - mixture medians vs. a stacked per-s differential reference
  - dominant-contribution regime ids vs. a per-s reference
  - q-chunking parity under the LF
"""

from __future__ import annotations

import numpy as np
import pytest

import grb_detect.detection_rate as dr
from grb_detect.core import make_rate_model
from grb_detect.detection_rate import _pow_bracket

BASE = dict(A_log=-4.68, f_live=0.1, t_overhead_s=0.0, omega_exp_deg2=47.0)

# Strategy points spanning the regime families (N_exp, t_cad_s).
N_PTS = np.array([1.0, 30.0, 585.0, 585.0, 100.0, 585.0])
T_PTS = np.array([3e2, 3.6e3, 8.64e4, 3e6, 1e5, 1e3])

LF_DEFAULT = dict(lf_on=True, lf_alpha=-2.0, lf_log10_L_min=42.5,
                  lf_log10_L_max=45.5)

# Physics changes that affect the peak-to-one-day conversion, including
# density/angle choices that put the jet break before one day.
ONE_DAY_PHYSICS = [
    pytest.param({}, id="default"),
    pytest.param(dict(p=2.6), id="p"),
    pytest.param(dict(E_kiso_log10=54.0), id="energy"),
    pytest.param(dict(n0_log10=2.0), id="density-early-jet"),
    pytest.param(dict(gamma0_log10=3.0), id="gamma"),
    pytest.param(dict(theta_j_rad=0.05), id="angle-early-jet"),
    pytest.param(dict(theta_j_rad=0.15), id="angle-late-jet"),
]


def _lin(logR):
    return np.where(np.isfinite(logR), 10.0 ** logR, 0.0)


def _ref_rate(model_off, i_det, N, t, alpha, lmin, lmax, *, full,
              n_s=1500, **kw):
    """Dense log-s trapezoid over the legacy engine (scaled-derived copies)."""
    L0 = model_off.L0_erg_s()
    s1, s2 = 10.0 ** lmin / L0, 10.0 ** lmax / L0
    ln_s = np.linspace(np.log(s1), np.log(s2), n_s)
    s_vals = np.exp(ln_s)
    w = s_vals ** (alpha + 1.0)
    norm = np.trapezoid(w, ln_s)
    vals = []
    for s in s_vals:
        m = model_off._lf_scaled_copy(s)
        lr = (m.rate_log10_full_integral(i_det, N, t, **kw) if full
              else m.rate_log10(i_det, N, t, **kw))
        vals.append(_lin(lr))
    vals = np.stack(vals).reshape(n_s, -1)
    out = np.trapezoid((w[:, None] / norm) * vals, ln_s, axis=0)
    return out.reshape(np.broadcast(N, t).shape)


def _rel_err(got, want):
    denom = np.maximum(np.maximum(got, want), 1e-300)
    both_zero = (got <= 0) & (want <= 1e-10 * float(np.nanmax(want) + 1e-300))
    return np.where(both_zero, 0.0, np.abs(got - want) / denom)


# ---------------------------------------------------------------------------
# The power bracket
# ---------------------------------------------------------------------------

def test_pow_bracket_basic_and_singular():
    # plain power
    assert np.isclose(float(_pow_bracket(1.0, 2.0, 2.0)), (8.0 - 1.0) / 3.0)
    # e = −1 exactly → log
    assert np.isclose(float(_pow_bracket(1.0, np.e, -1.0)), 1.0)
    # near-singular continuity
    a = float(_pow_bracket(0.5, 7.0, -1.0))
    b = float(_pow_bracket(0.5, 7.0, -1.0 + 1e-13))
    assert np.isclose(a, b, rtol=1e-9)
    # empty bracket
    assert float(_pow_bracket(2.0, 1.0, 0.5)) == 0.0


@pytest.mark.parametrize("physics", ONE_DAY_PHYSICS)
def test_L0_matches_validation_formula(physics):
    """One-day LF normalization follows the current on-axis light curve."""
    m = make_rate_model(**BASE, **physics)
    # Evaluate the PLS-G normalization directly at one day, independently
    # of the engine's extrapolation from the derived peak flux.
    F_1d = m.pls.F_dec_Jy(m.phys, m.micro, dr.DAY_S)
    # After the jet break, the temporal index steepens from -3(p-1)/4 to -p.
    F_1d *= min(1.0, (m.derived.t_j_s / dr.DAY_S) ** ((m.phys.p + 3.0) / 4.0))
    want = m.phys.nu_hz * 4.0 * np.pi * m.phys.D_euc_cm ** 2 * F_1d * 1e-23
    assert np.isclose(m.L0_erg_s(), want, rtol=1e-12)


# ---------------------------------------------------------------------------
# Closed forms vs. dense reference — the physics-correctness anchor
# ---------------------------------------------------------------------------

FILTER_CASES = [
    dict(),
    dict(q_min=2.0),
    dict(D_min_cm=3.0857e27),                       # 1 Gpc floor
    dict(s_fade=0.5),
    dict(s_rise=0.5),
    dict(s_fade=0.3, s_rise=0.3, q_min=1.5, D_min_cm=1.5e27),
    dict(s_fade=0.5, s_mode="continuous"),
]


@pytest.mark.parametrize("full", [False, True], ids=["dominant", "full"])
@pytest.mark.parametrize("alpha", [-3.5, -2.5, -2.0, -1.0, 0.0])
def test_lf_rate_matches_reference(full, alpha):
    """Closed-form LF rate vs. dense per-s reference, α-singular hits included.

    The residual is the reference's own trapezoid discretization (~1e-4 at
    n_s = 1500 over kinked integrands), far below the app's 5% budget — the
    closed forms themselves are exact.
    """
    for lmin, lmax in [(42.5, 45.5), (41.0, 47.0)]:
        for kw in FILTER_CASES:
            m_on = make_rate_model(**BASE, lf_on=True, lf_alpha=alpha,
                                   lf_log10_L_min=lmin, lf_log10_L_max=lmax)
            m_off = make_rate_model(**BASE)
            lr = (m_on.rate_log10_full_integral(2, N_PTS, T_PTS, **kw) if full
                  else m_on.rate_log10(2, N_PTS, T_PTS, **kw))
            got = _lin(lr)
            want = _ref_rate(m_off, 2, N_PTS, T_PTS, alpha, lmin, lmax,
                             full=full, **kw)
            err = float(np.nanmax(_rel_err(got, want)))
            assert err < 2e-3, f"L=({lmin},{lmax}) kw={kw}: rel err {err:.2e}"


def test_lf_rate_window_flags_match_reference():
    """win_i_minus_one / win_from_peak compose with the LF."""
    for flags in [dict(win_i_minus_one=True), dict(win_from_peak=True),
                  dict(win_i_minus_one=True, win_from_peak=True)]:
        m_on = make_rate_model(**BASE, **flags, **LF_DEFAULT)
        m_off = make_rate_model(**BASE, **flags)
        # win_from_peak routes rate_log10 through the full integral internally.
        got = _lin(m_on.rate_log10(2, N_PTS, T_PTS, s_rise=0.3, s_fade=0.3))
        want = _ref_rate(m_off, 2, N_PTS, T_PTS, -2.0, 42.5, 45.5,
                         full=bool(flags.get("win_from_peak")),
                         s_rise=0.3, s_fade=0.3)
        err = float(np.nanmax(_rel_err(got, want)))
        assert err < 2e-3, f"{flags}: rel err {err:.2e}"


def test_lf_hard_boundary_flags_match_reference():
    """rise/fade random-start=False variants stay exact under the LF."""
    kw = dict(s_rise=0.4, s_fade=0.4)
    for rs, fs in [(False, True), (True, False), (False, False)]:
        m_on = make_rate_model(**BASE, **LF_DEFAULT)
        m_off = make_rate_model(**BASE)
        got = _lin(m_on.rate_log10_full_integral(
            2, N_PTS, T_PTS, rise_random_start=rs, fade_random_start=fs, **kw))
        want = _ref_rate(m_off, 2, N_PTS, T_PTS, -2.0, 42.5, 45.5, full=True,
                         rise_random_start=rs, fade_random_start=fs, **kw)
        err = float(np.nanmax(_rel_err(got, want)))
        assert err < 2e-3, f"rs={rs} fs={fs}: rel err {err:.2e}"


# ---------------------------------------------------------------------------
# Structural parity, degenerate limit, override invariance, monotonicity
# ---------------------------------------------------------------------------

def test_off_state_is_same_cached_model():
    """lf_on=False must resolve to the very same cached model as no lf kwargs."""
    assert make_rate_model(**BASE) is make_rate_model(**BASE, lf_on=False)
    m = make_rate_model(**BASE)
    assert m.lf is None


@pytest.mark.parametrize("physics", ONE_DAY_PHYSICS)
def test_degenerate_limit_reproduces_single_L(physics):
    """L_min = L_max = log10 L0 reproduces the legacy single-L rate exactly."""
    m_off = make_rate_model(**BASE, **physics)
    lL0 = np.log10(m_off.L0_erg_s())
    m_deg = make_rate_model(**BASE, **physics, lf_on=True, lf_alpha=-2.0,
                            lf_log10_L_min=lL0, lf_log10_L_max=lL0)
    for full in (False, True):
        for kw in [dict(), dict(s_fade=0.3, s_rise=0.3, q_min=1.5)]:
            f = (m_deg.rate_log10_full_integral if full else m_deg.rate_log10)
            g = (m_off.rate_log10_full_integral if full else m_off.rate_log10)
            got, want = _lin(f(2, N_PTS, T_PTS, **kw)), _lin(g(2, N_PTS, T_PTS, **kw))
            assert float(np.nanmax(_rel_err(got, want))) < 1e-6


def test_narrow_bracket_converges_to_single_L():
    m_off = make_rate_model(**BASE)
    lL0 = np.log10(m_off.L0_erg_s())
    m_nar = make_rate_model(**BASE, lf_on=True, lf_alpha=-2.0,
                            lf_log10_L_min=lL0 - 0.01, lf_log10_L_max=lL0 + 0.01)
    got = _lin(m_nar.rate_log10(2, N_PTS, T_PTS))
    want = _lin(m_off.rate_log10(2, N_PTS, T_PTS))
    assert float(np.nanmax(_rel_err(got, want))) < 5e-3


def test_fdec_override_is_noop_under_lf():
    """s-bounds are computed lazily from the derived scales, so a copy-on-write
    F_dec override rescales F_dec,0 and L0 together — the LF-on rate is
    invariant (per-burst flux depends on L alone)."""
    m = make_rate_model(**BASE, **LF_DEFAULT)
    m_ov = m._lf_scaled_copy(7.3)          # the override mechanism, k = 7.3
    m_ov.lf = m.lf                          # keep the LF on the copy
    for full in (False, True):
        f = (m.rate_log10_full_integral if full else m.rate_log10)
        g = (m_ov.rate_log10_full_integral if full else m_ov.rate_log10)
        a = f(2, N_PTS, T_PTS, s_fade=0.3, s_rise=0.3)
        b = g(2, N_PTS, T_PTS, s_fade=0.3, s_rise=0.3)
        finite = np.isfinite(a) | np.isfinite(b)
        assert np.allclose(a[finite], b[finite], rtol=1e-10, equal_nan=True)


def test_monotone_in_L_bounds():
    """Raising L_max or L_min shifts the (normalized) population brighter, so
    the rate can only rise."""
    kwargs = dict(**BASE, lf_on=True, lf_alpha=-2.0)
    prev = None
    for lmax in (43.5, 45.5, 47.0):
        m = make_rate_model(**kwargs, lf_log10_L_min=42.5, lf_log10_L_max=lmax)
        R = _lin(m.rate_log10(2, N_PTS, T_PTS))
        if prev is not None:
            assert np.all(R >= prev * (1 - 1e-12))
        prev = R
    prev = None
    for lmin in (41.0, 42.5, 44.0):
        m = make_rate_model(**kwargs, lf_log10_L_min=lmin, lf_log10_L_max=45.5)
        R = _lin(m.rate_log10(2, N_PTS, T_PTS))
        if prev is not None:
            assert np.all(R >= prev * (1 - 1e-12))
        prev = R


# ---------------------------------------------------------------------------
# Mixture medians and differential distributions
# ---------------------------------------------------------------------------

def _ref_marginals(model_off, i_det, Nv, tv, alpha, lmin, lmax, n_s=200, **kw):
    """Stacked per-s dR/dq and dR/dD arrays, φ-weighted (grids are node-invariant)."""
    L0 = model_off.L0_erg_s()
    s1, s2 = 10.0 ** lmin / L0, 10.0 ** lmax / L0
    ln_s = np.linspace(np.log(s1), np.log(s2), n_s)
    s_vals = np.exp(ln_s)
    w = s_vals ** (alpha + 1.0)
    w = w / np.trapezoid(w, ln_s)
    dq_acc = dd_acc = 0.0
    dls = ln_s[1] - ln_s[0]
    for s, wt in zip(s_vals, w):
        m = model_off._lf_scaled_copy(s)
        q_grid, dq = m.dR_dq_full_integral(i_det, Nv, tv, **kw)
        D_grid, dd = m.dR_dD_full_integral(i_det, Nv, tv, **kw)
        dq_acc = dq_acc + wt * dls * dq
        dd_acc = dd_acc + wt * dls * dd
    return q_grid, dq_acc, D_grid, dd_acc


def _median_from_pdf(x, pdf):
    c = np.cumsum(pdf)
    return float(x[np.argmax(c / c[-1] >= 0.5)])


@pytest.mark.parametrize("kw", [dict(), dict(s_fade=0.3, s_rise=0.3)],
                         ids=["plain", "cuts"])
def test_lf_mixture_medians_match_stacked_reference(kw):
    m_on = make_rate_model(**BASE, **LF_DEFAULT)
    m_off = make_rate_model(**BASE)
    for Nv, tv in [(585.0, 8.64e4), (100.0, 1e5), (585.0, 1e3)]:
        q_med, D_med_cm = m_on.compute_medians(
            2, np.array([Nv]), np.array([tv]), full_integral=True, **kw)
        q_grid, dq, D_grid, dd = _ref_marginals(
            m_off, 2, Nv, tv, -2.0, 42.5, 45.5, **kw)
        if not np.isfinite(q_med[0]):
            assert float(np.nansum(dq)) < 1e-12
            continue
        q_ref = _median_from_pdf(q_grid, dq)
        D_ref = _median_from_pdf(D_grid, dd)
        assert abs(q_med[0] - q_ref) < 0.06 * q_grid[-1], (Nv, tv)
        assert abs(D_med_cm[0] - D_ref) < 0.05 * D_grid[-1], (Nv, tv)


@pytest.mark.parametrize("kw", [dict(), dict(s_rise=0.4, s_fade=0.2)],
                         ids=["plain", "cuts"])
def test_lf_dR_views_match_stacked_reference(kw):
    """dR/dq and dR/dD under the LF equal the φ-weighted stacked per-s views."""
    m_on = make_rate_model(**BASE, **LF_DEFAULT)
    m_off = make_rate_model(**BASE)
    Nv, tv = 585.0, 8.64e4
    q_grid, dq_ref, D_grid, dd_ref = _ref_marginals(
        m_off, 2, Nv, tv, -2.0, 42.5, 45.5, n_s=800, **kw)
    _, dq = m_on.dR_dq_full_integral(2, Nv, tv, **kw)
    _, dd = m_on.dR_dD_full_integral(2, Nv, tv, **kw)
    # integral agreement (pointwise has reference-quadrature noise at kinks)
    Iq, Iq_ref = np.trapezoid(dq, q_grid), np.trapezoid(dq_ref, q_grid)
    Id, Id_ref = np.trapezoid(dd, D_grid), np.trapezoid(dd_ref, D_grid)
    assert np.isclose(Iq, Iq_ref, rtol=5e-3)
    assert np.isclose(Id, Id_ref, rtol=5e-3)
    # pointwise agreement away from zero
    mask = dq_ref > 1e-3 * dq_ref.max()
    assert float(np.nanmax(np.abs(dq[mask] - dq_ref[mask]) / dq_ref[mask])) < 2e-2
    mask = dd_ref > 1e-3 * dd_ref.max()
    assert float(np.nanmax(np.abs(dd[mask] - dd_ref[mask]) / dd_ref[mask])) < 2e-2


def test_lf_regime_ids_match_reference():
    """Dominant-contribution regime = argmax of per-s φ-weighted contributions."""
    m_on = make_rate_model(**BASE, **LF_DEFAULT)
    m_off = make_rate_model(**BASE)
    L0 = m_off.L0_erg_s()
    s1, s2 = 10.0 ** 42.5 / L0, 10.0 ** 45.5 / L0
    ln_s = np.linspace(np.log(s1), np.log(s2), 800)
    s_vals = np.exp(ln_s)
    w = s_vals ** (-2.0 + 1.0)
    w = w / np.trapezoid(w, ln_s)
    dls = ln_s[1] - ln_s[0]

    rid_lf = m_on.regime_id_lf(2, N_PTS, T_PTS)
    acc = np.zeros((7,) + N_PTS.shape)
    for s, wt in zip(s_vals, w):
        m = m_off._lf_scaled_copy(s)
        R = _lin(m.rate_log10(2, N_PTS, T_PTS))
        masks = m.region_masks(2, N_PTS, T_PTS)
        for k, key in enumerate(("A1", "A2", "A3", "A4", "A5", "A6", "A7")):
            acc[k] += np.where(masks[key], wt * dls * R, 0.0)
    rid_ref = np.argmax(acc, axis=0) + 1.0
    total = acc.sum(axis=0)
    ok = total > 0
    # allow disagreement only where the top two contributions are within 5%
    for i in np.nonzero(ok)[0]:
        if rid_lf[i] != rid_ref[i]:
            top = np.sort(acc[:, i])[::-1]
            assert top[1] > 0.95 * top[0], (i, rid_lf[i], rid_ref[i])


# ---------------------------------------------------------------------------
# Chunking parity under the LF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(), dict(s_rise=0.4, s_fade=0.3)],
                         ids=["plain", "rise"])
def test_lf_chunked_equals_unchunked(kw, monkeypatch):
    m = make_rate_model(**BASE, **LF_DEFAULT)
    big = m.rate_log10_full_integral(2, N_PTS, T_PTS, **kw)
    monkeypatch.setattr(dr, "_FULL_INTEGRAL_CHUNK_ELEMS", 64)
    small = m.rate_log10_full_integral(2, N_PTS, T_PTS, **kw)
    finite = np.isfinite(big) | np.isfinite(small)
    assert np.allclose(big[finite], small[finite], rtol=1e-12, equal_nan=True)
