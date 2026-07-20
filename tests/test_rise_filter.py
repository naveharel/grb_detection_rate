"""Physics verification tests for the rise-rate filter (s_rise).

A previous-visit non-detection at the limiting magnitude implies a rise rate
of at least (m_lim − m_first)/t_cad; requiring that to exceed s_rise [mag/day]
is equivalent to F(t_first) ≥ η·F_lim with η = 10^(s_rise·t_cad/(2.5·DAY_S)).

The two rate modes treat the first-detection time differently:
  * Dominant-term (closed-form): best-case start at the peak → the raised
    limit shifts the flux-limited boundaries (qE_r, q_ri, D_dec·η^{−1/2});
    each of r1-r6 is the max of the angular-truncated term and an on-axis
    term so the deep-cut limit stays correct (no saturation plateau).
  * Full-integral: uniform start t_first = t_p,eff + u, u ~ U(0, t_cad) →
    joint fade+rise survival weight w = clip(min(t_fs, t_lim) − t_p,eff, 0,
    t_cad)/t_cad folded into a closed-form weighted D-volume V_w.  Both cuts
    share the SAME start offset u (comonotone), so the joint window is the
    min, not a product.

Unlike the discrete fade filter, the rise cut applies at i_det = 1 (it needs
only the previous visit + the first detection).

Run with::

    .venv/Scripts/python -m pytest tests/test_rise_filter.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.detection_rate import DetectionRateModel
from grb_detect.params import (
    CM_TO_GPC,
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


def _F_p_at_Deuc(model, q):
    """First-principles off-axis peak flux at D_euc (PLS module docstring law)."""
    p = model.phys.p
    aII = float(model.pls.a_II(p))
    aIII = float(model.pls.a_III(p))
    F_dec = float(model.derived.F_dec_Jy)
    qd_t = float(model.derived.q_dec) - 1.0
    q_j = float(model.derived.q_j)
    qt = q - 1.0
    if q < q_j:
        return F_dec * (qt / qd_t) ** (-2.0 * aII)
    return F_dec * qd_t ** (-2.0) * (qt / qd_t) ** (-2.0 * aIII)


# --------------------------------------------------------------------------- #
# R.1 - s_rise = 0 is bit-identical to the fade-only baseline                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_fade", [0.0, 0.5])
@pytest.mark.parametrize("full", [False, True])
def test_R01_srise_zero_matches_baseline(model, strategy_grid, s_fade, full):
    N, T = strategy_grid
    i_det = 3
    if full:
        R_base = model.rate_log10_full_integral(i_det, N, T, s_fade=s_fade, N_q=300)
        R_cut = model.rate_log10_full_integral(
            i_det, N, T, s_fade=s_fade, s_rise=0.0, N_q=300,
        )
    else:
        R_base = model.rate_log10(i_det, N, T, s_fade=s_fade)
        R_cut = model.rate_log10(i_det, N, T, s_fade=s_fade, s_rise=0.0)
    np.testing.assert_array_equal(R_base, R_cut)


def test_R01b_srise_zero_derived_quantities(model, strategy_grid):
    """Medians, dR/dq and dR/dD at s_rise=0 match the fade-only calls exactly."""
    N, T = strategy_grid
    i_det = 3
    for full in (False, True):
        q_a, D_a = model.compute_medians(i_det, N, T, full_integral=full, s_fade=0.4)
        q_b, D_b = model.compute_medians(
            i_det, N, T, full_integral=full, s_fade=0.4, s_rise=0.0,
        )
        np.testing.assert_array_equal(q_a, q_b)
        np.testing.assert_array_equal(D_a, D_b)

    Nv, tv = 30.0, 2.0 * DAY_S
    _, dr_a = model.dR_dq_full_integral(i_det, Nv, tv, s_fade=0.4)
    _, dr_b = model.dR_dq_full_integral(i_det, Nv, tv, s_fade=0.4, s_rise=0.0)
    np.testing.assert_array_equal(dr_a, dr_b)
    _, dD_a = model.dR_dD_full_integral(i_det, Nv, tv, s_fade=0.4)
    _, dD_b = model.dR_dD_full_integral(i_det, Nv, tv, s_fade=0.4, s_rise=0.0)
    np.testing.assert_array_equal(dD_a, dD_b)


# --------------------------------------------------------------------------- #
# R.2 - Monotone non-increasing in s_rise                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("full", [False, True])
def test_R02_monotone_in_srise(model, strategy_grid, full):
    N, T = strategy_grid
    i_det = 3
    prev = None
    for s in np.linspace(0.0, 2.0, 9):
        if full:
            R = model.rate_log10_full_integral(i_det, N, T, s_rise=float(s), N_q=300)
        else:
            R = model.rate_log10(i_det, N, T, s_rise=float(s))
        if prev is not None:
            both = np.isfinite(R) & np.isfinite(prev)
            assert np.all(R[both] <= prev[both] + 1e-10), (
                f"full={full}: rate increased at s_rise={s}"
            )
        prev = R


# --------------------------------------------------------------------------- #
# R.3 - qE_r round-trip: peak flux at the raised-limit boundary equals η·F_lim #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("s_rise", [0.2, 0.6, 1.2])
def test_R03_qEr_roundtrip(model, s_rise):
    tv = 2.0 * DAY_S
    t_exp = model.t_exp_s(np.array([30.0]), np.array([tv]))
    F_lim = model.F_lim_Jy(t_exp)
    eta, _ = model._rise_eta(np.array([tv]), s_rise)
    qE_r = float(model.q_Euc(eta * F_lim)[0])
    F_target = float(eta[0] * F_lim[0])
    F_check = _F_p_at_Deuc(model, qE_r)
    assert math.isclose(F_check, F_target, rel_tol=1e-12), (
        f"s_rise={s_rise}: F_p(qE_r)={F_check:.6e} vs η·F_lim={F_target:.6e}"
    )


# --------------------------------------------------------------------------- #
# R.4 - q_ri(η=1) equals q_i exactly (D_i = D_max(q_i) identity)               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("t_cad_d", [0.3, 2.0, 15.0])
def test_R04_qri_eta1_equals_qi(model, t_cad_d):
    tv = t_cad_d * DAY_S
    i_det = 3
    t_exp = model.t_exp_s(np.array([30.0]), np.array([tv]))
    F_lim = model.F_lim_Jy(t_exp)
    D_i = model.D_i(i_det, np.array([tv]), F_lim)
    q_ri_1 = float(model.q_Euc(F_lim * (D_i / model.phys.D_euc_cm) ** 2)[0])
    q_i_v = float(model.q_i(i_det, np.array([tv]))[0])
    assert math.isclose(q_ri_1, q_i_v, rel_tol=1e-10), (
        f"t_cad={t_cad_d}d: q_ri(η=1)={q_ri_1} vs q_i={q_i_v}"
    )


# --------------------------------------------------------------------------- #
# R.5 - Closed-form V_w vs breakpoint-aware quadrature of the joint weight     #
# --------------------------------------------------------------------------- #


def _V_w_reference(model, qv, Dtm, Deff, Dmin, i_det, tv, s_fade, s_rise, s_mode):
    """Quadrature of 3D̃²·w with dense grids between the analytic breakpoints."""
    t_p, t_fs, k, ise = model._rise_fade_windows(
        np.array([qv]), i_det, np.array([tv]), s_fade, s_rise, s_mode)
    t_p, t_fs, k, ise = float(t_p[0]), float(t_fs[0]), float(k[0]), float(ise[0])
    D0 = Dtm * ise
    w_f = min(max(t_fs - t_p, 0.0), tv)
    Dc = D0 * (t_p / (t_p + w_f)) ** (1.0 / k) if w_f > 0 else D0
    edges = sorted({Dmin, Deff, min(max(Dc, Dmin), Deff), min(max(D0, Dmin), Deff)})
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        Dg = np.linspace(a, b, 8001)
        w = model._joint_survival(
            np.full_like(Dg, qv), Dg, np.full_like(Dg, Dtm),
            i_det, np.full_like(Dg, tv), s_fade, s_rise, s_mode)
        total += float(np.trapezoid(3.0 * Dg ** 2 * w, Dg))
    return total


@pytest.mark.parametrize("qv", [0.7, 1.02, 1.6, 6.0])       # on-axis, II, III
@pytest.mark.parametrize("s_fade,t_fs_kind", [(0.0, "inf"), (0.4, "finite"), (5.0, "tiny")])
@pytest.mark.parametrize("Dtm", [1e-3, 0.3, 2.0])           # D̃_0 below D_min / mid / above D_eff
def test_R05_weighted_volume_vs_quadrature(model, qv, s_fade, t_fs_kind, Dtm):
    i_det = 3
    tv = 1.5 * DAY_S
    s_rise = 0.5
    Deff = min(Dtm, 0.8)
    Dmin = 0.05
    V = float(model._weighted_D_volume(
        np.array([qv]), np.array([Dtm]), np.array([Deff]), Dmin,
        i_det, np.array([tv]), s_fade, s_rise, "discrete")[0])
    V_ref = _V_w_reference(model, qv, Dtm, Deff, Dmin, i_det, tv,
                           s_fade, s_rise, "discrete")
    assert V >= 0.0
    if V_ref > 1e-12:
        assert abs(V - V_ref) / V_ref < 5e-4, (
            f"q={qv}, s_fade={s_fade}, Dtm={Dtm}: V={V:.6e} vs quad={V_ref:.6e}"
        )
    else:
        assert V < 1e-10


# --------------------------------------------------------------------------- #
# R.6 - Weighted full rate ≤ best-case (u=0) hard-cut rate; dominant ≤ it too  #
# --------------------------------------------------------------------------- #


def test_R06_weight_below_hard_cut(model, deep_model, strategy_grid):
    """The uniform-start weight satisfies w ≤ 1{D̃ ≤ D̃_0(q)} pointwise, so the
    weighted full-integral rate must sit below the best-case (first detection
    at the peak) hard-cut rate built by quadrature from the same D̃ profile.
    The dominant-term rate drops the D-structured tail beyond the boundary
    angle, so it sits below the hard-cut reference as well.  (Note: full vs
    dominant themselves are NOT ordered under the rise cut — the tail the
    dominant drops can outweigh the uniform-start suppression.)"""
    N, T = strategy_grid
    i_det = 3
    checked = 0
    for m in (model, deep_model):
        masks = m.region_masks(i_det, N, T)
        for name in ("A1", "A2", "A5", "A6"):
            idx = np.argwhere(masks[name])
            if not len(idx):
                continue
            i, j = idx[len(idx) // 2]
            Nv, tv = float(N[i, j]), float(T[i, j])
            for s_rise in (0.3, 1.0):
                q_vals, D_eff_norm, D_tilde_max, prefactor, t_exp = (
                    m._D_eff_q_profile_scalar(i_det, Nv, tv, 800))
                if not math.isfinite(t_exp):
                    continue
                _, ise = m._rise_eta(np.array([tv]), s_rise)
                D_hard = np.minimum(D_eff_norm, D_tilde_max * float(ise[0]))
                R_hard = math.log10(prefactor * np.trapezoid(
                    q_vals * D_hard ** 3, q_vals))

                R_full = float(m.rate_log10_full_integral(
                    i_det, np.array([Nv]), np.array([tv]),
                    s_rise=s_rise, N_q=800)[0])
                R_dom = float(m.rate_log10(
                    i_det, np.array([Nv]), np.array([tv]), s_rise=s_rise)[0])
                if math.isfinite(R_full):
                    assert R_full <= R_hard + 5e-3, (
                        f"{name} s_rise={s_rise}: weighted {R_full:.4f} exceeds "
                        f"hard-cut reference {R_hard:.4f}"
                    )
                    checked += 1
                if math.isfinite(R_dom):
                    assert R_dom <= R_hard + 5e-3, (
                        f"{name} s_rise={s_rise}: dominant {R_dom:.4f} exceeds "
                        f"hard-cut reference {R_hard:.4f}"
                    )
    assert checked >= 4, "too few finite cells exercised"


# --------------------------------------------------------------------------- #
# R.7 - i_det = 1: the rise cut bites while the discrete fade cut bypasses     #
# --------------------------------------------------------------------------- #


def test_R07_idet1_rise_bites(model, strategy_grid):
    N, T = strategy_grid
    R_base = model.rate_log10(1, N, T)
    R_fade = model.rate_log10(1, N, T, s_fade=1.0, s_mode="discrete")
    R_rise = model.rate_log10(1, N, T, s_rise=1.0)

    np.testing.assert_array_equal(R_base, R_fade)   # fade bypassed at i=1
    both = np.isfinite(R_base) & np.isfinite(R_rise)
    assert np.all(R_rise[both] <= R_base[both] + 1e-12)
    assert np.any(R_rise[both] < R_base[both] - 1e-6), (
        "rise cut at i_det=1 should reduce the rate somewhere on the grid"
    )


# --------------------------------------------------------------------------- #
# R.8 - Joint fade+rise ≤ min(fade-only, rise-only) (comonotone windows)       #
# --------------------------------------------------------------------------- #


def test_R08_joint_below_each_single_cut(model, strategy_grid):
    N, T = strategy_grid
    i_det = 3
    kw = dict(s_mode="discrete", N_q=300)
    R_j = model.rate_log10_full_integral(i_det, N, T, s_fade=0.5, s_rise=0.5, **kw)
    R_f = model.rate_log10_full_integral(i_det, N, T, s_fade=0.5, **kw)
    R_r = model.rate_log10_full_integral(i_det, N, T, s_rise=0.5, **kw)
    both = np.isfinite(R_j) & np.isfinite(R_f) & np.isfinite(R_r)
    assert np.all(R_j[both] <= R_f[both] + 1e-10)
    assert np.all(R_j[both] <= R_r[both] + 1e-10)


# --------------------------------------------------------------------------- #
# R.9 - The rise cut culls distant sources: D_med decreases (full mode)        #
# --------------------------------------------------------------------------- #


def test_R09_Dmed_decreases(strategy_grid):
    """Pick the longest-cadence flux-limited cell (η grows exponentially with
    t_cad, and the D-median grid resolves shifts only above ~1%) and check the
    full-mode median distance strictly drops under the rise cut.

    Uses an explicitly pinned bright normalization (the pre-recalibration
    ε_B = 1e-2, D_Euc = 1.63e28 cm) rather than the engine defaults: whether
    the median shift resolves on the D-median grid depends on how deep the
    chosen flux-limited cell is, which is a property of this scenario, not of
    the default parameter set."""
    # Frozen-dataclass gotcha: R_int_yr is baked at class definition from the
    # default rho/D_euc — recompute explicitly for the custom D_euc.
    _D_euc = 1.63e28
    _rho = AfterglowPhysicalParams().rho_grb_gpc3_yr
    _R_int = (4.0 / 3.0) * math.pi * _rho * (_D_euc * CM_TO_GPC) ** 3
    model = DetectionRateModel(
        AfterglowPhysicalParams(D_euc_cm=_D_euc, R_int_yr=_R_int),
        SurveyInstrumentParams(),
        MicrophysicsParams(epsilon_B=1e-2),
    )
    N, T = strategy_grid
    i_det = 3
    masks = model.region_masks(i_det, N, T)
    for name in ("A2", "A3", "A1"):
        idx = np.argwhere(masks[name])
        if len(idx):
            k = int(np.argmax(T[idx[:, 0], idx[:, 1]]))
            i, j = idx[k]
            Nv, tv = float(N[i, j]), float(T[i, j])
            break
    else:
        pytest.skip("No flux-limited cell in strategy_grid")

    q0, D0 = model.compute_medians(
        i_det, np.array([Nv]), np.array([tv]), full_integral=True)
    q1, D1 = model.compute_medians(
        i_det, np.array([Nv]), np.array([tv]), full_integral=True, s_rise=1.0)
    if not (np.isfinite(D0[0]) and np.isfinite(D1[0])):
        pytest.skip("median NaN at chosen cell")
    assert D1[0] < D0[0], (
        f"{name} (t_cad={tv / DAY_S:.2f} d): D_med should drop under the rise "
        f"cut ({D1[0]:.4e} !< {D0[0]:.4e})"
    )


# --------------------------------------------------------------------------- #
# R.10 - t_dec floor regression: on-axis sources well inside the raised limit  #
#        keep w ≈ 1 (the t_p → 0 extrapolation would zero them)                #
# --------------------------------------------------------------------------- #


def test_R10_tdec_floor_onaxis_survival(model):
    tv = 1.0 * DAY_S
    s_rise = 0.3
    _, ise = model._rise_eta(np.array([tv]), s_rise)
    D_tilde_max = np.array([0.5])
    D0 = float(D_tilde_max[0] * ise[0])
    # Well inside the raised-limit horizon: t_lim = t_dec·(D0/D̃)^k ≫ t_cad.
    D_test = np.array([1e-3 * D0])
    w = model._joint_survival(
        np.array([0.8]), D_test, D_tilde_max,
        3, np.array([tv]), 0.0, s_rise, "discrete")
    assert w[0] > 0.99, (
        f"on-axis source far inside the raised limit should pass (w={w[0]:.4f}); "
        f"w ≈ 0 would indicate the t_p → 0 extrapolation regression"
    )
    # And w ≡ 0 beyond the zero-window distance D̃_0.
    w0 = model._joint_survival(
        np.array([0.8]), np.array([1.01 * D0]), D_tilde_max,
        3, np.array([tv]), 0.0, s_rise, "discrete")
    assert w0[0] == 0.0


# --------------------------------------------------------------------------- #
# R.11 - Deep cut: the dominant rate decays (no saturation plateau)            #
# --------------------------------------------------------------------------- #


def test_R11_deep_cut_no_plateau(model):
    """With the on-axis max-term + validity gating, the dominant rate follows
    the physical η^{−3/2} decay instead of plateauing when the raised-limit
    boundary angle collapses to q = 1."""
    tv = np.array([20.0 * DAY_S])
    Nv = np.array([30.0])
    vals = []
    for s in (0.5, 1.0, 1.5, 2.0):
        vals.append(float(model.rate_log10(3, Nv, tv, s_rise=s)[0]))
    diffs = np.diff(vals)
    assert np.all(diffs < 0.0), f"dominant deep-cut rate plateaued: {vals}"
    # η^{−3/2} scaling: Δlog10 R = −(3/2)·ΔE with ΔE = Δs·t_cad/(2.5·DAY_S).
    dE = 0.5 * 20.0 / 2.5
    expected = -1.5 * dE
    assert abs(diffs[-1] - expected) < 0.3, (
        f"deep-cut slope {diffs[-1]:.2f} vs expected η^(-3/2) scaling {expected:.2f}"
    )
