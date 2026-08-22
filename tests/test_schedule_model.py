"""Verification of the night-schedule physics in DetectionRateModel (N_v >= 2).

Covers (docs/implementation_reference.tex, Sec. "Two-Timescale Night Schedule"):
  1. N_v = 1 schedule ≡ legacy model, bit-parity (budget, rates, medians).
  2. The pointwise schedule weight `_joint_survival` vs the independently
     brute-force-verified NightSchedule primitives (P_i identity; per-channel
     tau plumbing vs `NightSchedule.joint_weight`).
  3. The closed-form `_weighted_D_volume` schedule branch vs dense numerical
     quadrature of the pointwise weight — the load-bearing test for
     `_ramp_volume` (all cut / window / random-start combinations).
  4. End-to-end exact rate at N_v >= 2 vs a direct P_i(T)-weighted double
     integral built from schedule.P_detect (no cuts, win_from_peak).
  5. Dominant mixture: convention ordering, the ZTF-public conservative
     identity (both channels collapse to T_req = t_cad), medians routing.
  6. Physics regressions: the intra-night pair extends the ZTF-public
     median distance; the overnight rise gap restores the rise cut's teeth.

Run with::

    .venv/Scripts/python -m pytest tests/test_schedule_model.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from grb_detect.constants import DAY_S
from grb_detect.core import make_rate_model

T_NIGHT_S = 36000.0  # 10 h


def sched_model(N_v, dt_v_h, *, f_live=0.2, t_oh=0.0, win_iminus1=False,
                win_tp=False, on=True):
    return make_rate_model(
        A_log=-4.68, f_live=f_live, t_overhead_s=t_oh, omega_exp_deg2=47.0,
        schedule_on=on, N_v=N_v, dt_v_s=dt_v_h * 3600.0, t_night_s=T_NIGHT_S,
        win_i_minus_one=win_iminus1, win_from_peak=win_tp,
    )


# Strategy points on the discrete band (N_exp, t_cad in days).
POINTS = [(319.0, 2), (53.0, 1), (30.0, 3), (500.0, 1)]


def _NT(point):
    N, days = point
    return np.array([N]), np.array([days * DAY_S])


# --------------------------------------------------------------------------- #
# 1. N_v = 1 parity                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("full", [False, True], ids=["dominant", "exact"])
@pytest.mark.parametrize("kw", [
    dict(),
    dict(win_iminus1=True),
    dict(win_tp=True),
], ids=["legacy", "iminus1", "win-tp"])
def test_nv1_schedule_is_legacy(full, kw):
    m_sched = sched_model(1, 2.0, **kw)
    m_leg = sched_model(1, 2.0, on=False, **kw)
    cuts = dict(q_min=1.03, D_min_cm=0.0, s_fade=0.3, s_rise=0.5)
    for point in POINTS:
        N, t = _NT(point)
        fn = "rate_log10_full_integral" if full else "rate_log10"
        Z_s = float(getattr(m_sched, fn)(2, N, t, **cuts)[0])
        Z_l = float(getattr(m_leg, fn)(2, N, t, **cuts)[0])
        assert (Z_s == Z_l) or (math.isnan(Z_s) and math.isnan(Z_l)), (
            f"N_v=1 parity broken at {point}: {Z_s} vs {Z_l}")
        te_s = float(m_sched.t_exp_s(N, t)[0])
        te_l = float(m_leg.t_exp_s(N, t)[0])
        assert te_s == te_l


def test_budget_divides_by_Nv():
    m1 = sched_model(1, 1.5)
    m6 = sched_model(6, 1.5)
    N, t = np.array([100.0]), np.array([2 * DAY_S])
    assert float(m6.t_exp_s(N, t)[0]) == pytest.approx(
        float(m1.t_exp_s(N, t)[0]) / 6.0, rel=1e-12)


def test_infeasible_schedule_is_nan_everywhere():
    m = sched_model(8, 1.6)  # (8-1)*1.6 h = 11.2 h > t_night = 10 h
    N, t = np.array([100.0]), np.array([2 * DAY_S])
    assert not np.isfinite(m.t_exp_s(N, t)[0])
    assert not np.isfinite(m.rate_log10(2, N, t)[0])
    assert not np.isfinite(m.rate_log10_full_integral(2, N, t)[0])


# --------------------------------------------------------------------------- #
# 2. Pointwise schedule weight vs the schedule-module primitives              #
# --------------------------------------------------------------------------- #

def _profile(model, i_det, point, N_q=200):
    N, days = point
    (q_vals, D_eff_norm, D_max, pref, t_exp, Ddec) = (
        model._D_eff_q_profile_scalar(i_det, N, days * DAY_S, N_q))
    assert np.isfinite(t_exp)
    return q_vals, D_eff_norm, D_max, pref, Ddec


def test_pointwise_weight_is_Pi_without_cuts():
    """win_from_peak, no cuts: W(q, D̃) must equal P_i(T) with
    T = t_p,eff·[(D̃_max/D̃)^k − 1] — the exact detection probability."""
    i_det, point = 2, (319.0, 2)
    m = sched_model(2, 2.0, win_tp=True)
    q_vals, D_eff_norm, D_max, _pref, Ddec = _profile(m, i_det, point)
    t_arr = np.array([point[1] * DAY_S])

    d = np.linspace(1e-4, 1.0, 101)[:, None] * D_eff_norm[None, :]
    W = m._joint_survival(
        q_vals[None, :], d, D_max[None, :], i_det, t_arr,
        0.0, 0.0, "discrete", D_tilde_dec=Ddec)

    t_p = m._t_p_eff(q_vals)
    q_j = float(m.derived.q_j)
    p = m.phys.p
    k = np.where(q_vals < q_j,
                 2.0 / abs(float(m.pls.alpha_II_temporal(p))),
                 2.0 / abs(float(m.pls.alpha_III_temporal(p))))
    with np.errstate(over="ignore", divide="ignore"):
        T = t_p[None, :] * ((D_max[None, :] / np.maximum(d, 1e-300))
                            ** k[None, :] - 1.0)
    P = m.schedule.P_detect(np.minimum(T, 1e18), i_det, t_arr)
    np.testing.assert_allclose(W, P, rtol=1e-10, atol=1e-12)


def test_pointwise_weight_matches_schedule_joint_weight():
    """Channel/tau plumbing: `_joint_survival` must equal
    NightSchedule.joint_weight fed the same per-channel deadlines."""
    i_det, point = 3, (100.0, 2)
    m = sched_model(4, 1.5, win_tp=True)
    s_fade, s_rise, s_mode = 0.4, 0.6, "discrete"
    q_vals, D_eff_norm, D_max, _pref, Ddec = _profile(m, i_det, point)
    t_arr = np.array([point[1] * DAY_S])

    d = 0.35 * D_eff_norm
    W = m._joint_survival(q_vals, d, D_max, i_det, t_arr,
                          s_fade, s_rise, s_mode, D_tilde_dec=Ddec)

    t_p = m._t_p_eff(q_vals)
    q_j = float(m.derived.q_j)
    p = m.phys.p
    k = np.where(q_vals < q_j,
                 2.0 / abs(float(m.pls.alpha_II_temporal(p))),
                 2.0 / abs(float(m.pls.alpha_III_temporal(p))))
    T = t_p * ((D_max / np.maximum(d, 1e-300)) ** k - 1.0)

    taus = []
    for m_c, g_c, S_c in m.schedule.channels(i_det, t_arr):
        tII, tIII = m._t_first_caps(i_det, t_arr, s_fade, s_mode,
                                    baseline_s=S_c)
        t_fs = np.where(q_vals < q_j, tII, tIII)
        _, ise = m._rise_eta(t_arr, s_rise, gap_s=g_c)
        with np.errstate(over="ignore", divide="ignore"):
            t_lim = t_p * (D_max * ise / np.maximum(d, 1e-300)) ** k
        taus.append(np.minimum(t_fs, t_lim) - t_p)
    W_ref = m.schedule.joint_weight(i_det, t_arr, T=T, tau_by_channel=taus)
    np.testing.assert_allclose(W, W_ref, rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------- #
# 3. Closed-form weighted volume vs dense quadrature of the pointwise weight  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("win_tp", [False, True], ids=["rect", "from-peak"])
@pytest.mark.parametrize("cuts", [
    dict(s_fade=0.0, s_rise=0.0),
    dict(s_fade=0.5, s_rise=0.0),
    dict(s_fade=0.0, s_rise=0.7),
    dict(s_fade=0.3, s_rise=0.5),
    dict(s_fade=0.3, s_rise=0.5, fade_random_start=False),
    dict(s_fade=0.3, s_rise=0.5, rise_random_start=False),
], ids=["none", "fade", "rise", "both", "hard-fade", "hard-rise"])
@pytest.mark.parametrize("N_v,dt_v_h,days,i_det", [
    (2, 2.0, 2, 2),      # ZTF public
    (6, 1.5, 1, 2),      # ZTF high-cadence
    (3, 3.0, 2, 7),      # i > N_v, wraps nights
])
def test_weighted_volume_matches_quadrature(win_tp, cuts, N_v, dt_v_h, days, i_det):
    m = sched_model(N_v, dt_v_h, win_tp=win_tp)
    point = (150.0, days)
    q_vals, D_eff_norm, D_max, _pref, Ddec = _profile(m, i_det, point, N_q=60)
    t_arr = np.array([days * DAY_S])
    D_min_norm = 0.05
    flags = dict(rise_random_start=cuts.get("rise_random_start", True),
                 fade_random_start=cuts.get("fade_random_start", True))
    s_fade, s_rise = cuts["s_fade"], cuts["s_rise"]

    V = m._weighted_D_volume(
        q_vals, D_max, D_eff_norm, D_min_norm, i_det, t_arr,
        s_fade, s_rise, "discrete", D_tilde_dec=Ddec, **flags)

    # Dense trapezoid of 3D̃²·W over [D_min, D_eff_glob(q)] per q.
    n_dense = 4000
    frac = np.linspace(0.0, 1.0, n_dense)[:, None]
    D_grid = D_min_norm + frac * np.maximum(D_eff_norm - D_min_norm, 0.0)[None, :]
    W = m._joint_survival(
        q_vals[None, :], D_grid, D_max[None, :], i_det, t_arr,
        s_fade, s_rise, "discrete", D_tilde_dec=Ddec, **flags)
    V_ref = np.trapezoid(3.0 * D_grid ** 2 * W, D_grid, axis=0)

    scale = max(float(np.nanmax(V_ref)), 1e-300)
    np.testing.assert_allclose(
        V, V_ref, rtol=5e-3, atol=5e-3 * scale,
        err_msg=f"closed-form V_w deviates from quadrature "
                f"(N_v={N_v}, i={i_det}, win_tp={win_tp}, cuts={cuts})")


# --------------------------------------------------------------------------- #
# 4. End-to-end exact rate vs direct P_i(T) double integral                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("N_v,dt_v_h,days,i_det", [
    (2, 2.0, 2, 2),
    (6, 1.5, 1, 2),
])
def test_exact_rate_matches_Pi_reference(N_v, dt_v_h, days, i_det):
    m = sched_model(N_v, dt_v_h, win_tp=True)
    N, t_cad = 150.0, days * DAY_S
    N_q = 400
    (q_vals, D_eff_norm, D_max, pref, t_exp, _Ddec) = (
        m._D_eff_q_profile_scalar(i_det, N, t_cad, N_q))
    assert np.isfinite(t_exp)

    t_p = m._t_p_eff(q_vals)
    q_j = float(m.derived.q_j)
    p = m.phys.p
    k = np.where(q_vals < q_j,
                 2.0 / abs(float(m.pls.alpha_II_temporal(p))),
                 2.0 / abs(float(m.pls.alpha_III_temporal(p))))
    d = np.linspace(0.0, 1.0, 2000)[:, None] * D_eff_norm[None, :]
    with np.errstate(over="ignore", divide="ignore"):
        T = t_p[None, :] * ((D_max[None, :] / np.maximum(d, 1e-300))
                            ** k[None, :] - 1.0)
    P = m.schedule.P_detect(np.minimum(T, 1e18), i_det,
                            np.array([t_cad]))
    V = np.trapezoid(3.0 * d ** 2 * P, d, axis=0)
    R_ref = pref * np.trapezoid(q_vals * V, q_vals)

    Z = float(m.rate_log10_full_integral(
        i_det, np.array([N]), np.array([t_cad]), N_q=N_q)[0])
    assert 10.0 ** Z == pytest.approx(R_ref, rel=2e-2)


# --------------------------------------------------------------------------- #
# 5. Dominant mixture                                                         #
# --------------------------------------------------------------------------- #

def test_mixture_convention_ordering():
    """T_req,c = S_c (optimistic) gives at least the rate of S_c + g_c
    (conservative) — pointwise, dominant mode and exact-rect mode."""
    for full in (False, True):
        m_opt = sched_model(2, 2.0, win_iminus1=True)
        m_con = sched_model(2, 2.0, win_iminus1=False)
        for point in POINTS:
            N, t = _NT(point)
            fn = "rate_log10_full_integral" if full else "rate_log10"
            Z_o = float(getattr(m_opt, fn)(2, N, t)[0])
            Z_c = float(getattr(m_con, fn)(2, N, t)[0])
            if math.isfinite(Z_o) and math.isfinite(Z_c):
                assert Z_o >= Z_c - 1e-12


def test_ztf_public_conservative_identity():
    """i=2, N_v=2: both channels' conservative durations equal t_cad, so the
    mixture must equal the legacy rectangle at T_req = t_cad — i.e. a
    schedule-free model with win_i_minus_one and the same t_cad (its
    T_req = (i−1)·t_cad = t_cad), cuts off."""
    m_mix = sched_model(2, 2.0)                       # conservative convention
    m_ref = sched_model(1, 0.0, on=False, win_iminus1=True)
    N = np.array([319.0])
    t = np.array([2 * DAY_S])
    # Budget must match too: the reference needs the same t_exp, so evaluate
    # the mixture model's channels against a reference with half the f_live
    # (N_v = 2 halves the budget).
    m_ref_half = make_rate_model(
        A_log=-4.68, f_live=0.1, t_overhead_s=0.0, omega_exp_deg2=47.0,
        win_i_minus_one=True)
    assert float(m_mix.t_exp_s(N, t)[0]) == pytest.approx(
        float(m_ref_half.t_exp_s(N, t)[0]), rel=1e-12)
    Z_mix = float(m_mix.rate_log10(2, N, t)[0])
    Z_ref = float(m_ref_half.rate_log10(2, N, t)[0])
    assert Z_mix == pytest.approx(Z_ref, rel=1e-12)
    del m_ref


def test_medians_route_numerical_at_Nv2():
    m = sched_model(2, 2.0)
    N, t = np.array([319.0]), np.array([2 * DAY_S])
    q_a, D_a = m.compute_medians(2, N, t, full_integral=False)
    q_n, D_n = m.compute_medians_numerical(2, N, t)
    assert float(q_a[0]) == float(q_n[0])
    assert float(D_a[0]) == float(D_n[0])


# --------------------------------------------------------------------------- #
# 6. Physics regressions                                                      #
# --------------------------------------------------------------------------- #

def test_intranight_pair_extends_median_distance():
    """ZTF-public-like: modelling the 2 visits/night as a schedule must move
    the median distance well beyond the one-visit-per-2-nights encoding
    (the motivating result of the whole extension)."""
    i_det = 2
    N, t = np.array([319.0]), np.array([2 * DAY_S])
    m_new = sched_model(2, 2.0, win_tp=True)
    m_old = sched_model(1, 0.0, win_tp=True)   # legacy encoding, same budget/2?
    # Compare at matched depth: give the legacy model the schedule model's
    # halved budget so only the detection geometry differs.
    m_old_matched = make_rate_model(
        A_log=-4.68, f_live=0.1, t_overhead_s=0.0, omega_exp_deg2=47.0,
        win_from_peak=True)
    _, D_new = m_new.compute_medians(i_det, N, t, full_integral=True)
    _, D_old = m_old_matched.compute_medians(i_det, N, t, full_integral=True)
    assert float(D_new[0]) > 1.3 * float(D_old[0]), (
        f"intra-night pair did not extend the horizon: "
        f"{float(D_new[0]):.3g} vs {float(D_old[0]):.3g} cm")
    del m_old


def test_overnight_gap_restores_rise_cut():
    """ZTF-HC-like (N_v=6 nightly): the rise cut must bite through the
    overnight gap G (the legacy sub-day encoding had η → 1, toothless)."""
    m = sched_model(6, 1.5, win_tp=True)
    N, t = np.array([53.0]), np.array([DAY_S])
    Z0 = float(m.rate_log10_full_integral(2, N, t, s_rise=0.0)[0])
    Z1 = float(m.rate_log10_full_integral(2, N, t, s_rise=1.0)[0])
    assert math.isfinite(Z0) and math.isfinite(Z1)
    assert 10.0 ** (Z1 - Z0) < 0.9, (
        f"rise cut nearly toothless at N_v=6: ratio {10.0 ** (Z1 - Z0):.3f}")
