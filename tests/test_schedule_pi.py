"""Verification of grb_detect/schedule.py against brute-force references.

Covers (docs/implementation_reference.tex, Sec. "Two-Timescale Night Schedule"):
  1. P_i(T) (Eq. P_i) vs a dense-phase brute-force count over the periodic
     visit schedule, across (N_v, dt_v, i, t_cad) including i > N_v and i = 1.
  2. The channel decomposition (Eq. W_schedule) vs a brute force that tracks
     the first visit's channel and applies per-channel cut deadlines.
  3. Closed-form identities: N_v = 1 uniform ramp, span/endpoint identities,
     the dt_v -> 0 limit, the f_night recovery, the mixture bracket, and the
     ZTF-public worked example.

Run with::

    .venv/Scripts/python -m pytest tests/test_schedule_pi.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from grb_detect.schedule import NightSchedule

DAY_S = 86400.0
T_NIGHT_S = 36000.0  # 10 h


# --------------------------------------------------------------------------- #
# Brute-force references                                                      #
# --------------------------------------------------------------------------- #

def _visit_grid(T, t_cad, N_v, dt_v):
    """Visit times and within-night slot indices covering [0, T + 2 t_cad]."""
    n_per = int(np.ceil((T + t_cad) / t_cad)) + 2
    m = np.arange(n_per)[:, None]
    j = np.arange(N_v)[None, :]
    v = (m * t_cad + j * dt_v).ravel()
    slots = np.broadcast_to(j, (n_per, N_v)).ravel()
    order = np.argsort(v, kind="stable")
    return v[order], slots[order]


def brute_P(T, i, t_cad, N_v, dt_v, n_phase=200_000):
    """P_i(T) by counting visits in [u, u+T] over dense midpoint phases."""
    v, _ = _visit_grid(T, t_cad, N_v, dt_v)
    u = (np.arange(n_phase) + 0.5) * (t_cad / n_phase)
    n_in = np.searchsorted(v, u + T, side="right") - np.searchsorted(v, u, side="left")
    return float(np.mean(n_in >= i))


def brute_W(T, i, t_cad, N_v, dt_v, tau_G, tau_Adlt, tau_B, n_phase=200_000):
    """Joint weight by simulating the wait to the first visit and its channel."""
    v, slots = _visit_grid(T, t_cad, N_v, dt_v)
    u = (np.arange(n_phase) + 0.5) * (t_cad / n_phase)
    first = np.searchsorted(v, u, side="left")
    wait = v[first] - u
    j = slots[first]
    mu, rho = divmod(i - 1, N_v)
    S_A = mu * t_cad + rho * dt_v
    S_B = (mu + 1) * t_cad - (N_v - rho) * dt_v
    S = np.where(j <= N_v - 1 - rho, S_A, S_B)
    tau = np.where(j == 0, tau_G, np.where(j <= N_v - 1 - rho, tau_Adlt, tau_B))
    ok = (wait <= T - S) & (wait <= tau)
    return float(np.mean(ok))


def sched(N_v, dt_v, t_night=T_NIGHT_S):
    return NightSchedule(N_v=N_v, dt_v_s=dt_v, t_night_s=t_night)


# --------------------------------------------------------------------------- #
# 1. P_i(T) vs brute force                                                    #
# --------------------------------------------------------------------------- #

BRUTE_CASES = [
    # (N_v, i, dt_v [s], t_cad [s])
    (1, 2, 0.0, DAY_S),
    (1, 6, 0.0, 3 * DAY_S),
    (2, 2, 7200.0, 2 * DAY_S),      # ZTF public
    (2, 2, 1800.0, DAY_S),
    (2, 3, 7200.0, 2 * DAY_S),      # i > N_v, wraps one night
    (3, 7, 3600.0, DAY_S),          # mu = 2
    (6, 2, 5400.0, DAY_S),          # ZTF high-cadence
    (6, 6, 1800.0, 2 * DAY_S),
    (10, 1, 3600.0, DAY_S),         # i = 1
    (2, 2, 50000.0, 2 * DAY_S),     # dt_v > G would need dt_v > t_cad - dt_v; here G = 122800 s > dt_v
    (3, 4, 40000.0, 2 * DAY_S),     # dt_v > G = 92800? G = 172800 - 80000 = 92800 > dt_v; still large-dt case
]


@pytest.mark.parametrize("N_v,i,dt_v,t_cad", BRUTE_CASES)
def test_P_matches_brute_force(N_v, i, dt_v, t_cad):
    s = sched(N_v, dt_v)
    S_A, S_B, G, _mu, _rho = s.spans(i, t_cad)
    # Log-spaced windows plus every breakpoint and its neighborhood.
    T_vals = list(np.logspace(np.log10(600.0), np.log10(20 * DAY_S), 12))
    for bp in (float(S_A), float(S_B), float(S_A) + float(G),
               float(S_A) + dt_v, float(S_B) + dt_v):
        T_vals += [bp, max(bp - 977.0, 1.0), bp + 977.0]
    for T in T_vals:
        P = float(s.P_detect(np.asarray(T), i, np.asarray(t_cad)))
        Pb = brute_P(T, i, t_cad, N_v, dt_v)
        assert abs(P - Pb) < 1e-3, (
            f"P mismatch at T={T:.1f}: closed {P:.6f} vs brute {Pb:.6f}")


# --------------------------------------------------------------------------- #
# 2. Joint weight (channels + cut deadlines) vs brute force                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("N_v,i,dt_v,t_cad,taus", [
    (2, 2, 7200.0, 2 * DAY_S, (30000.0, 30000.0, 3000.0)),
    (2, 2, 7200.0, 2 * DAY_S, (1e12, 1e12, 1e12)),      # no cuts == P_i
    (6, 2, 5400.0, DAY_S, (20000.0, 2000.0, 2000.0)),
    (3, 7, 3600.0, DAY_S, (50000.0, 500.0, 40000.0)),
    (1, 3, 0.0, 2 * DAY_S, (100000.0, 0.0, 0.0)),
    (6, 4, 5400.0, 2 * DAY_S, (-5.0, 4000.0, 1e12)),    # tau <= 0 kills a channel
])
def test_joint_weight_matches_brute_force(N_v, i, dt_v, t_cad, taus):
    s = sched(N_v, dt_v)
    for T in (0.3 * DAY_S, 1.7 * DAY_S, 3.1 * DAY_S, 9.7 * DAY_S):
        W = float(s.joint_weight(
            i, np.asarray(t_cad), T=np.asarray(T),
            tau_by_channel=[np.asarray(t) for t in taus]))
        Wb = brute_W(T, i, t_cad, N_v, dt_v, *taus)
        assert abs(W - Wb) < 1e-3, (
            f"W mismatch at T={T:.0f}: closed {W:.6f} vs brute {Wb:.6f}")


def test_joint_weight_without_taus_is_P():
    s = sched(4, 4000.0)
    t_cad = np.asarray([DAY_S, 2 * DAY_S, 5 * DAY_S])
    T = np.asarray([0.4 * DAY_S, 2.2 * DAY_S, 3.3 * DAY_S])
    np.testing.assert_allclose(
        s.joint_weight(3, t_cad, T=T), s.P_detect(T, 3, t_cad), rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 3. Closed-form identities and limits                                        #
# --------------------------------------------------------------------------- #

def test_nv1_is_uniform_ramp():
    s = sched(1, 0.0)
    t_cad = np.asarray([DAY_S, 2 * DAY_S, 7 * DAY_S])
    for i in (1, 2, 5):
        T = np.linspace(0.0, 10 * DAY_S, 401)[:, None]
        P = s.P_detect(T, i, t_cad[None, :])
        expected = np.clip(T / t_cad[None, :] - (i - 1), 0.0, 1.0)
        np.testing.assert_allclose(P, expected, rtol=0, atol=1e-15)


def test_span_identities():
    for N_v, i, dt_v, t_cad in [(2, 2, 7200.0, 2 * DAY_S), (6, 4, 5400.0, DAY_S),
                                (3, 8, 3600.0, 2 * DAY_S)]:
        s = sched(N_v, dt_v)
        S_A, S_B, G, mu, rho = s.spans(i, np.asarray(t_cad))
        assert np.isclose(S_B - S_A, G - dt_v)          # S_B = S_A + G - dt_v
        if i <= N_v:
            assert np.isclose(S_A, (i - 1) * dt_v)      # intra-night run
        assert mu == (i - 1) // N_v and rho == (i - 1) % N_v
    s1 = sched(1, 0.0)
    S_A, _, G, _, _ = s1.spans(4, np.asarray(3 * DAY_S))
    assert np.isclose(S_A, 3 * (3 * DAY_S)) and np.isclose(G, 3 * DAY_S)
    assert np.isclose(s1.T_certain(4, np.asarray(3 * DAY_S)), 4 * (3 * DAY_S))


def test_endpoints():
    for N_v, i, dt_v, t_cad in [(2, 2, 7200.0, 2 * DAY_S), (6, 2, 5400.0, DAY_S),
                                (3, 7, 3600.0, DAY_S), (1, 3, 0.0, 2 * DAY_S)]:
        s = sched(N_v, dt_v)
        t = np.asarray(t_cad)
        T0 = float(s.T_first_pass(i, t))
        T1 = float(s.T_certain(i, t))
        assert float(s.P_detect(np.asarray(T0), i, t)) == 0.0
        assert float(s.P_detect(np.asarray(T0 + 60.0), i, t)) > 0.0
        assert float(s.P_detect(np.asarray(T1), i, t)) == pytest.approx(1.0, abs=1e-12)
        assert float(s.P_detect(np.asarray(T1 - 60.0), i, t)) < 1.0


def test_dt_zero_limit():
    """dt_v -> 0: N_v simultaneous visits, only ceil(i/N_v)-night accumulation."""
    t_cad = np.asarray(2 * DAY_S)
    T = np.linspace(0.0, 12 * DAY_S, 301)
    for N_v, i in [(2, 3), (3, 7), (6, 6)]:
        s = sched(N_v, 0.0)
        mu = (i - 1) // N_v
        np.testing.assert_allclose(
            s.P_detect(T, i, t_cad),
            np.clip(T / t_cad - mu, 0.0, 1.0), rtol=0, atol=1e-15)


def test_f_night_recovery():
    """Degenerate nightly schedule reproduces the legacy f_night factor."""
    N_v, i = 10, 2
    dt_v = T_NIGHT_S / N_v
    s = sched(N_v, dt_v)
    t_cad = np.asarray(DAY_S)
    S_A = (i - 1) * dt_v
    P = float(s.P_detect(np.asarray(S_A + dt_v), i, t_cad))
    assert P == pytest.approx((N_v - i + 1) * dt_v / DAY_S, rel=1e-12)
    f_night = T_NIGHT_S / DAY_S
    assert abs(P - f_night) / f_night < 0.15  # ~= f_night for i << N_v


def test_monotonicity():
    s = sched(4, 5000.0)
    t_cad = np.asarray(2 * DAY_S)
    T = np.linspace(0.0, 15 * DAY_S, 500)
    P2 = s.P_detect(T, 2, t_cad)
    assert np.all(np.diff(P2) >= -1e-15)                 # nondecreasing in T
    for i in (3, 5, 9):
        assert np.all(s.P_detect(T, i, t_cad) <= P2 + 1e-15)  # nonincreasing in i


def test_mixture_bracket():
    """sum w_c 1{T >= S_c+g_c}  <=  P_i(T)  <=  sum w_c 1{T >= S_c}."""
    for N_v, i, dt_v, t_cad in [(2, 2, 7200.0, 2 * DAY_S), (6, 2, 5400.0, DAY_S),
                                (3, 7, 3600.0, DAY_S)]:
        s = sched(N_v, dt_v)
        t = np.asarray(t_cad)
        T = np.linspace(0.0, 12 * DAY_S, 700)
        lo = np.zeros_like(T)
        hi = np.zeros_like(T)
        for m_c, g_c, S_c in s.channels(i, t):
            if m_c == 0:
                continue
            w_c = m_c * float(g_c) / t_cad
            lo += w_c * (T >= float(S_c) + float(g_c))
            hi += w_c * (T >= float(S_c))
        P = s.P_detect(T, i, t)
        assert np.all(lo <= P + 1e-12) and np.all(P <= hi + 1e-12)


def test_channels_partition():
    for N_v, i, dt_v in [(1, 2, 0.0), (2, 2, 7200.0), (6, 4, 5400.0), (3, 8, 3600.0)]:
        s = sched(N_v, dt_v)
        t_cad = np.asarray([DAY_S, 2 * DAY_S])
        total = sum(m_c * np.asarray(g_c, dtype=float)
                    for m_c, g_c, _S in s.channels(i, t_cad))
        np.testing.assert_allclose(total, t_cad, rtol=1e-15)


def test_ztf_public_worked_example():
    """i=2, N_v=2, t_cad=2 d: P = [min(G,[T-dt]+) + min(dt,[T-(2d-dt)]+)]/2d."""
    dt_v = 7200.0
    t_cad = 2 * DAY_S
    s = sched(2, dt_v)
    G = t_cad - dt_v
    for T in (0.0, dt_v, 0.5 * DAY_S, 1.5 * DAY_S, t_cad - dt_v + 3600.0, 3 * DAY_S):
        expected = (min(G, max(T - dt_v, 0.0))
                    + min(dt_v, max(T - (t_cad - dt_v), 0.0))) / t_cad
        assert float(s.P_detect(np.asarray(T), 2, np.asarray(t_cad))) == (
            pytest.approx(expected, rel=1e-14))
    # An event visible for half a period is caught with P ~ T/t_cad.
    P_half = float(s.P_detect(np.asarray(DAY_S), 2, np.asarray(t_cad)))
    assert P_half == pytest.approx((DAY_S - dt_v) / t_cad, rel=1e-14)


def test_broadcasting():
    s = sched(3, 3600.0)
    t_cad = np.asarray([DAY_S, 2 * DAY_S, 4 * DAY_S])
    T = np.linspace(0.1 * DAY_S, 9 * DAY_S, 7)[:, None]
    P = s.P_detect(T, 4, t_cad[None, :])
    assert P.shape == (7, 3)
    assert np.all((P >= 0.0) & (P <= 1.0))


def test_feasible():
    s = sched(6, 5400.0)  # run = 5*5400 = 27000 <= t_night = 36000
    t_exp = np.asarray([30.0, 5400.0, np.nan])
    mask = s.feasible(t_exp, t_oh_s=15.0)
    assert mask.tolist() == [True, False, False]  # overlap: 5400 < 5400+15
    s_bad = sched(8, 5400.0)  # run 37800 > t_night
    assert not s_bad.feasible(np.asarray(30.0), t_oh_s=0.0)
    s_one = sched(1, 0.0)
    assert bool(s_one.feasible(np.asarray(np.nan)))  # N_v = 1 always feasible


def test_input_validation():
    with pytest.raises(ValueError):
        NightSchedule(N_v=0, dt_v_s=0.0, t_night_s=T_NIGHT_S)
    with pytest.raises(ValueError):
        NightSchedule(N_v=2, dt_v_s=-1.0, t_night_s=T_NIGHT_S)
    with pytest.raises(ValueError):
        sched(2, 3600.0).spans(0, np.asarray(DAY_S))
