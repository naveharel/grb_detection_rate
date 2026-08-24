"""Two-timescale night-schedule math.

Implements the schedule combinatorics of docs/implementation_reference.tex,
Sec. "Two-Timescale Night Schedule": the i-visit spans S_A/S_B (Eq. spans),
the exact detection probability P_i(T) (Eq. P_i), the per-channel joint
identification weight (Eq. W_schedule), and the schedule feasibility masks.

Pure schedule geometry only — no coupling to the afterglow model.
`DetectionRateModel` consumes these primitives; `tests/test_schedule_pi.py`
verifies them against a brute-force dense-phase reference.

Notation (mirrors the tex):
    t_cad   schedule period = inter-night revisit interval [s]
    N_v     visits per visit-night
    dt_v    intra-night visit spacing [s]
    G       overnight gap  = t_cad - (N_v - 1) * dt_v
    mu, rho = divmod(i - 1, N_v)
    S_A     = mu * t_cad + rho * dt_v          (intra-anchored i-visit span)
    S_B     = (mu + 1) * t_cad - (N_v - rho) * dt_v   (span wrapping one extra night)

Channels (groups of identical starting gaps, ordered [A_G, A_dlt, B]):
    A_G:   multiplicity 1,             gap G,    span S_A
    A_dlt: multiplicity N_v - 1 - rho, gap dt_v, span S_A
    B:     multiplicity rho,           gap dt_v, span S_B
with phase weights w_c = m_c * g_c / t_cad summing to 1 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NightSchedule:
    """Periodic visit schedule: N_v visits at spacing dt_v_s inside each
    visit-night; visit-nights repeat with the runtime period t_cad_s.

    Parameters
    ----------
    N_v:
        Visits per visit-night (>= 1). N_v = 1 is the legacy one-visit
        schedule (dt_v_s is then inert).
    dt_v_s:
        Intra-night visit spacing [s] (>= 0).
    t_night_s:
        Night-window length [s] (> 0); used only by `feasible()`.
    """

    N_v: int
    dt_v_s: float
    t_night_s: float

    def __post_init__(self):
        if int(self.N_v) < 1:
            raise ValueError("N_v must be >= 1")
        if not (self.dt_v_s >= 0.0):
            raise ValueError("dt_v_s must be >= 0")
        if not (self.t_night_s > 0.0):
            raise ValueError("t_night_s must be > 0")

    # ---------- Spans and channels ----------
    def spans(self, i_det: int, t_cad_s: np.ndarray):
        """Return (S_A, S_B, G, mu, rho) — Eq. (spans) of the tex.

        S_A, S_B, G broadcast against ``t_cad_s``; mu, rho are ints.
        """
        i_det = int(i_det)
        if i_det < 1:
            raise ValueError("i_det must be >= 1")
        t_cad = np.asarray(t_cad_s, dtype=float)
        N_v = int(self.N_v)
        dt_v = float(self.dt_v_s)
        mu, rho = divmod(i_det - 1, N_v)
        G = t_cad - (N_v - 1) * dt_v
        S_A = mu * t_cad + rho * dt_v
        S_B = (mu + 1) * t_cad - (N_v - rho) * dt_v
        # A visit run longer than the period (G < 0) is geometrically
        # impossible; poison those cells with NaN so every downstream quantity
        # (P_i, joint weights, mixture rates) reads invalid instead of
        # silently overflowing (P_i > 1, negative channel weights).  The UI
        # cannot reach this (integer-day cadences + the fits-night mask keep
        # G >= 0); direct API calls with sub-period t_cad are the target.
        if np.any(G < 0.0):
            bad = G < 0.0
            G = np.where(bad, np.nan, G)
            S_A = np.where(bad, np.nan, S_A)
            S_B = np.where(bad, np.nan, S_B)
        return S_A, S_B, G, mu, rho

    def channels(self, i_det: int, t_cad_s: np.ndarray):
        """The <= 3 phase channels as (multiplicity, gap, span) triples.

        Ordered [A_G, A_dlt, B]; zero-multiplicity channels are included so
        callers can rely on the fixed order (skip m_c == 0 entries).
        Identity: sum_c m_c * g_c = t_cad exactly.
        """
        S_A, S_B, G, _mu, rho = self.spans(i_det, t_cad_s)
        dt_v = np.broadcast_to(float(self.dt_v_s), np.shape(S_A))
        return [
            (1, G, S_A),
            (int(self.N_v) - 1 - rho, dt_v, S_A),
            (rho, dt_v, S_B),
        ]

    # ---------- Detection probability and joint weight ----------
    def joint_weight(
        self,
        i_det: int,
        t_cad_s: np.ndarray,
        *,
        T: np.ndarray | None = None,
        tau_by_channel=None,
    ) -> np.ndarray:
        """Eq. (W_schedule): (1/t_cad) * sum_c m_c * [min(g_c, T - S_c, tau_c)]_+ .

        Parameters
        ----------
        T:
            Detectable-window length [s], broadcastable against ``t_cad_s``.
            ``None`` drops the detection-ramp term (the window is then a hard
            per-channel rectangle handled by the caller; only the cut
            deadlines ride the phase ramp).
        tau_by_channel:
            Sequence of 3 cut deadlines tau_c [s] (arrays broadcastable
            against ``t_cad_s``, or ``None`` for no cut on that channel),
            ordered like `channels()`. ``None`` means no cuts at all.

        With ``tau_by_channel=None`` and ``T`` given this evaluates the exact
        detection probability P_i(T), Eq. (P_i).
        """
        t_cad = np.asarray(t_cad_s, dtype=float)
        if tau_by_channel is None:
            tau_by_channel = (None, None, None)
        acc = None
        for (m_c, g_c, S_c), tau_c in zip(
            self.channels(i_det, t_cad), tau_by_channel
        ):
            if m_c == 0:
                continue
            lim = np.asarray(g_c, dtype=float)
            if T is not None:
                lim = np.minimum(lim, np.asarray(T, dtype=float) - S_c)
            if tau_c is not None:
                lim = np.minimum(lim, np.asarray(tau_c, dtype=float))
            term = m_c * np.maximum(lim, 0.0)
            acc = term if acc is None else acc + term
        return acc / t_cad

    def P_detect(
        self, T: np.ndarray, i_det: int, t_cad_s: np.ndarray
    ) -> np.ndarray:
        """Exact detection probability P_i(T), Eq. (P_i) of the tex."""
        return self.joint_weight(i_det, t_cad_s, T=T)

    # ---------- Hard-boundary window conventions ----------
    def T_first_pass(self, i_det: int, t_cad_s: np.ndarray) -> np.ndarray:
        """Shortest window with P_i > 0: min_c S_c (= S_A unless dt_v > G)."""
        S_A, S_B, _G, _mu, rho = self.spans(i_det, t_cad_s)
        return np.minimum(S_A, S_B) if rho > 0 else S_A

    def T_certain(self, i_det: int, t_cad_s: np.ndarray) -> np.ndarray:
        """Shortest window with P_i = 1: T_100 = max_c (S_c + g_c), nonempty c.

        A_G gives S_A + G and channel B's S_B + dt_v = S_A + G identically, so
        T_100 = S_A + max(G, dt_v) in general — but S_A + G when the A_dlt
        channel is empty (rho = N_v − 1, where the dt_v term has no starting
        visit; this includes every N_v = 1 case, giving i·t_cad regardless of
        the inert dt_v).
        """
        S_A, _S_B, G, _mu, rho = self.spans(i_det, t_cad_s)
        if rho < int(self.N_v) - 1:
            return S_A + np.maximum(G, float(self.dt_v_s))
        return S_A + G

    # ---------- Feasibility ----------
    def feasible(
        self,
        t_exp_s: np.ndarray,
        *,
        t_oh_s: float = 0.0,
    ) -> np.ndarray:
        """Schedule feasibility mask (tex Sec. sched_feasibility).

        (N_v - 1) * dt_v <= t_night   (the visit run fits the night)
        dt_v >= t_exp + t_OH          (no overlapping revisit)

        N_v = 1 is always schedule-feasible; ``t_exp > 0`` is the caller's
        (region-A0) responsibility. NaN t_exp yields False.
        """
        t_exp = np.asarray(t_exp_s, dtype=float)
        if int(self.N_v) == 1:
            return np.ones(np.shape(t_exp), dtype=bool)
        dt_v = float(self.dt_v_s)
        fits_night = (int(self.N_v) - 1) * dt_v <= float(self.t_night_s)
        with np.errstate(invalid="ignore"):
            no_overlap = dt_v >= t_exp + float(t_oh_s)
        return np.broadcast_to(fits_night, np.shape(t_exp)) & no_overlap
