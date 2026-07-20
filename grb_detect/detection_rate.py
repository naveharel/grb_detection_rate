"""Analytic detection-rate surface in Python.

This module implements the piecewise rate model used for survey-strategy
optimisation. The implementation is vectorised and organised so that future
generalizations (wind-like media, additional PLSs, finite-exposure averaging,
etc.) can be introduced with minimal disruption.

Current scope:
- ISM external medium (constant density)
- Euclidean rate treatment with distance scale D_euc and intrinsic rate R_int
- PLS G off-axis scalings via the PLSModel interface (default: PLSG)
- Limiting flux model F_lim ∝ t_exp^{-alpha}

Conventions:
- i_det: the required number of detections within a cadence cycle
- N_exp and t_cad_s correspond to the strategy degrees of freedom

Outputs are returned as log10 R_det [yr^-1] by default to match common plotting
conventions for large dynamic range surfaces.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .afterglow_ism import q_dec as q_dec_fn
from .afterglow_ism import q_j as q_j_fn
from .afterglow_ism import q_nr as q_nr_fn
from .afterglow_ism import t_dec_s as t_dec_s_fn
from .afterglow_ism import t_j_s as t_j_s_fn
from .constants import DAY_S
from .params import AfterglowPhysicalParams, MicrophysicsParams, SurveyInstrumentParams, SurveyStrategy
from .pls import PLSG, PLSModel
from .survey import N_exp_max, exposure_time_s, is_strategy_physical, limiting_flux_Jy, sky_fraction

# Tolerance used to make boundary cases robust (e.g. N_exp = N_exp_max exactly)
_REGION_BOUNDARY_EPS: float = 1e-12

# Memory budget for the full-integral q-chunks: max float64 elements per
# (n_chunk, *grid) intermediate (2e6 el ≈ 16 MiB; ~15 concurrent slabs in the
# s_rise > 0 branch of _weighted_D_volume → ~250 MiB peak, safe inside
# Pyodide's WASM32 ~4 GiB address space independent of the strategy-grid
# size).  Read at call time so tests can monkeypatch it.
_FULL_INTEGRAL_CHUNK_ELEMS: int = 2_000_000


@dataclass(frozen=True)
class DerivedAfterglowScales:
    """Cached derived scales for a given physical+microphysics setup."""

    t_dec_s: float
    t_j_s: float
    q_dec: float
    q_j: float
    q_nr: float

    F_dec_Jy: float
    F_j_Jy: float
    F_nr_Jy: float


def _safe_log10(x: np.ndarray) -> np.ndarray:
    """log10 with masking for non-positive inputs."""

    out = np.full_like(x, np.nan, dtype=float)
    mask = x > 0
    out[mask] = np.log10(x[mask])
    return out


class DetectionRateModel:
    """Implements the   piecewise rate surface.

    Detection-window settings (both default off — the legacy simple mode):

    win_i_minus_one
        Require the detectability window to span only (i−1)·t_cad — the gaps
        between i detections — instead of i·t_cad, everywhere the required
        duration enters (q_i, D_i, and the window-mode D_eff).
    win_from_peak
        Measure the required window from the light-curve turn-on t_on ≈ t_p(q)
        instead of from the burst: an event at viewing angle q counts only if
        t_+(D) ≥ t_p,eff(q) + T_req (the off-axis flux is negligible before the
        peak, so epochs before t_p cannot be detections).  This makes the
        cadence distance cap q-dependent, so no closed dominant-term rectangle
        exists: `rate_log10` evaluates the q-integral internally and
        `compute_medians` always uses the numerical path.  Region masks keep
        the uncorrected q_i boundaries as an approximate classification.

    With both settings on, the criterion is t_+(D) − t_p,eff(q) ≥ (i−1)·t_cad.
    """

    def __init__(
        self,
        phys: AfterglowPhysicalParams,
        instrument: SurveyInstrumentParams,
        micro: MicrophysicsParams | None = None,
        pls: PLSModel | None = None,
        *,
        win_i_minus_one: bool = False,
        win_from_peak: bool = False,
    ):
        self.phys = phys
        self.instrument = instrument
        self.micro = micro if micro is not None else MicrophysicsParams()
        self.pls = pls if pls is not None else PLSG()
        self.win_i_minus_one = bool(win_i_minus_one)
        self.win_from_peak = bool(win_from_peak)

        self._derived = self._compute_derived_scales()

    # ---------- Derived scales ----------
    def _compute_derived_scales(self) -> DerivedAfterglowScales:
        t_dec = float(t_dec_s_fn(self.phys, self.micro))
        t_j = float(t_j_s_fn(self.phys, t_dec))

        qd = float(q_dec_fn(self.phys))
        qj = float(q_j_fn())
        qnr = float(q_nr_fn(self.phys))

        F_dec = float(self.pls.F_dec_Jy(self.phys, self.micro, t_dec))

        # These two flux scales are used to decide the branch for q_Euc and to
        # provide completeness (F_nr is also useful for debugging/plotting boundaries).
        aII = float(self.pls.a_II(self.phys.p))
        aIII = float(self.pls.a_III(self.phys.p))
        qd_tilde = qd - 1.0

        # Flux at q=q_j (q_tilde=1) at distance D_euc.
        # F_j = F_dec * qd_tilde^{2 a_II}
        F_j = F_dec * (qd_tilde ** (2.0 * aII))

        # Flux at q=q_nr (q_tilde=qnr-1) at distance D_euc, using phase III scaling.
        qnr_tilde = max(qnr - 1.0, 0.0)
        # F_nr/F_dec = qd_tilde^{-2} (qnr_tilde/qd_tilde)^{-2 a_III}
        F_nr = F_dec * (qd_tilde ** (-2.0)) * ((qnr_tilde / qd_tilde) ** (-2.0 * aIII))

        return DerivedAfterglowScales(
            t_dec_s=t_dec,
            t_j_s=t_j,
            q_dec=qd,
            q_j=qj,
            q_nr=qnr,
            F_dec_Jy=F_dec,
            F_j_Jy=F_j,
            F_nr_Jy=F_nr,
        )

    @property
    def derived(self) -> DerivedAfterglowScales:
        return self._derived

    # ---------- Core building blocks (vectorized) ----------
    def t_exp_s(self, N_exp: np.ndarray, t_cad_s: np.ndarray) -> np.ndarray:
        """Exposure time per pointing (vectorized). Returns NaN where t_exp ≤ 0."""

        t_exp = self.instrument.f_live * t_cad_s / N_exp - self.instrument.t_overhead_s
        return np.where(t_exp > 0, t_exp, np.nan)

    def F_lim_Jy(self, t_exp_s: np.ndarray) -> np.ndarray:
        """Limiting flux model F_lim ∝ t_exp^{-alpha} (vectorized)."""

        out = np.full_like(t_exp_s, np.inf, dtype=float)
        mask = t_exp_s > 0
        out[mask] = self.instrument.F_lim_ref_Jy * (t_exp_s[mask] / self.instrument.t_exp_ref_s) ** (
            -self.instrument.alpha
        )
        return out

    def f_Omega(self, N_exp: np.ndarray) -> np.ndarray:
        """Sky coverage fraction f_Omega = N_exp * Omega_exp / 4pi (vectorized)."""

        return (N_exp * self.instrument.omega_exp_sr) / (4.0 * np.pi)

    def q_Euc(self, F_lim: np.ndarray) -> np.ndarray:
        """q_Euc(F_lim): maximal q at which D_max(q) reaches D_euc.

        This matches the   piecewise definition, but is implemented in a
        PLS-aware manner via (a_II, a_III).
        """

        p = self.phys.p
        aII = float(self.pls.a_II(p))
        aIII = float(self.pls.a_III(p))

        qd_tilde = self.derived.q_dec - 1.0

        # Branch condition: F_lim < F_j  <=> q_Euc > q_j.
        deep = F_lim < self.derived.F_j_Jy

        q_tilde = np.empty_like(F_lim, dtype=float)

        # Deep branch: q_Euc in phase III.
        # q_tilde = qd_tilde^{(aIII-1)/aIII} (F_lim/F_dec)^(-1/(2 aIII))
        q_tilde[deep] = (qd_tilde ** ((aIII - 1.0) / aIII)) * (
            (F_lim[deep] / self.derived.F_dec_Jy) ** (-1.0 / (2.0 * aIII))
        )

        # Shallow branch: q_Euc in phase II.
        # q_tilde = qd_tilde (F_lim/F_dec)^(-1/(2 aII))
        q_tilde[~deep] = qd_tilde * (
            (F_lim[~deep] / self.derived.F_dec_Jy) ** (-1.0 / (2.0 * aII))
        )

        return 1.0 + q_tilde

    def _t_req_s(self, i_det: int, t_cad_s: np.ndarray) -> np.ndarray:
        """Required detectability duration T_req for i detections.

        T_req = i·t_cad in the legacy mode; (i−1)·t_cad under the
        `win_i_minus_one` setting (i detections only need to bracket the i−1
        cadence gaps between them).  At i_det = 1 the setting gives T_req = 0
        (a single detection at the peak suffices) — q_i then collapses to 1
        and D_i to +∞, i.e. the cadence constraint vanishes.
        """

        i_det = int(i_det)
        if i_det < 1:
            raise ValueError("i_det must be >= 1")
        i_eff = i_det - 1 if self.win_i_minus_one else i_det
        return float(i_eff) * np.asarray(t_cad_s, dtype=float)

    def q_i(self, i_det: int, t_cad_s: np.ndarray) -> np.ndarray:
        """q_i(t_cad): angle for which t_p(q_i) = T_req (= i * t_cad).

        Matches the   definition; T_req = (i−1)·t_cad under `win_i_minus_one`.
        """

        t_req = self._t_req_s(i_det, t_cad_s)
        before_tj = t_req < self.derived.t_j_s

        q_tilde = np.empty_like(t_req, dtype=float)
        q_tilde[before_tj] = (t_req[before_tj] / self.derived.t_j_s) ** (3.0 / 8.0)
        q_tilde[~before_tj] = (t_req[~before_tj] / self.derived.t_j_s) ** 0.5

        return 1.0 + q_tilde

    def D_dec(self, F_lim: np.ndarray) -> np.ndarray:
        """Distance at which an on-axis burst at t_dec reaches the limiting flux."""

        return self.phys.D_euc_cm * (F_lim / self.derived.F_dec_Jy) ** (-0.5)

    def D_i(self, i_det: int, t_cad_s: np.ndarray, F_lim: np.ndarray) -> np.ndarray:
        """D_i(t_cad, F_lim): distance where a burst at q_i is detectable at peak."""

        p = self.phys.p
        aII = float(self.pls.a_II(p))
        aIII = float(self.pls.a_III(p))

        qd_tilde = self.derived.q_dec - 1.0
        qi = self.q_i(i_det, t_cad_s)
        qi_tilde = qi - 1.0

        t_req = self._t_req_s(i_det, t_cad_s)
        before_tj = t_req < self.derived.t_j_s

        D_dec = self.D_dec(F_lim)

        out = np.empty_like(D_dec, dtype=float)
        # T_req = 0 (win_i_minus_one at i_det = 1) → q̃_i = 0 → the power below
        # diverges; the correct limit is no cadence constraint at all (+∞).
        with np.errstate(divide="ignore"):
            # Phase II: D_i = D_dec * (qi_tilde/qd_tilde)^(-aII)
            out[before_tj] = D_dec[before_tj] * ((qi_tilde[before_tj] / qd_tilde) ** (-aII))

            # Phase III: D_i = D_dec * qd_tilde^{-1} (qi_tilde/qd_tilde)^(-aIII)
            out[~before_tj] = D_dec[~before_tj] * (qd_tilde ** (-1.0)) * (
                (qi_tilde[~before_tj] / qd_tilde) ** (-aIII)
            )

        return out

    def D_from_t_plus(self, t_plus_s: np.ndarray, F_lim: np.ndarray) -> np.ndarray:
        """Distance at which the post-peak on-axis light curve crosses F_lim at t_+.

        Inverts the on-axis decline F(t, D) = F_dec·(D/D_euc)^{-2}·F̃(t) at
        F = F_lim for a prescribed crossing time t_+ = `t_plus_s` [s]:

            D(t_+) = D_dec · (t_+/t_dec)^{α_II/2},                    t_+ < t_j,
            D(t_+) = D_dec · (t_j/t_dec)^{α_II/2} · (t_+/t_j)^{α_III/2},  t_+ ≥ t_j,

        with α_phase the (negative) temporal flux indices of the PLS.  At
        t_+ = T_req this reproduces `D_i` exactly, and at t_+ = t_p(q) it
        reproduces the flux-limited D_max(q) — the two rectangle boundaries
        are the T_req-only and peak-only limits of this one relation.
        Valid for t_+ ≥ t_dec (the decline); t_plus_s ≤ 0 maps to +∞ (no
        constraint).  Result is clipped at D_dec, the t_+ = t_dec endpoint.
        """

        p = self.phys.p
        alpha_II = float(self.pls.alpha_II_temporal(p))    # < 0
        alpha_III = float(self.pls.alpha_III_temporal(p))  # < 0
        t_dec = float(self.derived.t_dec_s)
        t_j = float(self.derived.t_j_s)

        T = np.asarray(t_plus_s, dtype=float)
        D_dec = self.D_dec(F_lim)

        # over="ignore": the T ≤ 0 guard below discards the overflowed branch.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            T_safe = np.maximum(T, 1e-300)
            D_II = D_dec * (T_safe / t_dec) ** (alpha_II / 2.0)
            D_III = (
                D_dec
                * (t_j / t_dec) ** (alpha_II / 2.0)
                * (T_safe / t_j) ** (alpha_III / 2.0)
            )
            out = np.where(T < t_j, D_II, D_III)
            out = np.minimum(out, D_dec)
        return np.where(T > 0.0, out, np.inf)

    def _D_eff_window_cm(
        self,
        q: np.ndarray,
        i_det: int,
        t_cad_s: np.ndarray,
        F_lim: np.ndarray,
    ) -> np.ndarray:
        """q-dependent cadence distance under `win_from_peak` [cm].

        Largest distance at which the above-threshold window [t_p, t_+] spans
        the required duration: t_+(D) = t_p,eff(q) + T_req.  Off-axis events
        must stay above the limit for the full window *after* they turn on;
        the legacy D_i measures the window from the burst instead and so
        credits detections to the pre-peak interval where the off-axis flux
        is negligible.
        """

        T_req = self._t_req_s(i_det, t_cad_s)
        T_eff = self._t_p_eff(q) + T_req
        return self.D_from_t_plus(T_eff, F_lim)

    # ---------- Piecewise rate surface ----------
    def region_masks(
            self,
            i_det: int,
            N_exp: np.ndarray,
            t_cad_s: np.ndarray,
            *,
            include_unphysical: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Return boolean masks for the seven   regions.

        This version is robust on regime boundaries: any point in A0 is assigned
        to exactly one region even when equalities occur (within a tolerance).
        """

        Nmax = N_exp_max(self.instrument)
        t_exp = self.t_exp_s(N_exp, t_cad_s)

        eps = _REGION_BOUNDARY_EPS

        A0 = (
                (N_exp >= 1.0 - eps)
                & (N_exp <= Nmax + eps)
                & np.isfinite(t_exp)
                & (t_exp > 0.0)
        )

        F_lim = self.F_lim_Jy(self.t_exp_s(N_exp, t_cad_s))
        qE = self.q_Euc(F_lim)
        qi = self.q_i(i_det, t_cad_s)

        qnr = float(self.derived.q_nr)
        qj = float(self.derived.q_j)
        qd = float(self.derived.q_dec)

        eps = _REGION_BOUNDARY_EPS

        def ge(a, b):
            return a >= (b - eps)

        def lt(a, b):
            return a < (b - eps)

        # Primary region definitions (same logic, but tolerant)
        A1 = A0 & ge(qE, qnr) & ge(qE, qi)
        A2 = A0 & ge(qE, qj) & lt(qE, qnr) & ge(qE, qi)
        A3 = A0 & ge(qE, qd) & lt(qE, qj) & ge(qE, qi)

        A4 = A0 & ge(qi, qnr) & lt(qE, qi)
        A5 = A0 & ge(qi, qj) & lt(qi, qnr) & lt(qE, qi)
        A6 = A0 & ge(qi, qd) & lt(qi, qj) & lt(qE, qi)

        A7 = A0 & lt(qE, qd) & lt(qi, qd)

        # Fallback: ensure complete coverage of A0 (boundary equalities, roundoff, etc.)
        assigned = A1 | A2 | A3 | A4 | A5 | A6 | A7
        U = A0 & (~assigned)

        if np.any(U):
            # Case split matches the conceptual logic:
            # If qE >= qi then we are in the "qE-limited" family (A1/A2/A3/A7),
            # else in the "qi-limited" family (A4/A5/A6/A7).
            E_ge_I = ge(qE, qi)

            U_E = U & E_ge_I
            if np.any(U_E):
                m1 = U_E & ge(qE, qnr)
                A1 |= m1

                m2 = U_E & (~m1) & ge(qE, qj)
                A2 |= m2

                m3 = U_E & (~m1) & (~m2) & ge(qE, qd)
                A3 |= m3

                A7 |= U_E & (~m1) & (~m2) & (~m3)

            U_I = U & (~E_ge_I)
            if np.any(U_I):
                m4 = U_I & ge(qi, qnr)
                A4 |= m4

                m5 = U_I & (~m4) & ge(qi, qj)
                A5 |= m5

                m6 = U_I & (~m4) & (~m5) & ge(qi, qd)
                A6 |= m6

                A7 |= U_I & (~m4) & (~m5) & (~m6)

        # Optional: enforce mutual exclusivity (paranoia)
        # Priority order A1..A7
        A2 &= ~A1
        A3 &= ~(A1 | A2)
        A4 &= ~(A1 | A2 | A3)
        A5 &= ~(A1 | A2 | A3 | A4)
        A6 &= ~(A1 | A2 | A3 | A4 | A5)
        A7 &= ~(A1 | A2 | A3 | A4 | A5 | A6)

        return {"A0": A0, "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A6": A6, "A7": A7}

    # ---------- Fading-rate filter ----------
    def _t_first_caps(
        self,
        i_det: int,
        t_cad_s: np.ndarray,
        s_fade: float,
        s_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Phase-resolved caps on the first-detection time from the s_fade filter.

        Real surveys reject candidates that don't fade fast enough across their
        i detections.  For a post-peak power law F(t) ∝ t^α (α < 0) the measured
        slope between t_first and t_last = t_first + (i−1) t_cad depends only on
        t_first (not on t_p separately):
            s_meas = −2.5 α · log10(1 + (i−1) t_cad / t_first) · DAY_S / ((i−1) t_cad).
        Requiring s_meas ≥ s_fade and solving for the latest admissible first
        detection gives (discrete mode):
            t_f,s = (i−1) t_cad / (10^E − 1),  E = s_fade (i−1) t_cad / (DAY_S · 2.5|α|).
        The continuous version uses |dm/dt|(t_first) = 2.5|α|/(t_first · ln10·sec/day):
            t_f,s = 2.5|α| DAY_S / (s_fade · ln10).
        Pass ⇔ t_first ≤ t_f,s.

        Returns
        -------
        t_f_s_II, t_f_s_III : ndarray
            Latest admissible first-detection time [s] for Phase II and Phase
            III viewers, broadcast to the shape of t_cad_s.  +∞ = cut bypassed
            (s_fade ≤ 0, or i_det < 2 in discrete mode where the slope is
            undefined).
        """
        t_cad_arr = np.asarray(t_cad_s, dtype=float)

        # Bypass conditions.
        s_fade_f = float(s_fade)
        i_det_i = int(i_det)
        no_cut = (
            s_fade_f <= 0.0
            or (s_mode == "discrete" and i_det_i < 2)
        )
        if no_cut:
            inf_arr = np.full(t_cad_arr.shape, np.inf, dtype=float)
            return inf_arr, inf_arr

        p = self.phys.p
        LN10 = float(np.log(10.0))

        # |α| in temporal sense (positive numbers).
        A_II  = 2.5 * abs(float(self.pls.alpha_II_temporal(p)))
        A_III = 2.5 * abs(float(self.pls.alpha_III_temporal(p)))

        if s_mode == "continuous":
            t_f_s_II  = A_II  * DAY_S / (s_fade_f * LN10)
            t_f_s_III = A_III * DAY_S / (s_fade_f * LN10)
            t_f_s_II_arr  = np.full(t_cad_arr.shape, t_f_s_II,  dtype=float)
            t_f_s_III_arr = np.full(t_cad_arr.shape, t_f_s_III, dtype=float)
        else:  # "discrete"
            # dt = (i−1) · t_cad ; E_phase = s_fade · dt / (DAY_S · A_phase)
            dt = float(i_det_i - 1) * t_cad_arr
            E_II  = s_fade_f * dt / (DAY_S * A_II)
            E_III = s_fade_f * dt / (DAY_S * A_III)
            # 10^E − 1 = expm1(E·ln10); numerically robust for small E.
            denom_II  = np.expm1(E_II  * LN10)
            denom_III = np.expm1(E_III * LN10)
            # When denom → 0 (e.g. t_cad → 0) the cut becomes effectively absent.
            tiny = 1e-300
            t_f_s_II_arr  = np.where(denom_II  > tiny, dt / np.maximum(denom_II,  tiny), np.inf)
            t_f_s_III_arr = np.where(denom_III > tiny, dt / np.maximum(denom_III, tiny), np.inf)

        return t_f_s_II_arr, t_f_s_III_arr

    def _q_s_fading_caps(
        self,
        i_det: int,
        t_cad_s: np.ndarray,
        s_fade: float,
        s_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Phase-resolved viewing-angle caps imposed by the fading-rate filter.

        Best-case (dominant-term) form of the s_fade cut: the first detection is
        assumed to land at the light-curve peak, t_first ≈ t_p(q), so the
        first-detection cap t_f,s of `_t_first_caps` becomes a cap on the peak
        time and maps to an upper viewing-angle bound via t_p = t_j · q̃^β with
        β = 8/3 (II) or 2 (III):
            q_s = 1 + (t_f,s / t_j)^{1/β}.
        This is exact in the t_cad → 0 limit of the uniform-start model and an
        optimistic upper bound otherwise; the full-integral path uses the
        uniform-start survival probability `_fading_survival` instead.

        Returns
        -------
        q_s_II, q_s_III : ndarray
            Upper q-caps for Phase II and Phase III viewers, broadcast to the
            shape of t_cad_s.  +∞ means "no cap" (cut bypassed).
        """
        t_f_s_II, t_f_s_III = self._t_first_caps(i_det, t_cad_s, s_fade, s_mode)
        t_j = float(self.derived.t_j_s)

        # Map t_f,s → q̃_s → q_s per phase.  +∞ in → +∞ out (no cap).
        with np.errstate(over="ignore"):
            q_s_II  = 1.0 + (t_f_s_II  / t_j) ** (3.0 / 8.0)
            q_s_III = 1.0 + (t_f_s_III / t_j) ** 0.5
        return q_s_II, q_s_III

    def _fading_survival(
        self,
        q: np.ndarray,
        i_det: int,
        t_cad_s: np.ndarray,
        s_fade: float,
        s_mode: str,
        *,
        fade_random_start: bool = True,
    ) -> np.ndarray:
        """Pointwise pass probability of the s_fade filter for a uniform start.

        Generalizes the best-case t_first ≈ t_p(q) assumption: the survey's
        visit schedule has a random phase relative to the burst, so the first
        post-peak detection lands at t_first = t_p + u with u ~ U(0, t_cad).
        Pass ⇔ t_first ≤ t_f,s (see `_t_first_caps`), giving

            P(q) = clip( (t_f,s − t_p(q)) / t_cad, 0, 1 ),
            t_p(q) = t_j · max(q − 1, 0)^β,  β = 8/3 (II), 2 (III).

        The effective peak time carries the physical t_dec floor,
        t_p,eff = max(t_j·q̃₊^β, t_dec) (see `_t_p_eff`) — exact for on-axis
        viewers since t_j·q̃_dec^{8/3} = t_dec identically; for the fade weight
        the floor matters only at O(t_dec/t_cad).  On-axis viewers (q < q_dec)
        are lumped with Phase II, and P → 1 at the old hard cap's t_cad → 0
        limit.  When t_f,s < t_cad even on-axis bursts pass only with
        probability ≈ t_f,s/t_cad: with a long cadence the afterglow is usually
        caught late, when its mag/day decline is slow.

        Parameters
        ----------
        q : ndarray
            Viewing-angle grid, broadcastable against t_cad_s (e.g. shape
            (N_q, *grid) against (*grid)).

        Returns
        -------
        P : ndarray
            Pass probability in [0, 1], broadcast of q against t_cad_s.
            Exactly 1.0 everywhere when the cut is bypassed.
        """
        t_cad_arr = np.asarray(t_cad_s, dtype=float)
        q_arr = np.asarray(q, dtype=float)

        s_fade_f = float(s_fade)
        i_det_i = int(i_det)
        no_cut = (
            s_fade_f <= 0.0
            or (s_mode == "discrete" and i_det_i < 2)
        )
        if no_cut:
            shape = np.broadcast(q_arr, t_cad_arr).shape
            return np.ones(shape, dtype=float)

        t_f_s_II, t_f_s_III = self._t_first_caps(i_det, t_cad_arr, s_fade, s_mode)
        q_j = float(self.derived.q_j)

        t_p_eff = self._t_p_eff(q_arr)
        # clip() maps a per-element t_f,s = +∞ (expm1 underflow) to exactly 1.0,
        # so the bypass stays bit-neutral under multiplication.  With
        # fade_random_start=False the uniform-start ramp collapses to its u=0
        # best case: a hard gate 1{t_f,s ≥ t_p,eff} (+∞ cap ⇒ 1, bypass-neutral).
        with np.errstate(invalid="ignore", over="ignore"):
            if fade_random_start:
                P_II  = np.clip((t_f_s_II  - t_p_eff) / t_cad_arr, 0.0, 1.0)
                P_III = np.clip((t_f_s_III - t_p_eff) / t_cad_arr, 0.0, 1.0)
            else:
                P_II  = np.where(t_f_s_II  >= t_p_eff, 1.0, 0.0)
                P_III = np.where(t_f_s_III >= t_p_eff, 1.0, 0.0)
        # On-axis (q < q_dec) lumped with Phase II, mirroring the full-integral
        # phase masks (on_axis | phase_II ⇔ q < q_j).
        return np.where(q_arr < q_j, P_II, P_III)

    def _t_p_eff(self, q: np.ndarray) -> np.ndarray:
        """Effective peak time max(t_j·q̃₊^β, t_dec), β = 8/3 (q < q_j) else 2.

        The t_dec floor is exact for on-axis viewers: t_j·q̃_dec^{8/3} = t_dec
        identically, so the floor binds precisely for q ≤ q_dec (where the
        model's light curve peaks at t_dec with the constant D̃_dec horizon)
        and is a no-op for q ≥ q_dec.
        """
        q_arr = np.asarray(q, dtype=float)
        t_j = float(self.derived.t_j_s)
        t_dec = float(self.derived.t_dec_s)
        q_j = float(self.derived.q_j)
        qt = np.maximum(q_arr - 1.0, 0.0)
        t_p = np.where(q_arr < q_j, t_j * qt ** (8.0 / 3.0), t_j * qt ** 2.0)
        return np.maximum(t_p, t_dec)

    def _rise_eta(
        self,
        t_cad_s: np.ndarray,
        s_rise: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rise-filter flux-limit boost η and its inverse square root.

        A previous non-detection one cadence before the first detection,
        bounded by the limiting magnitude, implies a rise rate of at least
        (m_lim − m_first)/t_cad.  Requiring that to exceed s_rise [mag/day] is
        equivalent to F(t_first) ≥ η·F_lim with
            η = 10^E,  E = s_rise · t_cad / (2.5 · DAY_S).
        Returns (η, η^{−1/2}); the inverse root is computed directly as
        10^{−E/2} so it underflows gracefully to 0.0 where η overflows to +∞.
        Bypass (s_rise ≤ 0) returns (ones, ones).
        """
        t_cad_arr = np.asarray(t_cad_s, dtype=float)
        s_rise_f = float(s_rise)
        if s_rise_f <= 0.0:
            ones = np.ones(t_cad_arr.shape, dtype=float)
            return ones, ones
        E = s_rise_f * t_cad_arr / (2.5 * DAY_S)
        with np.errstate(over="ignore"):
            eta = 10.0 ** E
        inv_sqrt_eta = 10.0 ** (-0.5 * E)
        return eta, inv_sqrt_eta

    def _rise_fade_windows(
        self,
        q: np.ndarray,
        i_det: int,
        t_cad_s: np.ndarray,
        s_fade: float,
        s_rise: float,
        s_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Shared per-point quantities for the joint fade+rise weights.

        Returns (t_p_eff, t_fs_sel, k_sel, inv_sqrt_eta): the floored peak
        time, the phase-selected fade cap on t_first (+∞ when the fade cut is
        bypassed), the phase-selected exponent k = 2/|α_phase| of the
        limit-crossing time t_lim = t_p,eff·(D̃_0/D̃)^k, and the rise boost's
        η^{−1/2}.  Phase split at q_j (on-axis lumped with Phase II).
        """
        q_arr = np.asarray(q, dtype=float)
        t_cad_arr = np.asarray(t_cad_s, dtype=float)
        p = self.phys.p
        q_j = float(self.derived.q_j)
        is_II = q_arr < q_j

        t_p_eff = self._t_p_eff(q_arr)
        t_fs_II, t_fs_III = self._t_first_caps(i_det, t_cad_arr, s_fade, s_mode)
        t_fs_sel = np.where(is_II, t_fs_II, t_fs_III)

        k_II = 2.0 / abs(float(self.pls.alpha_II_temporal(p)))
        k_III = 2.0 / abs(float(self.pls.alpha_III_temporal(p)))
        k_sel = np.where(is_II, k_II, k_III)

        _, inv_sqrt_eta = self._rise_eta(t_cad_arr, s_rise)
        return t_p_eff, t_fs_sel, k_sel, inv_sqrt_eta

    def _joint_survival(
        self,
        q: np.ndarray,
        D_tilde: np.ndarray,
        D_tilde_max: np.ndarray,
        i_det: int,
        t_cad_s: np.ndarray,
        s_fade: float,
        s_rise: float,
        s_mode: str,
        *,
        rise_random_start: bool = True,
        fade_random_start: bool = True,
    ) -> np.ndarray:
        """Pointwise joint fade+rise pass probability for a uniform start.

        Both cuts are conditions on the SAME uniform start offset u
        (comonotone, not independent), so the joint probability is the min of
        the two admissible windows, not their product:

            pass_fade ⇔ u ≤ t_f,s − t_p,eff
            pass_rise ⇔ u ≤ t_lim(q, D̃) − t_p,eff
            w = clip( min(t_f,s, t_lim) − t_p,eff, 0, t_cad ) / t_cad,

        where t_lim = t_p,eff·(D̃_0/D̃)^k is the time the light curve fades to
        the raised limit η·F_lim, D̃_0(q) = D̃_max(q)·η^{−1/2}, k = 2/|α_phase|.

        `D_tilde` and `D_tilde_max` are distances normalized to D_euc.  With
        s_rise ≤ 0 this delegates to `_fading_survival` (bit-compatible).

        A `*_random_start=False` flag replaces that cut's uniform-start ramp
        with its u=0 best case — a hard gate on the same window (fade:
        1{t_f,s ≥ t_p,eff}; rise: 1{t_lim ≥ t_p,eff} ⇔ 1{D̃ ≤ D̃_0}).  Both
        True (default) reproduces the uniform-start survival weight exactly.
        """
        if float(s_rise) <= 0.0:
            return self._fading_survival(
                q, i_det, t_cad_s, s_fade, s_mode,
                fade_random_start=fade_random_start,
            )

        t_cad_arr = np.asarray(t_cad_s, dtype=float)
        t_p_eff, t_fs_sel, k_sel, ise = self._rise_fade_windows(
            q, i_det, t_cad_arr, s_fade, s_rise, s_mode
        )
        D0 = np.asarray(D_tilde_max, dtype=float) * ise
        D_safe = np.maximum(np.asarray(D_tilde, dtype=float), 1e-300)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            t_lim = t_p_eff * (D0 / D_safe) ** k_sel
            # Hard-boundary (best-case u=0) override per cut: gate the window
            # to +∞ when it passes at u=0, −∞ (kills the point) otherwise.
            if not fade_random_start:
                t_fs_sel = np.where(t_fs_sel >= t_p_eff, np.inf, -np.inf)
            if not rise_random_start:
                t_lim = np.where(t_lim >= t_p_eff, np.inf, -np.inf)
            w = np.clip(
                (np.minimum(t_fs_sel, t_lim) - t_p_eff) / t_cad_arr, 0.0, 1.0
            )
        return w

    def _weighted_D_volume(
        self,
        q: np.ndarray,
        D_tilde_max: np.ndarray,
        D_eff: np.ndarray,
        D_min_norm: float,
        i_det: int,
        t_cad_s: np.ndarray,
        s_fade: float,
        s_rise: float,
        s_mode: str,
        *,
        rise_random_start: bool = True,
        fade_random_start: bool = True,
    ) -> np.ndarray:
        """Closed-form joint-weighted D-volume for the full-integral q-integrand.

        Replaces max(D_eff³ − D̃_min³, 0)·P_fade with

            V_w(q) = ∫_{D̃_min}^{D̃_eff(q)} 3D̃² · w(q, D̃) dD̃,

        w being the joint survival of `_joint_survival`.  Since the rise
        window w_r(D̃) = clip(t_lim − t_p,eff, 0, t_cad) decreases with D̃ and
        the fade window w_f is D̃-independent, w = min(w_f, w_r)/t_cad is a
        constant piece followed by a power-law ramp and zero:

            w_f  = clip(t_f,s − t_p,eff, 0, t_cad)          (0 ⇒ V_w = 0)
            D̃_c = D̃_0·(t_p,eff/(t_p,eff + w_f))^{1/k}       (ramp start)
            V_w  = (w_f/t_cad)·[min(D̃_eff, D̃_c)³ − D̃_min³]₊
                 + (t_p,eff/t_cad)·[G(U) − G(L)]
            G(D̃) = 3·D̃_0^k·D̃^{3−k}/(3−k) − D̃³
            U = min(D̃_eff, D̃_0);  L = min(max(D̃_min, D̃_c), U)

        with k = 2/|α_phase| < 3 always (p > 2 ⇒ k_II ≤ 8/3·1/(p−1) < 3,
        k_III = 2/p < 1), so the (3−k) pole is unreachable.  With s_rise ≤ 0
        this reproduces the fade-only product exactly (bit-compatible).

        A `*_random_start=False` flag applies that cut as its u=0 best-case
        hard boundary inside the integral (matching the dominant-term
        treatment): hard fade gates t_f,s → ±∞ (a passing fade then gives
        w_f = t_cad, i.e. the rise-only integral; a failing one gives V_w = 0),
        and hard rise sets the ramp start D̃_c = D̃_0, collapsing the power-law
        ramp to a sharp cutoff at D̃ ≤ D̃_0 (term_ramp auto-zeros).  Both True
        (default) is the uniform-start survival weight.
        """
        if float(s_rise) <= 0.0:
            P_fade = self._fading_survival(
                q, i_det, t_cad_s, s_fade, s_mode,
                fade_random_start=fade_random_start,
            )
            return np.maximum(D_eff ** 3 - D_min_norm ** 3, 0.0) * P_fade

        t_cad_arr = np.asarray(t_cad_s, dtype=float)
        t_p_eff, t_fs_sel, k_sel, ise = self._rise_fade_windows(
            q, i_det, t_cad_arr, s_fade, s_rise, s_mode
        )
        D0 = np.asarray(D_tilde_max, dtype=float) * ise
        D_eff_arr = np.asarray(D_eff, dtype=float)
        D_min_n = float(D_min_norm)
        D_min3 = D_min_n ** 3

        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            # Hard fade (best-case u=0): gate t_f,s → ±∞ before forming w_f, so
            # a passing fade → w_f = t_cad and a failing one → w_f = 0.
            if not fade_random_start:
                t_fs_sel = np.where(t_fs_sel >= t_p_eff, np.inf, -np.inf)
            # Fade window [s], clipped; t_fs = +∞ (bypass) → exactly t_cad.
            w_f = np.clip(t_fs_sel - t_p_eff, 0.0, t_cad_arr)
            # Ramp start: where the rise window first drops to w_f.
            D_c = D0 * (t_p_eff / (t_p_eff + w_f)) ** (1.0 / k_sel)
            # Hard rise (best-case u=0): collapse the ramp to a step at D̃ ≤ D̃_0.
            if not rise_random_start:
                D_c = D0

            term_const = (w_f / t_cad_arr) * np.maximum(
                np.minimum(D_eff_arr, D_c) ** 3 - D_min3, 0.0
            )

            U = np.minimum(D_eff_arr, D0)
            L = np.minimum(np.maximum(D_min_n, D_c), U)
            expo = 3.0 - k_sel
            G_U = 3.0 * D0 ** k_sel * U ** expo / expo - U ** 3
            G_L = 3.0 * D0 ** k_sel * L ** expo / expo - L ** 3
            term_ramp = (t_p_eff / t_cad_arr) * np.maximum(G_U - G_L, 0.0)

        return term_const + term_ramp

    def rate_log10(
        self,
        i_det: int,
        N_exp: np.ndarray,
        t_cad_s: np.ndarray,
        *,
        q_min: float = 0.0,
        D_min_cm: float = 0.0,
        s_fade: float = 0.0,
        s_rise: float = 0.0,
        s_mode: str = "discrete",
        rise_random_start: bool = True,
        fade_random_start: bool = True,
        return_components: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute log10 R_det for the given strategy grid.

        This mirrors the   definition:

            R_T(x,y) = {A1: R1, A2: R2, ..., A7: R7}

        where each Rk is itself a log10(...) expression.

        Parameters
        ----------
        q_min, D_min_cm : float, optional
            Lower bounds applied to the detection volume: only count GRBs with
            viewing-angle parameter q ≥ q_min and source distance D ≥ D_min_cm.
            Each regime's rate gets clipping factors max(q_max² − q_min², 0)
            and max(D_eff³ − D_min³, 0).  Default 0 / 0 = no filter.
        s_fade : float, optional
            Minimum required fading rate of the i-detection light-curve segment
            in mag/day.  In this dominant-term (closed-form) mode it translates
            to a best-case phase-resolved upper q-cap via `_q_s_fading_caps()`
            (first detection assumed at the peak); the full-integral mode uses
            the uniform-start survival probability `_fading_survival()` instead.
            Default 0 = no fading filter.
        s_rise : float, optional
            Minimum required rise rate in mag/day implied by a previous-visit
            non-detection at the limiting magnitude: (m_lim − m_first)/t_cad ≥
            s_rise ⇔ F(t_first) ≥ η·F_lim with η = 10^(s_rise·t_cad/2.5 DAY_S)
            (see `_rise_eta`).  In this dominant-term mode (first detection at
            the peak) the raised limit shifts the flux-limited boundaries:
            q_Euc → q_Euc(η F_lim), the A4-A6 boundary → q_Euc(η F_lim (D_i/
            D_euc)²) (the angle where D_max(q)/√η = D_i; equals q_i at η = 1),
            and A7's D_dec → D_dec·η^{−1/2}.  Each of r1-r6 is the max of the
            angular-truncated term and an on-axis term [q ≤ q_dec, D ≤ D_dec·
            η^{−1/2}] — two closed-form lower bounds of the pass volume; the
            max keeps the deep-cut limit correct (rates → 0 instead of
            saturating when the boundary angle collapses to q = 1).  Unlike the
            fade filter, applies at i_det = 1 too.  Default 0 = off.
        s_mode : {"discrete", "continuous"}, optional
            Whether s_fade is interpreted as Δm/Δt across the i detections
            ("discrete", default — matches real survey alert filters and is
            undefined at i_det = 1, bypassed there) or |dm/dt| at the first
            detection ("continuous").  The two coincide in the t_cad → 0 limit.
            Fade-only: the rise measurement is inherently a previous-visit
            finite difference.

        Returns
        -------
        log10_rate : ndarray
            log10 R_det [yr^-1]. NaN outside the physical region A0.
        components : dict, optional
            If return_components=True, also returns a dict with intermediate
            arrays (F_lim, q_Euc, q_i, D_dec, D_i, masks...).

        Notes
        -----
        Under `win_from_peak` the cadence distance cap is q-dependent and no
        closed dominant-term rectangle exists, so the rate is evaluated with
        the q-integral (`rate_log10_full_integral`) internally; the component
        arrays (masks, q_i, D_i) keep their legacy rectangle definitions as an
        approximate classification.
        """

        if self.win_from_peak and not return_components:
            return self.rate_log10_full_integral(
                i_det, N_exp, t_cad_s,
                q_min=q_min, D_min_cm=D_min_cm,
                s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
                rise_random_start=rise_random_start,
                fade_random_start=fade_random_start,
            )

        N_exp = np.asarray(N_exp, dtype=float)
        t_cad_s = np.asarray(t_cad_s, dtype=float)

        # Broadcast
        shape = np.broadcast(N_exp, t_cad_s).shape
        N_exp_b = np.broadcast_to(N_exp, shape)
        t_cad_b = np.broadcast_to(t_cad_s, shape)

        t_exp = self.t_exp_s(N_exp_b, t_cad_b)
        F_lim = self.F_lim_Jy(t_exp)
        fO = self.f_Omega(N_exp_b)

        qE = self.q_Euc(F_lim)
        qi = self.q_i(i_det, t_cad_b)

        D_dec = self.D_dec(F_lim)
        D_i = self.D_i(i_det, t_cad_b, F_lim)

        masks = self.region_masks(i_det, N_exp_b, t_cad_b, include_unphysical=False)

        R_int = self.phys.R_int_yr
        theta_j = self.phys.theta_j_rad
        D_euc = self.phys.D_euc_cm
        q_nr_val = float(self.derived.q_nr)
        q_dec_val = float(self.derived.q_dec)

        # Filter clipping factors.  At q_min=0, D_min=0 these reduce to q_max² and D_eff³,
        # so each rₖ matches the pre-filter expression to floating-point.
        q_min_sq = float(q_min) ** 2
        D_min_norm_cubed = (float(D_min_cm) / D_euc) ** 3

        def _q_factor(q_max_sq: np.ndarray | float) -> np.ndarray:
            return np.maximum(q_max_sq - q_min_sq, 0.0)

        def _D_factor(D_norm_cubed: np.ndarray | float) -> np.ndarray:
            return np.maximum(D_norm_cubed - D_min_norm_cubed, 0.0)

        # Fading-rate q-caps.  +∞ at s_fade=0 (or i_det<2 in discrete mode) so the
        # np.minimum() below collapses to the original q_max and rates are unchanged.
        q_s_II, q_s_III = self._q_s_fading_caps(i_det, t_cad_b, s_fade, s_mode)

        # Rise-rate boundaries at the raised limit η·F_lim (previous-visit
        # non-detection).  Bypass (s_rise ≤ 0) → +∞ caps and zero on-axis
        # alternative terms, keeping every r_k value bit-identical to the
        # fade-only path (min/max with ∞/0 are exact no-ops on non-negatives).
        s_rise_on = float(s_rise) > 0.0
        if s_rise_on:
            eta, ise = self._rise_eta(t_cad_b, s_rise)
            with np.errstate(over="ignore", invalid="ignore"):
                qE_r = self.q_Euc(eta * F_lim)
                q_ri = self.q_Euc(eta * F_lim * (D_i / D_euc) ** 2)
            D_dec_r = D_dec * ise
            # The q_Euc-extrapolated boundary is the true flux boundary only
            # while it sits above q_dec; below the crossing the on-axis D̃_dec
            # plateau makes the angular box overcount, so the angular term is
            # zeroed and the on-axis term takes over.  At the crossing
            # (qE_r = q_dec ⇔ D_dec·η^{−1/2} = D_eff-scale) the two terms are
            # equal, so np.maximum() below stays continuous in s_rise.
            valid_E = qE_r >= q_dec_val
            valid_i = q_ri >= q_dec_val
        else:
            qE_r = q_ri = np.inf
            D_dec_r = np.inf
            valid_E = valid_i = True

        # Effective q_max per regime: phase-mapped fading cap and rise boundary
        # min'd with the regime's original geometric bound.  Fade phase map:
        # A1/A2/A4/A5 sit in Phase III (q > q_j), A3/A6/A7 in Phase II.  Rise
        # boundary: qE_r for the flux-limited A1-A3, q_ri for the cadence-
        # limited A4-A6 (r1/r4 split the previously shared q_nr cap).
        q_nr_eff   = np.minimum(q_nr_val,  q_s_III)
        q_nr_eff_1 = np.minimum(q_nr_eff,  qE_r)
        q_nr_eff_4 = np.minimum(q_nr_eff,  q_ri)
        qE_eff_III = np.minimum(np.minimum(qE, q_s_III), qE_r)
        qE_eff_II  = np.minimum(np.minimum(qE, q_s_II),  qE_r)
        qi_eff_III = np.minimum(np.minimum(qi, q_s_III), q_ri)
        qi_eff_II  = np.minimum(np.minimum(qi, q_s_II),  q_ri)
        q_dec_eff  = np.minimum(q_dec_val, q_s_II)

        # Region rates (linear), un-simplified so the q² / D³ factors are explicit.
        # r1's pre-filter form `fO * R_int` exploits θ_j² · q_nr² = 2; we re-introduce
        # the explicit factors here so the filter can clip them generically.
        D_eff7 = np.minimum(np.minimum(D_dec, D_i), D_dec_r) / D_euc
        D_norm_cubed_A123 = 1.0                                    # D̃_eff = 1 (D_Euc cap)
        D_norm_cubed_A456 = (D_i / D_euc) ** 3                     # D̃_eff = D_i / D_euc
        D_norm_cubed_A7   = D_eff7 ** 3                            # D̃_eff = min(D_dec, D_i, D_dec_r)/D_euc

        # On-axis alternative terms for the rise cut (deep-cut correctness):
        # counts the [q ≤ q_dec, D ≤ D_dec·η^{−1/2}] sub-volume, which survives
        # after the raised-limit boundary angle collapses to q = 1.  Each term
        # is a closed-form lower bound of the true pass volume; np.maximum()
        # below is the dominant-term selection between the two.
        if s_rise_on:
            D_tilde_dec_r = D_dec_r / D_euc

            def _onaxis_term(q_max_geo, D_tilde_eff_reg):
                q_hi = np.minimum(np.minimum(q_dec_val, q_max_geo), q_s_II)
                D_hi = np.minimum(D_tilde_eff_reg, D_tilde_dec_r)
                return _q_factor(q_hi ** 2) * _D_factor(D_hi ** 3)

            D_tilde_i = D_i / D_euc
            t_on_1  = _onaxis_term(q_nr_val, 1.0)
            t_on_23 = _onaxis_term(qE,       1.0)
            t_on_4  = _onaxis_term(q_nr_val, D_tilde_i)
            t_on_56 = _onaxis_term(qi,       D_tilde_i)
        else:
            t_on_1 = t_on_23 = t_on_4 = t_on_56 = 0.0

        base = 0.5 * fO * (theta_j ** 2) * R_int

        def _ang(term, valid):
            # Angular term, zeroed where the extrapolated rise boundary is
            # fictitious (below q_dec).  valid=True (bypass) → exact no-op.
            return np.where(valid, term, 0.0)

        r1 = base * np.maximum(_ang(_q_factor(q_nr_eff_1 ** 2) * _D_factor(D_norm_cubed_A123), valid_E), t_on_1)
        r2 = base * np.maximum(_ang(_q_factor(qE_eff_III ** 2) * _D_factor(D_norm_cubed_A123), valid_E), t_on_23)
        r3 = base * np.maximum(_ang(_q_factor(qE_eff_II  ** 2) * _D_factor(D_norm_cubed_A123), valid_E), t_on_23)
        r4 = base * np.maximum(_ang(_q_factor(q_nr_eff_4 ** 2) * _D_factor(D_norm_cubed_A456), valid_i), t_on_4)
        r5 = base * np.maximum(_ang(_q_factor(qi_eff_III ** 2) * _D_factor(D_norm_cubed_A456), valid_i), t_on_56)
        r6 = base * np.maximum(_ang(_q_factor(qi_eff_II  ** 2) * _D_factor(D_norm_cubed_A456), valid_i), t_on_56)
        r7 = base * _q_factor(q_dec_eff  ** 2) * _D_factor(D_norm_cubed_A7)

        # Convert to log10 safely.
        R1 = _safe_log10(r1)
        R2 = _safe_log10(r2)
        R3 = _safe_log10(r3)
        R4 = _safe_log10(r4)
        R5 = _safe_log10(r5)
        R6 = _safe_log10(r6)
        R7 = _safe_log10(r7)

        logR = np.full(shape, np.nan, dtype=float)

        # Assign region-by-region. Regions should be mutually exclusive.
        logR[masks["A1"]] = R1[masks["A1"]]
        logR[masks["A2"]] = R2[masks["A2"]]
        logR[masks["A3"]] = R3[masks["A3"]]
        logR[masks["A4"]] = R4[masks["A4"]]
        logR[masks["A5"]] = R5[masks["A5"]]
        logR[masks["A6"]] = R6[masks["A6"]]
        logR[masks["A7"]] = R7[masks["A7"]]

        # win_from_peak + return_components: the early return above was
        # skipped so the caller still gets the rectangle components, but the
        # rate itself comes from the q-integral (see Notes).
        if self.win_from_peak:
            logR = self.rate_log10_full_integral(
                i_det, N_exp, t_cad_s,
                q_min=q_min, D_min_cm=D_min_cm,
                s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
                rise_random_start=rise_random_start,
                fade_random_start=fade_random_start,
            )

        if not return_components:
            return logR

        components: Dict[str, np.ndarray] = {
            "t_exp_s": t_exp,
            "F_lim_Jy": F_lim,
            "f_Omega": fO,
            "q_Euc": qE,
            "q_i": qi,
            "D_dec_cm": D_dec,
            "D_i_cm": D_i,
            **masks,
        }
        return logR, components

    def rate(
        self,
        i_det: int,
        N_exp: np.ndarray,
        t_cad_s: np.ndarray,
    ) -> np.ndarray:
        """Return the detection rate R_det in yr^-1 (linear, not log10)."""

        logR = self.rate_log10(i_det, N_exp, t_cad_s)
        return 10.0 ** logR

    def rate_log10_full_integral(
        self,
        i_det: int,
        N_exp: np.ndarray,
        t_cad_s: np.ndarray,
        *,
        q_min: float = 0.0,
        D_min_cm: float = 0.0,
        s_fade: float = 0.0,
        s_rise: float = 0.0,
        s_mode: str = "discrete",
        rise_random_start: bool = True,
        fade_random_start: bool = True,
        N_q: int = 500,
    ) -> np.ndarray:
        """Exact rate via full q-integral (thesis Eq. 39/62). No dominant-term approximation.

        Instead of the piecewise dominant-term formulas (A1–A7), integrates over all
        viewing angles from 0 to q_nr:

            R = f_Omega * theta_j^2 * R_int * integral_0^{q_nr} q * min(D_max(q), D_eff)^3 dq

        where D_max(q) is the piecewise maximum detectable distance and D_eff =
        min(D_i/D_Euc, 1) encodes both the cadence distance scale and the Euclidean
        horizon.  The dominant-term result is the leading term of this integral evaluated
        at the boundary angle (q_Euc or q_i); the full integral adds the "tail"
        contribution from angles beyond that boundary.

        Parameters
        ----------
        q_min, D_min_cm : float, optional
            Lower bounds on viewing-angle parameter and source distance.  Only the
            shell q ≥ q_min, D_min ≤ D ≤ D_eff(q) contributes to the integral.
            Default 0 / 0 = no filter.
        s_fade, s_rise, s_mode : float, float, str, optional
            Fading- and rise-rate filters — see `rate_log10`, `_joint_survival`
            and `_weighted_D_volume`.  The D-volume of each q-point carries the
            joint uniform-start survival weight (min of the fade and rise
            windows over the same start offset; a continuous weight — the
            dominant-term mode keeps the best-case hard boundaries instead).
        N_q : int
            Number of integration points along q.  500 gives <0.3% numerical error
            relative to the analytic limit for typical parameters.

        Notes
        -----
        The (N_q, *shape) integration tensors are materialized in q-chunks of
        at most `_FULL_INTEGRAL_CHUNK_ELEMS` float64 elements each.  Adjacent
        chunks share their boundary q-point, so the accumulated per-chunk
        trapezoids equal the full-grid trapezoid exactly (additivity of the
        trapezoid rule over contiguous subintervals); results can differ from
        the unchunked form only by float summation order, and single-chunk
        calls (small grids — every scalar/1-D caller at the default budget)
        are bit-identical.  This bounds peak memory to a few hundred MiB for
        arbitrarily large strategy grids — required under Pyodide's WASM32
        ~4 GiB address space; the residual O(grid) footprint of the
        grid-shaped arrays is irreducible.
        """

        N_exp   = np.asarray(N_exp,   dtype=float)
        t_cad_s = np.asarray(t_cad_s, dtype=float)
        shape   = np.broadcast(N_exp, t_cad_s).shape
        N_exp_b = np.broadcast_to(N_exp,   shape)
        t_cad_b = np.broadcast_to(t_cad_s, shape)

        t_exp  = self.t_exp_s(N_exp_b, t_cad_b)
        F_lim  = self.F_lim_Jy(t_exp)
        fO     = self.f_Omega(N_exp_b)

        D_Euc   = self.phys.D_euc_cm
        theta_j = self.phys.theta_j_rad
        R_int   = self.phys.R_int_yr
        q_dec   = self.derived.q_dec
        q_j     = float(self.derived.q_j)
        q_nr    = float(self.derived.q_nr)
        q_td    = q_dec - 1.0  # q̃_dec
        a_II    = float(self.pls.a_II(self.phys.p))
        a_III   = float(self.pls.a_III(self.phys.p))

        # Cadence distance scale: D̃_eff = min(D_i / D_Euc, 1).  Under
        # win_from_peak the cap is q-dependent (window measured from t_p) and
        # is computed per q-chunk inside the loop instead.
        D_tilde_dec  = self.D_dec(F_lim) / D_Euc   # shape
        if self.win_from_peak:
            D_tilde_eff = None
        else:
            D_i         = self.D_i(i_det, t_cad_b, F_lim)
            D_tilde_eff = np.minimum(D_i / D_Euc, 1.0) # shape, capped at 1

        # Trapezoidal integral: ∫_{q_min}^{q_nr} q * max(D̃_eff^3 − D̃_min^3, 0) dq
        # At q_min=0, D_min=0 the weight and the max() reduce to 1 and D̃_eff³,
        # so the integral matches the pre-filter form exactly.
        D_min_norm = float(D_min_cm) / D_Euc
        q_min_f    = float(q_min)

        # q grid: shape (N_q,).  The (n_chunk, *shape) tensors below are built
        # in memory-bounded q-chunks (see the docstring's memory note).
        # Adjacent chunks share their boundary q-point (each chunk starts at
        # the previous chunk's last point), so summing the per-chunk
        # trapezoids covers every q-interval exactly once — additivity of the
        # trapezoid rule over contiguous subintervals.
        q_vals    = np.linspace(0.0, q_nr, N_q + 1)[1:]  # skip q=0 to avoid q̃=−1
        ndim      = len(shape)
        grid_size = max(1, int(np.prod(shape, dtype=np.int64)))
        n_chunk   = max(2, int(_FULL_INTEGRAL_CHUNK_ELEMS) // grid_size)

        I     = np.zeros(shape, dtype=float)
        n_q   = int(q_vals.size)
        start = 0
        while start < n_q - 1:
            stop = min(start + n_chunk, n_q)          # ≥ start + 2
            q_c  = q_vals[start:stop]
            q_g  = q_c.reshape((-1,) + (1,) * ndim)   # (n_chunk, *shape)
            qt_g = np.maximum(q_g - 1.0, 0.0)         # q̃ = q − 1

            # Piecewise D̃_max(q):
            #   q < q_dec  → D̃_dec (on-axis, constant)
            #   q_dec≤q<q_j → D̃_dec * (q̃/q̃_dec)^{-a_II}   (Phase II)
            #   q≥q_j       → (D̃_dec/q̃_dec) * (q̃/q̃_dec)^{-a_III}  (Phase III)
            on_axis   = q_g <  q_dec
            phase_II  = (q_g >= q_dec) & (q_g < q_j)
            phase_III = q_g >= q_j

            # Guard against zero denominators in masked regions
            safe_II  = np.where(phase_II,  np.maximum(qt_g, 1e-30), 1.0)
            safe_III = np.where(phase_III, np.maximum(qt_g, 1e-30), 1.0)

            D_max_II  = D_tilde_dec * (safe_II  / q_td) ** (-a_II)
            D_max_III = (D_tilde_dec / q_td) * (safe_III / q_td) ** (-a_III)

            D_tilde_max = np.where(on_axis,  D_tilde_dec,
                          np.where(phase_II, D_max_II,
                                             D_max_III))

            # Effective detectable distance: min(D̃_max, D̃_eff).  Under
            # win_from_peak D̃_eff is the per-q window distance (one extra
            # (n_chunk, *shape) slab inside the chunk memory budget).
            if self.win_from_peak:
                D_tilde_eff_g = np.minimum(
                    self._D_eff_window_cm(q_g, i_det, t_cad_b, F_lim) / D_Euc, 1.0
                )
            else:
                D_tilde_eff_g = D_tilde_eff
            D_eff = np.minimum(D_tilde_max, D_tilde_eff_g)  # (n_chunk, *shape)

            # Fade + rise filters: joint uniform-start survival weight folded
            # into the closed-form D-volume of each q-point (see
            # `_weighted_D_volume`).  With s_rise = 0 this is exactly
            # max(D_eff³ − D̃_min³, 0)·P_fade, and with s_fade = 0 too it is
            # bit-identical to the pre-filter integrand.
            V_w = self._weighted_D_volume(
                q_g, D_tilde_max, D_eff, D_min_norm,
                i_det, t_cad_b, s_fade, s_rise, s_mode,
                rise_random_start=rise_random_start,
                fade_random_start=fade_random_start,
            )
            q_keep = q_g >= q_min_f
            integrand = np.where(q_keep, q_g * V_w, 0.0)   # (n_chunk, *shape)
            I += np.trapezoid(integrand, q_c, axis=0)      # shape
            start = stop - 1

        R = fO * (theta_j ** 2) * R_int * I
        R = np.where(np.isfinite(t_exp), R, np.nan)
        return _safe_log10(R)

    # ---------- Median q and D for detected GRBs ----------

    def compute_medians_analytic(
        self,
        i_det: int,
        N_exp: np.ndarray,
        t_cad_s: np.ndarray,
        *,
        q_min: float = 0.0,
        D_min_cm: float = 0.0,
        s_fade: float = 0.0,
        s_rise: float = 0.0,
        s_mode: str = "discrete",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Median q and D for the dominant-term (normal) mode.

        In the dominant-term approximation, D_eff is constant within the
        integration range [q_min, q_max] for each regime, giving analytic medians:
            q_med  = sqrt((q_max² + q_min²) / 2)
            D_med  = ((D_eff_const³ + D_min³) / 2)^(1/3)
        At q_min=0, D_min=0 these reduce to the standard q_max/√2 and D_eff·2^(−1/3).
        When the filter excludes all weight in a regime (q_min ≥ q_max or
        D_min ≥ D_eff_const), both medians are NaN.

        The fading-rate filter (s_fade, s_mode — see `rate_log10`) is applied by
        substituting each regime's q_max with min(q_max, q_s,phase) — the
        best-case hard cap; the rise-rate filter (s_rise) likewise substitutes
        the raised-limit boundaries qE_r / q_ri into q_max and D_dec·η^{−1/2}
        into A7's D_eff (per-regime descriptive form — no on-axis max-term
        here).  The full-integral medians use the uniform-start survival
        weights instead.

        Returns
        -------
        q_med     : ndarray, same shape as broadcast(N_exp, t_cad_s)
        D_med_cm  : ndarray, median detectable distance [cm]
        """
        N_exp   = np.asarray(N_exp,   dtype=float)
        t_cad_s = np.asarray(t_cad_s, dtype=float)
        shape   = np.broadcast(N_exp, t_cad_s).shape
        N_exp_b = np.broadcast_to(N_exp,   shape)
        t_cad_b = np.broadcast_to(t_cad_s, shape)

        t_exp  = self.t_exp_s(N_exp_b, t_cad_b)
        F_lim  = self.F_lim_Jy(t_exp)

        q_E    = self.q_Euc(F_lim)
        qi     = self.q_i(i_det, t_cad_b)
        D_i    = self.D_i(i_det, t_cad_b, F_lim)
        D_dec  = self.D_dec(F_lim)

        q_nr   = float(self.derived.q_nr)
        q_dec  = float(self.derived.q_dec)
        D_Euc  = self.phys.D_euc_cm

        masks = self.region_masks(i_det, N_exp_b, t_cad_b, include_unphysical=False)

        # Fading-rate q-caps (per phase). +∞ when bypassed → minima below are no-ops.
        q_s_II, q_s_III = self._q_s_fading_caps(i_det, t_cad_b, s_fade, s_mode)

        # Rise-rate boundaries at the raised limit (see `rate_log10`).
        if float(s_rise) > 0.0:
            eta, ise = self._rise_eta(t_cad_b, s_rise)
            with np.errstate(over="ignore", invalid="ignore"):
                qE_r = self.q_Euc(eta * F_lim)
                q_ri = self.q_Euc(eta * F_lim * (D_i / D_Euc) ** 2)
            D_dec_r = D_dec * ise
        else:
            qE_r = q_ri = np.inf
            D_dec_r = np.inf

        # Effective q_max per regime (geometric bound ∧ phase-mapped fading cap
        # ∧ rise boundary; r1/r4 split the shared q_nr cap on the rise side).
        q_nr_III_eff_1 = np.minimum(np.minimum(q_nr, q_s_III), qE_r)
        q_nr_III_eff_4 = np.minimum(np.minimum(q_nr, q_s_III), q_ri)
        qE_III_eff     = np.minimum(np.minimum(q_E,  q_s_III), qE_r)
        qE_II_eff      = np.minimum(np.minimum(q_E,  q_s_II),  qE_r)
        qi_III_eff     = np.minimum(np.minimum(qi,   q_s_III), q_ri)
        qi_II_eff      = np.minimum(np.minimum(qi,   q_s_II),  q_ri)
        q_dec_eff      = np.minimum(q_dec, q_s_II)

        # q_max: upper integration limit per regime
        q_max = np.full(shape, np.nan, dtype=float)
        q_max = np.where(masks["A1"], q_nr_III_eff_1, q_max)
        q_max = np.where(masks["A2"], qE_III_eff,     q_max)
        q_max = np.where(masks["A3"], qE_II_eff,      q_max)
        q_max = np.where(masks["A4"], q_nr_III_eff_4, q_max)
        q_max = np.where(masks["A5"], qi_III_eff,     q_max)
        q_max = np.where(masks["A6"], qi_II_eff,      q_max)
        q_max = np.where(masks["A7"], q_dec_eff,      q_max)

        # D_eff_const: constant effective distance per regime
        D_eff_const = np.full(shape, np.nan, dtype=float)
        D_eff_const = np.where(masks["A1"] | masks["A2"] | masks["A3"], D_Euc,  D_eff_const)
        D_eff_const = np.where(masks["A4"] | masks["A5"] | masks["A6"], D_i,   D_eff_const)
        D_eff_const = np.where(masks["A7"], np.minimum(np.minimum(D_dec, D_i), D_dec_r), D_eff_const)

        # p(q) ∝ q on [q_min, q_max] → median q² = (q_max² + q_min²)/2.
        # p(D) ∝ D² on [D_min, D_eff_const] → median D³ = (D_eff_const³ + D_min³)/2.
        # Cells where the filter excludes all weight (q_max² ≤ q_min² or
        # D_eff_const³ ≤ D_min³) → NaN for both medians.
        q_min_sq = float(q_min) ** 2
        D_min_cm_f = float(D_min_cm)
        D_min_cubed = D_min_cm_f ** 3

        q_max_sq         = q_max ** 2
        D_eff_const_cubed = D_eff_const ** 3

        zero_weight = (q_max_sq <= q_min_sq) | (D_eff_const_cubed <= D_min_cubed)
        q_med    = np.where(zero_weight, np.nan,
                            np.sqrt((q_max_sq + q_min_sq) / 2.0))
        D_med_cm = np.where(zero_weight, np.nan,
                            ((D_eff_const_cubed + D_min_cubed) / 2.0) ** (1.0 / 3.0))

        invalid  = ~np.isfinite(t_exp) | (t_exp <= 0)
        return np.where(invalid, np.nan, q_med), np.where(invalid, np.nan, D_med_cm)

    def compute_medians_numerical(
        self,
        i_det: int,
        N_exp: np.ndarray,
        t_cad_s: np.ndarray,
        N_q: int = 200,
        *,
        q_min: float = 0.0,
        D_min_cm: float = 0.0,
        s_fade: float = 0.0,
        s_rise: float = 0.0,
        s_mode: str = "discrete",
        rise_random_start: bool = True,
        fade_random_start: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Median q and D for the full-integral mode, computed numerically.

        Uses the same D_eff(q) = min(D_max(q), D_tilde_eff) profile as
        rate_log10_full_integral.  Integrand for marginal q (filtered):
        w(q) = 1{q ≥ q_min} · q · V_w(q),
        with V_w the joint fade+rise weighted D-volume of `_weighted_D_volume`
        (= max(D_eff³ − D̃_min³, 0)·P_fade when the rise cut is off).  Marginal
        D (filtered): p(D) ∝ D² · ∫ q · 1{D_eff(q) ≥ D} · 1{q ≥ q_min} ·
        w_joint(q, D̃) dq restricted to D ≥ D_min, with w_joint the pointwise
        joint survival of `_joint_survival`.

        Returns
        -------
        q_med     : ndarray
        D_med_cm  : ndarray [cm]
        """
        N_exp   = np.asarray(N_exp,   dtype=float)
        t_cad_s = np.asarray(t_cad_s, dtype=float)
        shape   = np.broadcast(N_exp, t_cad_s).shape
        N_exp_b = np.broadcast_to(N_exp,   shape)
        t_cad_b = np.broadcast_to(t_cad_s, shape)

        t_exp  = self.t_exp_s(N_exp_b, t_cad_b)
        F_lim  = self.F_lim_Jy(t_exp)

        D_Euc   = self.phys.D_euc_cm
        q_dec   = self.derived.q_dec
        q_j     = float(self.derived.q_j)
        q_nr    = float(self.derived.q_nr)
        q_td    = q_dec - 1.0
        a_II    = float(self.pls.a_II(self.phys.p))
        a_III   = float(self.pls.a_III(self.phys.p))

        D_tilde_dec = self.D_dec(F_lim) / D_Euc

        # q grid — same construction as rate_log10_full_integral
        q_vals = np.linspace(0.0, q_nr, N_q + 1)[1:]
        ndim   = len(shape)
        q_g    = q_vals.reshape((-1,) + (1,) * ndim)
        qt_g   = np.maximum(q_g - 1.0, 0.0)

        # Cadence distance cap: q-dependent under win_from_peak (window
        # measured from t_p), constant D_i otherwise.
        if self.win_from_peak:
            D_tilde_eff = np.minimum(
                self._D_eff_window_cm(q_g, i_det, t_cad_b, F_lim) / D_Euc, 1.0
            )
        else:
            D_i_arr     = self.D_i(i_det, t_cad_b, F_lim)
            D_tilde_eff = np.minimum(D_i_arr / D_Euc, 1.0)

        on_axis   = q_g <  q_dec
        phase_II  = (q_g >= q_dec) & (q_g < q_j)
        phase_III = q_g >= q_j

        safe_II  = np.where(phase_II,  np.maximum(qt_g, 1e-30), 1.0)
        safe_III = np.where(phase_III, np.maximum(qt_g, 1e-30), 1.0)

        D_max_II  = D_tilde_dec * (safe_II  / q_td) ** (-a_II)
        D_max_III = (D_tilde_dec / q_td) * (safe_III / q_td) ** (-a_III)

        D_tilde_max = np.where(on_axis,  D_tilde_dec,
                      np.where(phase_II, D_max_II,
                                         D_max_III))

        D_eff = np.minimum(D_tilde_max, D_tilde_eff)   # (N_q, *shape), normalized

        # Filter: only q ≥ q_min and D ≥ D_min count.  At q_min=0, D_min=0
        # the integrand below reduces to the original q · D_eff³.
        q_min_f = float(q_min)
        D_min_norm = float(D_min_cm) / D_Euc

        # Fade + rise filters: joint uniform-start survival weights — continuous,
        # not hard cuts (see `_weighted_D_volume` / `_joint_survival`; both
        # reduce to the fade-only forms at s_rise = 0, and to 1 when bypassed).
        q_keep_mask = q_g >= q_min_f                                         # (N_q, 1, ...)

        # ── Median q ──────────────────────────────────────────────────────────
        # Filtered marginal: w(q) = 1{q ≥ q_min} · q · V_w(q).
        V_w = self._weighted_D_volume(
            q_g, D_tilde_max, D_eff, D_min_norm,
            i_det, t_cad_b, s_fade, s_rise, s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )
        w_q        = np.where(q_keep_mask, q_g * V_w, 0.0)                   # (N_q, *shape)
        cumsum_q   = np.cumsum(w_q, axis=0)
        total_q    = cumsum_q[-1:] + 1e-300
        idx_q      = np.argmax(cumsum_q / total_q >= 0.5, axis=0)            # (*shape)
        has_weight = (cumsum_q[-1] > 1e-290)
        q_med      = np.where(has_weight, q_vals[idx_q], np.nan)

        # ── Median D ──────────────────────────────────────────────────────────
        # p(D) ∝ D² · ∫_{q_min}^{q_nr} q · 1{D_eff(q) ≥ D} · w_joint(q, D̃) dq,
        # restricted to D ≥ D_min.  Without the filters and on the contiguous
        # slope of D_eff, the q-integral reduces to (Q(D)² − q_min²)/2 (the
        # analytic form).  With a filter active the integrand carries the
        # continuous joint survival weight, so we integrate numerically.
        D_eff_max  = np.maximum(D_eff.max(axis=0), 1e-30)  # (*shape)
        D_eff_norm = D_eff / D_eff_max[np.newaxis, ...]    # (N_q, *shape), in [0, 1]

        N_D    = 100
        d_grid = np.linspace(1.0 / N_D, 1.0, N_D)     # normalized D levels, (N_D,)

        # Q_sq_eff[j] = 2·∫ q · 1{D_eff_norm ≥ d_j} · 1{q ≥ q_min} · w_joint dq.
        # (twice the integral so the semantic matches the original Q² − q_min²).
        # Trapezoid coefficients for the uniform q-grid: np.trapezoid weights the
        # endpoints by 0.5·Δq and the interior by Δq, so applying them once turns
        # each masked integral into a plain weighted sum over q-points.
        dq       = (q_vals[1] - q_vals[0]) if q_vals.size > 1 else 1.0
        trap_c   = np.full(q_vals.shape, dq, dtype=float)
        trap_c[0]  *= 0.5
        trap_c[-1] *= 0.5
        trap_c   = trap_c.reshape((-1,) + (1,) * ndim)                        # (N_q, 1, …)

        if float(s_rise) <= 0.0:
            # Rise cut off ⇒ w_joint = _fading_survival is independent of the
            # distance level d.  Then Q_sq_eff over all N_D levels is a survival
            # function: bin each q-point by the highest level it clears
            # (searchsorted 'right' matches the ">= d_grid[j]" test exactly),
            # scatter-add its trapezoid-weighted contribution, and reverse-cumsum.
            # This replaces the N_D-length loop's O(N_D·N_q·grid) with one
            # O((N_D+N_q)·grid) pass; the result equals the loop's up to float
            # summation order (~1e-15), well below the d-grid discretization.
            w = self._joint_survival(                # (N_q, *shape); == _fading_survival
                q_g, D_tilde_max, D_tilde_max,
                i_det, t_cad_b, s_fade, s_rise, s_mode,
                rise_random_start=rise_random_start,
                fade_random_start=fade_random_start,
            )
            base   = (trap_c * q_g * q_keep_mask) * w                         # (N_q, *shape)
            n_qm   = base.shape[0]
            M      = int(np.prod(shape, dtype=np.int64)) if shape else 1
            base_f = np.ascontiguousarray(base).reshape(n_qm, M)
            De_f   = np.ascontiguousarray(D_eff_norm).reshape(n_qm, M)
            # bins[i] = highest level index cleared by point i (−1 ⇒ clears none)
            bins   = np.searchsorted(d_grid, De_f, side="right") - 1          # (N_q, M)
            cols   = np.broadcast_to(np.arange(M), bins.shape)
            keep   = bins >= 0
            lin    = bins[keep].astype(np.int64) * M + cols[keep]
            Hf     = np.bincount(lin, weights=base_f[keep], minlength=N_D * M)
            H      = Hf.reshape((N_D,) + shape)
            # Q_sq_eff[j] = 2·Σ_{k ≥ j} H[k]  (reverse cumulative sum over levels)
            Q_sq_eff = 2.0 * np.cumsum(H[::-1], axis=0)[::-1]
        else:
            # Rise cut active ⇒ w_joint depends on d, so the survival trick does
            # not apply.  Keep the per-level loop but hoist every d-independent
            # quantity out of it (only t_lim and the final clip depend on d).
            t_p_eff, t_fs_sel, k_sel, ise = self._rise_fade_windows(
                q_g, i_det, t_cad_b, s_fade, s_rise, s_mode
            )
            if not fade_random_start:
                t_fs_sel = np.where(t_fs_sel >= t_p_eff, np.inf, -np.inf)
            D0       = np.asarray(D_tilde_max, dtype=float) * ise             # (N_q, *shape)
            Q_sq_eff = np.empty((N_D,) + shape, dtype=float)
            for j, d in enumerate(d_grid):
                D_safe = np.maximum(d * D_eff_max[np.newaxis, ...], 1e-300)
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    t_lim = t_p_eff * (D0 / D_safe) ** k_sel
                    if not rise_random_start:
                        t_lim = np.where(t_lim >= t_p_eff, np.inf, -np.inf)
                    w_joint = np.clip(
                        (np.minimum(t_fs_sel, t_lim) - t_p_eff) / t_cad_b, 0.0, 1.0
                    )
                above_and_keep = (D_eff_norm >= d) & q_keep_mask              # (N_q, *shape)
                integrand_q    = np.where(above_and_keep, q_g * w_joint, 0.0) # (N_q, *shape)
                Q_sq_eff[j]    = 2.0 * np.trapezoid(integrand_q, q_vals, axis=0)

        d_g      = d_grid.reshape((-1,) + (1,) * ndim)
        # Distance-floor mask: zero out levels below D_min_cm.
        D_phys = d_g * D_eff_max[np.newaxis, ...] * D_Euc       # (N_D, *shape) in cm
        D_keep = D_phys >= float(D_min_cm)
        w_D      = np.where(D_keep, d_g ** 2 * Q_sq_eff, 0.0)  # (N_D, *shape)
        cumsum_D = np.cumsum(w_D, axis=0)
        total_D  = cumsum_D[-1:] + 1e-300
        idx_D    = np.argmax(cumsum_D / total_D >= 0.5, axis=0)    # (*shape)
        d_med    = d_grid[idx_D]
        D_has_weight = (cumsum_D[-1] > 1e-290)
        D_med_cm = np.where(has_weight & D_has_weight, d_med * D_eff_max * D_Euc, np.nan)
        # If the q-marginal had no weight, q_med must also be NaN regardless of D.
        q_med    = np.where(has_weight & D_has_weight, q_med, np.nan)

        invalid  = ~np.isfinite(t_exp) | (t_exp <= 0)
        return np.where(invalid, np.nan, q_med), np.where(invalid, np.nan, D_med_cm)

    # ---------- Differential rates dR/dq, dR/dD (full-integral mode) ----------

    def _D_eff_q_profile_scalar(
        self,
        i_det: int,
        N_exp: float,
        t_cad_s: float,
        N_q: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Helper for dR/dq, dR/dD: scalar-input D_eff(q) profile.

        Returns
        -------
        q_vals     : (N_q,)            — q grid on (0, q_nr].
        D_eff_norm : (N_q,)            — D̃_eff(q) = D_eff(q) / D_Euc.
        D_tilde_max: (N_q,)            — flux-limited D̃_max(q) profile (for the
                                         rise-filter weights).
        prefactor  : float             — f_Omega · θ_j² · R_int.
        t_exp      : float (or NaN)    — exposure time; NaN if strategy is unphysical.
        """
        # Several model methods (F_lim_Jy, D_i, D_dec) use boolean array
        # indexing internally — wrap the scalar inputs in 1-element arrays.
        N_arr  = np.array([float(N_exp)])
        t_arr  = np.array([float(t_cad_s)])

        t_exp_arr = self.t_exp_s(N_arr, t_arr)
        t_exp     = float(t_exp_arr[0])
        if not np.isfinite(t_exp) or t_exp <= 0:
            return (
                np.linspace(0.0, float(self.derived.q_nr), N_q + 1)[1:],
                np.full(N_q, np.nan),
                np.full(N_q, np.nan),
                float("nan"),
                float("nan"),
            )

        F_lim_arr = self.F_lim_Jy(t_exp_arr)
        fO_arr    = self.f_Omega(N_arr)
        fO        = float(fO_arr[0]) if hasattr(fO_arr, "__len__") else float(fO_arr)

        D_Euc   = self.phys.D_euc_cm
        theta_j = self.phys.theta_j_rad
        R_int   = self.phys.R_int_yr
        q_dec   = self.derived.q_dec
        q_j     = float(self.derived.q_j)
        q_nr    = float(self.derived.q_nr)
        q_td    = q_dec - 1.0
        a_II    = float(self.pls.a_II(self.phys.p))
        a_III   = float(self.pls.a_III(self.phys.p))

        D_dec_arr   = self.D_dec(F_lim_arr)
        D_tilde_dec = float(D_dec_arr[0]) / D_Euc

        q_vals = np.linspace(0.0, q_nr, N_q + 1)[1:]
        qt_g   = np.maximum(q_vals - 1.0, 0.0)

        # Cadence distance cap: q-dependent (N_q,) profile under win_from_peak,
        # constant D_i otherwise.
        if self.win_from_peak:
            D_tilde_eff = np.minimum(
                self._D_eff_window_cm(q_vals, i_det, t_arr, F_lim_arr) / D_Euc, 1.0
            )
        else:
            D_i_arr     = self.D_i(i_det, t_arr, F_lim_arr)
            D_tilde_eff = min(float(D_i_arr[0]) / D_Euc, 1.0)

        on_axis   = q_vals <  q_dec
        phase_II  = (q_vals >= q_dec) & (q_vals < q_j)
        phase_III = q_vals >= q_j

        safe_II  = np.where(phase_II,  np.maximum(qt_g, 1e-30), 1.0)
        safe_III = np.where(phase_III, np.maximum(qt_g, 1e-30), 1.0)

        D_max_II  = D_tilde_dec * (safe_II  / q_td) ** (-a_II)
        D_max_III = (D_tilde_dec / q_td) * (safe_III / q_td) ** (-a_III)

        D_tilde_max = np.where(on_axis,  D_tilde_dec,
                      np.where(phase_II, D_max_II,
                                         D_max_III))
        D_eff_norm = np.minimum(D_tilde_max, D_tilde_eff)

        prefactor = fO * (theta_j ** 2) * R_int
        return q_vals, D_eff_norm, D_tilde_max, prefactor, t_exp

    def dR_dq_full_integral(
        self,
        i_det: int,
        N_exp: float,
        t_cad_s: float,
        *,
        q_min: float = 0.0,
        D_min_cm: float = 0.0,
        s_fade: float = 0.0,
        s_rise: float = 0.0,
        s_mode: str = "discrete",
        rise_random_start: bool = True,
        fade_random_start: bool = True,
        N_q: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Differential rate dR/dq for the full-integral (exact) mode.

        Returns
        -------
        q_vals : (N_q,)  — q grid on (0, q_nr].
        dR_dq  : (N_q,)  — yr⁻¹ per unit q.  All-NaN if the strategy is
                           t_exp-invalid (e.g. f_live · t_cad / N_exp ≤ t_overhead).

        Math:
            dR/dq = f_Omega · θ_j² · R_int · 1{q ≥ q_min} · q · V_w(q),
        with V_w the joint fade+rise weighted D-volume (see
        `_weighted_D_volume`; = max(D̃_eff³ − D̃_min³, 0)·P_fade at s_rise = 0).
        """
        q_vals, D_eff_norm, D_tilde_max, prefactor, t_exp = self._D_eff_q_profile_scalar(
            i_det, N_exp, t_cad_s, N_q
        )
        if not np.isfinite(t_exp):
            return q_vals, np.full(N_q, np.nan)

        D_min_norm = float(D_min_cm) / self.phys.D_euc_cm
        keep       = q_vals >= float(q_min)

        # Joint fade+rise weighted D-volume (scalar t_cad → (N_q,) weight).
        V_w = self._weighted_D_volume(
            q_vals, D_tilde_max, D_eff_norm, D_min_norm,
            i_det, np.array([float(t_cad_s)]), s_fade, s_rise, s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )

        dR_dq = np.where(keep, prefactor * q_vals * V_w, 0.0)
        return q_vals, dR_dq

    def dR_dD_full_integral(
        self,
        i_det: int,
        N_exp: float,
        t_cad_s: float,
        *,
        q_min: float = 0.0,
        D_min_cm: float = 0.0,
        s_fade: float = 0.0,
        s_rise: float = 0.0,
        s_mode: str = "discrete",
        rise_random_start: bool = True,
        fade_random_start: bool = True,
        N_q: int = 500,
        N_D: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Differential rate dR/dD for the full-integral (exact) mode.

        Returns
        -------
        D_grid_cm : (N_D,) — linear grid on [0, D_Euc] in cm.
        dR_dD     : (N_D,) — yr⁻¹ per cm.  All-NaN if the strategy is t_exp-invalid.

        Math (skeptic-derived; the inner q-integral contributes the 1/2):
            dR/dD = (3 · f_Omega · θ_j² · R_int / D_Euc) · D̃² ·
                    ∫_{q_min}^{q_nr} q · 1{D_eff(q) ≥ D̃} · w_joint(q, D̃) dq,
        with w_joint the pointwise joint fade+rise survival of
        `_joint_survival` (≡ 1 when both cuts are bypassed; the rise part makes
        it genuinely D̃-dependent).  Without the filters and along the
        contiguous slope of D_eff(q), the inner integral reduces to
        (Q(D̃)² − q_min²)/2 with Q = max{q : D_eff(q) ≥ D̃}.
        Values for D < D_min_cm are zeroed out.
        """
        D_Euc = self.phys.D_euc_cm
        D_grid_cm = np.linspace(0.0, D_Euc, N_D)

        q_vals, D_eff_norm, D_tilde_max, prefactor, t_exp = self._D_eff_q_profile_scalar(
            i_det, N_exp, t_cad_s, N_q
        )
        if not np.isfinite(t_exp):
            return D_grid_cm, np.full(N_D, np.nan)

        d_grid = D_grid_cm / D_Euc  # (N_D,) in [0, 1]

        # Joint fade+rise survival on the (N_D, N_q) grid — d_grid IS D̃ here.
        w_joint = self._joint_survival(
            q_vals[np.newaxis, :], d_grid[:, np.newaxis],
            D_tilde_max[np.newaxis, :],
            i_det, np.array([float(t_cad_s)]), s_fade, s_rise, s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )                                                                # (N_D, N_q)
        q_keep_f = (q_vals >= float(q_min)).astype(float)                # (N_q,)

        # q-integral per d.  Vectorised over (N_D, N_q) — D_eff_norm has no shape
        # beyond N_q for this scalar-strategy entry point, so memory is modest.
        above    = D_eff_norm[np.newaxis, :] >= d_grid[:, np.newaxis]    # (N_D, N_q)
        integrand_q = q_vals[np.newaxis, :] * above * w_joint * q_keep_f[np.newaxis, :]
        q_integral = np.trapezoid(integrand_q, q_vals, axis=1)           # (N_D,)

        dR_dD = (3.0 * prefactor / D_Euc) * (d_grid ** 2) * q_integral
        dR_dD = np.where(D_grid_cm >= float(D_min_cm), dR_dD, 0.0)
        return D_grid_cm, dR_dD

    def compute_medians(
        self,
        i_det: int,
        N_exp: np.ndarray,
        t_cad_s: np.ndarray,
        *,
        full_integral: bool = False,
        N_q: int = 200,
        q_min: float = 0.0,
        D_min_cm: float = 0.0,
        s_fade: float = 0.0,
        s_rise: float = 0.0,
        s_mode: str = "discrete",
        rise_random_start: bool = True,
        fade_random_start: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_med, D_med_cm) using analytic (normal) or numerical (full-integral) formula.

        Under `win_from_peak` the analytic per-regime constants no longer
        describe the rate (D_eff is q-dependent), so the numerical path is
        used regardless of `full_integral`.  The `*_random_start` flags apply
        only on the numerical path (the analytic path is the hard-boundary
        dominant-term treatment already).
        """
        if full_integral or self.win_from_peak:
            return self.compute_medians_numerical(
                i_det, N_exp, t_cad_s, N_q,
                q_min=q_min, D_min_cm=D_min_cm,
                s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
                rise_random_start=rise_random_start,
                fade_random_start=fade_random_start,
            )
        return self.compute_medians_analytic(
            i_det, N_exp, t_cad_s,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
        )

    # ---------- Analytic optimal strategy (from  ) ----------
    def analytic_optimum(self, i_det: int) -> Dict[str, float]:
        """Analytic optimum from the   derivation.

        This reproduces the   definitions:
        - N_exp,opt = Omega_srv,max / Omega_exp
        - t_cad,opt: piecewise formula supplied by the user
        - t_exp,opt from the exposure-time relation

        Returns
        -------
        dict
            Keys: N_exp_opt, t_cad_opt_s, t_exp_opt_s, log10_R_det_opt, R_det_opt
        """

        i_det = int(i_det)
        if i_det < 1:
            raise ValueError("i_det must be >= 1")

        # N_exp,opt
        N_opt = N_exp_max(self.instrument)

        t_dec = self.derived.t_dec_s
        p = self.phys.p

        theta_j = self.phys.theta_j_rad
        Gamma0 = self.phys.gamma0

        F_dec = self.derived.F_dec_Jy
        A = self.instrument.F_lim_ref_Jy  # F_ref in the limiting-flux model
        t_ref = self.instrument.t_exp_ref_s
        f_live = self.instrument.f_live

        # Candidate expressions (as in the   code)
        exprA = (
            (t_dec / i_det) ** (2.0 * p)
            * (Gamma0 * theta_j) ** (4.0 * (p + 3.0) / 3.0)
            * (A / F_dec) ** (-2.0)
            * (f_live / (t_ref * N_opt))
        ) ** (1.0 / (2.0 * p - 1.0))

        exprB = (
            (t_dec / i_det) ** (3.0 * (p - 1.0) / 2.0)
            * (A / F_dec) ** (-2.0)
            * (f_live / (t_ref * N_opt))
        ) ** (2.0 / (3.0 * p - 5.0))

        # Piecewise choice (interpreting the middle condition as t_dec/i < exprB < exprA)
        if exprA < exprB:
            t_cad_opt = float(exprA)
        elif (t_dec / i_det) < exprB < exprA:
            t_cad_opt = float(exprB)
        else:
            t_cad_opt = float(t_dec / i_det)

        strat_opt = SurveyStrategy(N_exp=N_opt, t_cad_s=t_cad_opt)
        t_exp_opt = exposure_time_s(strat_opt, self.instrument)

        logR_opt = float(self.rate_log10(i_det, np.array([N_opt]), np.array([t_cad_opt]))[0])
        R_opt = float(10.0**logR_opt)

        return {
            "N_exp_opt": float(N_opt),
            "t_cad_opt_s": float(t_cad_opt),
            "t_exp_opt_s": float(t_exp_opt),
            "log10_R_det_opt": logR_opt,
            "R_det_opt_yr": R_opt,
        }

    def grid_search_optimum(
        self,
        i_det: int,
        N_exp_grid: np.ndarray,
        t_cad_grid_s: np.ndarray,
    ) -> Dict[str, float]:
        """Brute-force optimum on a rectangular grid.

        Useful as a sanity check against the analytic optimum and as a tool
        when the model is extended to cases where no closed-form optimum is
        known.
        """

        logR = self.rate_log10(i_det, N_exp_grid[:, None], t_cad_grid_s[None, :])

        if np.all(np.isnan(logR)):
            raise RuntimeError("All strategies on the provided grid are unphysical (A0 is empty).")

        idx = np.nanargmax(logR)
        iN, it = np.unravel_index(idx, logR.shape)

        N_best = float(N_exp_grid[iN])
        t_cad_best = float(t_cad_grid_s[it])
        logR_best = float(logR[iN, it])

        t_exp_best = float(self.t_exp_s(np.array([N_best]), np.array([t_cad_best]))[0])

        return {
            "N_exp_opt": N_best,
            "t_cad_opt_s": t_cad_best,
            "t_exp_opt_s": t_exp_best,
            "log10_R_det_opt": logR_best,
            "R_det_opt_yr": float(10.0**logR_best),
        }
