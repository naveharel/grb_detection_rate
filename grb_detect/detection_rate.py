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
from .params import AfterglowPhysicalParams, MicrophysicsParams, SurveyInstrumentParams, SurveyStrategy
from .pls import PLSG, PLSModel
from .survey import N_exp_max, exposure_time_s, is_strategy_physical, limiting_flux_Jy, sky_fraction


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
    """Implements the   piecewise rate surface."""

    def __init__(
        self,
        phys: AfterglowPhysicalParams,
        instrument: SurveyInstrumentParams,
        micro: MicrophysicsParams | None = None,
        pls: PLSModel | None = None,
    ):
        self.phys = phys
        self.instrument = instrument
        self.micro = micro if micro is not None else MicrophysicsParams()
        self.pls = pls if pls is not None else PLSG()

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
        """Exposure time per pointing (vectorized)."""

        return self.instrument.f_live * t_cad_s / N_exp - self.instrument.t_overhead_s

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

    def q_i(self, i_det: int, t_cad_s: np.ndarray) -> np.ndarray:
        """q_i(t_cad): angle for which t_p(q_i) = i * t_cad.

        Matches the   definition.
        """

        i_det = int(i_det)
        if i_det < 1:
            raise ValueError("i_det must be >= 1")

        t_req = i_det * t_cad_s
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

        t_req = i_det * t_cad_s
        before_tj = t_req < self.derived.t_j_s

        D_dec = self.D_dec(F_lim)

        out = np.empty_like(D_dec, dtype=float)
        # Phase II: D_i = D_dec * (qi_tilde/qd_tilde)^(-aII)
        out[before_tj] = D_dec[before_tj] * ((qi_tilde[before_tj] / qd_tilde) ** (-aII))

        # Phase III: D_i = D_dec * qd_tilde^{-1} (qi_tilde/qd_tilde)^(-aIII)
        out[~before_tj] = D_dec[~before_tj] * (qd_tilde ** (-1.0)) * (
            (qi_tilde[~before_tj] / qd_tilde) ** (-aIII)
        )

        return out

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

        # Tolerance to make boundary cases (like N_exp = Omega_srv,max / Omega_exp) robust
        eps = 1e-12

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

        # Tolerance for boundary robustness
        eps = 1e-12

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

    def rate_log10(
        self,
        i_det: int,
        N_exp: np.ndarray,
        t_cad_s: np.ndarray,
        *,
        return_components: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute log10 R_det for the given strategy grid.

        This mirrors the   definition:

            R_T(x,y) = {A1: R1, A2: R2, ..., A7: R7}

        where each Rk is itself a log10(...) expression.

        Returns
        -------
        log10_rate : ndarray
            log10 R_det [yr^-1]. NaN outside the physical region A0.
        components : dict, optional
            If return_components=True, also returns a dict with intermediate
            arrays (F_lim, q_Euc, q_i, D_dec, D_i, masks...).
        """

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

        # Region rates (linear)
        # Note: these are the *linear* rate expressions inside the log10.
        r1 = fO * R_int
        r2 = 0.5 * fO * (theta_j**2) * (qE**2) * R_int
        r3 = r2
        r4 = 0.5 * fO * (theta_j**2) * (self.derived.q_nr**2) * ((D_i / D_euc) ** 3) * R_int
        r5 = 0.5 * fO * (theta_j**2) * (qi**2) * ((D_i / D_euc) ** 3) * R_int
        r6 = r5
        r7 = 0.5 * fO * (theta_j**2) * (self.derived.q_dec**2) * ((D_dec / D_euc) ** 3) * R_int

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
