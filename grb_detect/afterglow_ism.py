"""ISM (constant density) afterglow scalings.

This module implements derived scales that enter the analytic rate expressions.

Notation follows the Bachelor's project:
- t_dec: deceleration time
- t_j: jet break time
- q_dec: boundary between on-axis and off-axis onset (q_dec-1 = (Gamma_0 theta_j)^{-1})
- q_j = 2: viewing-angle parameter at which theta_obs = 2 theta_j
- q_nr: viewing-angle parameter at which the flow becomes Newtonian (approx)

Assumptions:
- top-hat jet,
- Blandford–McKee ISM dynamics up to t_j,
- Euclidean approximation for the rate calculation (optional also for flux/time normalisation).
"""


from __future__ import annotations

import numpy as np

from .params import AfterglowPhysicalParams, MicrophysicsParams


def t_dec_s(phys: AfterglowPhysicalParams, micro: MicrophysicsParams) -> float:
    """Deceleration time t_dec in seconds.

    Tier-1 expression (see the Bachelor's project around Eq. (30)):

        t_dec = 27 * zeta * E_52^{1/3} * n_0^{-1/3} * Gamma_{2.5}^{-8/3}  [s]

    where zeta = (1+z)/3.

    In the Euclidean approximation used for the analytic rate calculation, the
    baseline effectively takes z -> 0 (so zeta = 1/3).
    """

    E52 = phys.E_kiso_erg / 1e52
    n0 = phys.n0_cm3
    Gamma_2p5 = phys.gamma0 / 10**2.5

    zeta = (1.0 + micro.z) / 3.0 if micro.include_redshift_factors else 1.0 / 3.0

    return 27.0 * zeta * (E52 ** (1.0 / 3.0)) * (n0 ** (-1.0 / 3.0)) * (Gamma_2p5 ** (-8.0 / 3.0))


def t_j_s(phys: AfterglowPhysicalParams, t_dec_s_: float) -> float:
    """Jet break time (Tier 1): t_j = t_dec * (Gamma_0 theta_j)^{8/3}."""

    return t_dec_s_ * (phys.gamma0 * phys.theta_j_rad) ** (8.0 / 3.0)


def q_dec(phys: AfterglowPhysicalParams) -> float:
    """q_dec = 1 + (Gamma_0 theta_j)^{-1}."""

    return 1.0 + (phys.gamma0 * phys.theta_j_rad) ** (-1.0)


def q_j() -> float:
    """By definition q_j = 2."""

    return 2.0


def q_nr(phys: AfterglowPhysicalParams) -> float:
    """Approximate Newtonian boundary used in Tier 1: q_nr = sqrt(2)/theta_j."""

    return np.sqrt(2.0) / phys.theta_j_rad
