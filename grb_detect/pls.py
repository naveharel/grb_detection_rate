"""Power-Law Segment (PLS) models.

The Bachelor's project (Tier 1) focuses the detailed detection-rate
calculation on PLS G (nu_m < nu < nu_c) as the typical optical case.
Nevertheless, the Master's thesis is intended to incorporate PLS D, G, H.

This module provides a small abstraction layer:

- Each PLS model must provide the on-axis flux normalization F_dec (flux at
  t_dec, at D_euc), consistent with the analytic expressions.
- Each PLS model must provide the exponents controlling how the off-axis
  peak flux scales with viewing-angle parameter q~ in phase II and phase III.

For PLS G, the needed relations are explicitly given in Tier 1 and in the
Desmos prototype.

Notation:
- q := theta_obs / theta_j
- q_tilde := q - 1 (so q_tilde=0 is on-axis)

The off-axis peak flux ratio f_p(q_tilde) is taken as a power-law of q_tilde
in phase II and III, with exponents encoded by (a_II, a_III):

Phase II:  F_p / F_dec ~ (q_tilde / q_tilde_dec)^(-2 a_II)
Phase III: F_p / F_dec ~ q_tilde_dec^(-2) (q_tilde / q_tilde_dec)^(-2 a_III)

For PLS G:
- a_II = p-1
- a_III = p

This structure is designed so that adding PLS H later only requires changing
(a_II, a_III) and F_dec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .constants import DAY_S
from .params import AfterglowPhysicalParams, MicrophysicsParams


class PLSModel(Protocol):
    """Interface for a PLS model used by the rate calculator."""

    name: str

    def a_II(self, p: float) -> float:
        """Exponent a_II(p) controlling the phase-II scaling of F_p(q)."""

    def a_III(self, p: float) -> float:
        """Exponent a_III(p) controlling the phase-III scaling of F_p(q)."""

    def F_dec_Jy(
        self,
        phys: AfterglowPhysicalParams,
        micro: MicrophysicsParams,
        t_dec_s: float,
    ) -> float:
        """Flux density at t_dec, evaluated at distance D_euc.

        Returns
        -------
        float
            F_nu(t_dec, D_euc) in Jy.
        """


@dataclass(frozen=True)
class PLSG:
    """PLS G: nu_m < nu < nu_c (typical optical)"""

    name: str = "G"

    def a_II(self, p: float) -> float:  # noqa: N802
        return p - 1.0

    def a_III(self, p: float) -> float:  # noqa: N802
        return p

    def F_dec_Jy(
        self,
        phys: AfterglowPhysicalParams,
        micro: MicrophysicsParams,
        t_dec_s: float,
    ) -> float:
        """Eq. (24) of the Bachelor's project (Tier 1) evaluated at t_dec.

        IMPORTANT: The Tier-1 calculation adopts a Euclidean approximation
        and explicitly takes z -> 0 in the flux normalization. Therefore, the
        redshift factor (1+z)^(...) is set to 1 by default.

        The expression matches the Desmos implementation:

        F_dec = 0.461/1000 * (p-0.04) * exp(2.53 p)
                * (eps_e * (p-2)/(p-1))^(p-1)
                * eps_B^((p+1)/4) * n^(1/2) * E_52^((p+3)/4)
                * t_dec,days^(-3(p-1)/4) * D_28^(-2)
                * nu_14^(-(p-1)/2)

        Returns Jy.
        """

        p = phys.p

        if p <= 2.0:
            raise ValueError("p must be > 2 for the standard afterglow scalings.")

        # Dimensionless scalings
        E52 = phys.E_kiso_erg / 1e52
        n0 = phys.n0_cm3
        t_days = t_dec_s / DAY_S
        D28 = phys.D_euc_cm / 1e28
        nu14 = phys.nu_hz / 1e14

        # Microphysics
        eps_e = micro.epsilon_e
        eps_B = micro.epsilon_B
        # \bar{eps}_e = eps_e * (p-2)/(p-1)
        eps_e_bar = eps_e * (p - 2.0) / (p - 1.0)

        # In Tier 1 the z-factor is taken to 0 for consistency with Euclidean
        # treatment. Keep this as a switch for future work.
        z_factor = (
            (1.0 + micro.z) ** ((p + 3.0) / 4.0)
            if micro.include_redshift_factors
            else 1.0
        )

        F_mJy = (
            0.461
            * (p - 0.04)
            * np.exp(2.53 * p)
            * z_factor
            * (eps_e_bar ** (p - 1.0))
            * (eps_B ** ((p + 1.0) / 4.0))
            * (n0 ** 0.5)
            * (E52 ** ((p + 3.0) / 4.0))
            * (t_days ** (-3.0 * (p - 1.0) / 4.0))
            * (D28 ** (-2.0))
            * (nu14 ** (-(p - 1.0) / 2.0))
        )

        return F_mJy / 1000.0  # mJy -> Jy


@dataclass(frozen=True)
class PLSH:
    """PLS H: nu > nu_c.

    This segment is not used in the Tier-1 detection-rate calculation, but the
    Master's thesis plans to incorporate it.

    We implement the on-axis normalisation at t_dec from Eq. (25) of the
    Bachelor's project. The off-axis q-scaling exponents follow from combining
    the temporal slope in each phase with t_p(q) (same reasoning as for PLS G).
    """

    name: str = "H"

    def a_II(self, p: float) -> float:  # noqa: N802
        # Phase II temporal slope for PLS H is alpha = -(3p-2)/4.
        # Together with t_p \propto \tilde{q}^{8/3} this gives
        # F_p \propto \tilde{q}^{(8/3)alpha} = \tilde{q}^{-2(3p-2)/3}.
        return (3.0 * p - 2.0) / 3.0

    def a_III(self, p: float) -> float:  # noqa: N802
        # Phase III temporal slope is -p in the Tier-1 jet-break approximation,
        # giving F_p \propto \tilde{q}^{-2p}.
        return p

    def F_dec_Jy(
        self,
        phys: AfterglowPhysicalParams,
        micro: MicrophysicsParams,
        t_dec_s: float,
    ) -> float:
        """Eq. (25) of the Bachelor's project (Tier 1) evaluated at t_dec.

        Returns Jy.
        """

        p = phys.p

        if p <= 2.0:
            raise ValueError("p must be > 2 for the standard afterglow scalings.")

        # Dimensionless scalings
        E52 = phys.E_kiso_erg / 1e52
        t_days = t_dec_s / DAY_S
        D28 = phys.D_euc_cm / 1e28
        nu14 = phys.nu_hz / 1e14

        # Microphysics
        eps_e = micro.epsilon_e
        eps_B = micro.epsilon_B
        eps_e_bar = eps_e * (p - 2.0) / (p - 1.0)

        z_factor = (
            (1.0 + micro.z) ** ((p + 2.0) / 4.0)
            if micro.include_redshift_factors
            else 1.0
        )

        # Eq. (25) gives flux in mJy.
        F_mJy = (
            0.855
            * (p - 0.98)
            * np.exp(1.95 * p)
            * z_factor
            * (eps_e_bar ** (p - 1.0))
            * (eps_B ** ((p - 2.0) / 4.0))
            * (E52 ** ((p + 2.0) / 4.0))
            * (t_days ** (-(3.0 * p - 2.0) / 4.0))
            * (D28 ** (-2.0))
            * (nu14 ** (-p / 2.0))
        )

        return F_mJy / 1000.0  # mJy -> Jy
