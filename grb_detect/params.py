"""Parameter containers (dataclasses).

This module collects the parameter sets used throughout the analytic GRB afterglow
detection-rate calculation.

Conceptually the parameters split into:

1) Afterglow physics (E_k, n, p, theta_j, Gamma_0, ...)
2) Microphysics (epsilon_e, epsilon_B, optional redshift factors)
3) Survey/instrument parameters
   - telescope/exposure-level parameters: (Omega_exp, f_live, A, t_overhead)
   - survey-model definitions: (Omega_srv,max, t_exp,ref, alpha, t_night)
4) Survey strategy variables: (N_exp, t_cad)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np

from .constants import DEG2_TO_SR


# -------------------------------------------------------
# Simple cosmology helper (flat LCDM)
# -------------------------------------------------------

GPC_TO_CM: float = 3.085677581e27
CM_TO_GPC: float = 1.0 / GPC_TO_CM
C_KM_S: float = 299792.458


@lru_cache(maxsize=256)
def comoving_distance_gpc(
    z: float,
    H0_km_s_Mpc: float = 70.0,
    Omega_m: float = 0.3,
) -> float:
    """Comoving distance in Gpc for flat ΛCDM.

    The integral is cached by (z, H0, Omega_m) to avoid repeated computation in
    interactive contexts.
    """
    if z < 0:
        raise ValueError("z must be >= 0")

    Omega_L = 1.0 - Omega_m

    # Grid integration is sufficient for the current use (z up to a few).
    N = 2000
    z_grid = np.linspace(0.0, z, N)
    Ez = np.sqrt(Omega_m * (1.0 + z_grid) ** 3 + Omega_L)

    trapz = getattr(np, "trapezoid", np.trapz)
    integral = trapz(1.0 / Ez, z_grid)

    D_c_Mpc = (C_KM_S / H0_km_s_Mpc) * integral
    return D_c_Mpc / 1000.0  # Mpc -> Gpc


# -------------------------------------------------------
# Physical parameters
# -------------------------------------------------------

@dataclass(frozen=True)
class AfterglowPhysicalParams:
    """Fiducial physical parameters for the analytic afterglow model.

    Notes
    -----
    The current code uses a Euclidean calibration for:
      - D_euc_cm: a fixed distance scale used in the rate normalization
      - R_int_yr: intrinsic GRB rate within the Euclidean volume

    The commented `__post_init__` below provides a way to compute these from z_Euc
    in flat ΛCDM, but is not enabled by default to preserve established behaviour.
    """

    # Electron distribution
    p: float = 2.5

    # Energetics and environment
    E_kiso_erg: float = 1e53
    n0_cm3: float = 1.0
    theta_j_rad: float = 0.1
    gamma0: float = 10**2.5

    # Observing frequency
    nu_hz: float = 5e14

    # Cosmological calibration (not currently used to derive D_euc_cm by default)
    z_Euc: float = 2.0
    H0_km_s_Mpc: float = 70.0
    Omega_m: float = 0.3

    # Volumetric GRB rate
    rho_grb_gpc3_yr: float = 260.0  # [Gpc^-3 yr^-1]

    # Euclidean calibration (kept explicit for reproducibility with existing runs).
    # D_euc_cm = 1.63e28 cm = 5.28 Gpc, consistent with z_Euc ≈ 2.0 in flat ΛCDM
    # (H0 = 70 km/s/Mpc, Ω_m = 0.3): comoving distance D_c ≈ 5.27 Gpc.
    D_euc_cm: float = 1.63e28
    R_int_yr: float = (4.0 / 3.0) * math.pi * rho_grb_gpc3_yr * (CM_TO_GPC * D_euc_cm) ** 3

    # Uncomment to compute D_euc_cm and R_int_yr from z_Euc:
    # D_euc_cm: float = field(init=False)
    # R_int_yr: float = field(init=False)
    #
    # def __post_init__(self) -> None:
    #     z_key = float(round(self.z_Euc, 6))
    #     H0_key = float(round(self.H0_km_s_Mpc, 6))
    #     Om_key = float(round(self.Omega_m, 6))
    #
    #     D_c_gpc = comoving_distance_gpc(z_key, H0_key, Om_key)
    #     D_c_cm = D_c_gpc * GPC_TO_CM
    #     object.__setattr__(self, "D_euc_cm", D_c_cm)
    #
    #     volume_gpc3 = (4.0 / 3.0) * math.pi * (D_c_gpc ** 3)
    #     R_int = self.rho_grb_gpc3_yr * volume_gpc3
    #     object.__setattr__(self, "R_int_yr", R_int)


# -------------------------------------------------------
# Microphysics
# -------------------------------------------------------

@dataclass(frozen=True)
class MicrophysicsParams:
    epsilon_e: float = 1e-1
    epsilon_B: float = 1e-2

    z: float = 0.0
    include_redshift_factors: bool = False


# -------------------------------------------------------
# Survey / Instrument
# -------------------------------------------------------

@dataclass(frozen=True)
class SurveyTelescopeParams:
    """Exposure-level parameters set by the instrument and operations."""

    omega_exp_sr: float = 47.0 * DEG2_TO_SR
    f_live: float = 0.1
    t_overhead_s: float = 0.0

    # Limiting flux model anchor: F_lim(t_exp_ref) = F_lim_ref
    F_lim_ref_Jy: float = 10 ** (-4.68)


@dataclass(frozen=True)
class SurveyDesignParams:
    """Survey-model definitions and fixed design constraints.

    These are not typically varied when comparing strategies for a specific
    instrument, but they affect the mapping from (N_exp, t_cad) to detectability.
    """

    omega_survey_max_sr: float = 27500.0 * DEG2_TO_SR

    # Reference exposure for the limiting-flux model
    t_exp_ref_s: float = 30.0

    # F_lim ∝ t_exp^{-alpha} (alpha=1/2 for background-limited regime)
    alpha: float = 0.5

    # Default astronomical night length (used by the optical-survey cadence logic)
    t_night_s: float = 10.0 * 3600.0


@dataclass(frozen=True)
class SurveyInstrumentParams:
    """Container joining telescope parameters and survey design definitions."""

    telescope: SurveyTelescopeParams = field(default_factory=SurveyTelescopeParams)
    design: SurveyDesignParams = field(default_factory=SurveyDesignParams)

    # Convenience accessors (keeps call sites concise)
    @property
    def omega_exp_sr(self) -> float:
        return self.telescope.omega_exp_sr

    @property
    def omega_survey_max_sr(self) -> float:
        return self.design.omega_survey_max_sr

    @property
    def f_live(self) -> float:
        return self.telescope.f_live

    @property
    def t_overhead_s(self) -> float:
        return self.telescope.t_overhead_s

    @property
    def t_exp_ref_s(self) -> float:
        return self.design.t_exp_ref_s

    @property
    def F_lim_ref_Jy(self) -> float:
        return self.telescope.F_lim_ref_Jy

    @property
    def alpha(self) -> float:
        return self.design.alpha

    @property
    def t_night_s(self) -> float:
        return self.design.t_night_s


# -------------------------------------------------------
# Strategy variables
# -------------------------------------------------------

@dataclass(frozen=True)
class SurveyStrategy:
    N_exp: float
    t_cad_s: float
