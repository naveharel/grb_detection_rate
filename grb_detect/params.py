"""Parameter containers (dataclasses).

The Bachelor's project and the Desmos prototype use a clear separation between:

1) GRB/afterglow physical parameters (p, E_k, n, theta_j, Gamma_0, ...)
2) microphysics parameters (epsilon_e, epsilon_B, redshift factors)
3) survey/instrument parameters (FoV, maximal survey area, live fraction, overhead)
4) strategy variables (N_exp and t_cad)

Defaults in this file are the fiducial values used by demo_desmos_surface.py
(the Desmos-matching prototype setup).
"""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, field
import math
import numpy as np


# -------------------------------------------------------
# Constants
# -------------------------------------------------------

DEG2_TO_SR: float = 0.00030461741978670857  # (pi/180)^2

GPC_TO_CM: float = 3.085677581e27
CM_TO_GPC: float = 1.0 / GPC_TO_CM

C_KM_S: float = 299792.458


# -------------------------------------------------------
# Cosmology (flat LCDM)
# -------------------------------------------------------

@lru_cache(maxsize=256)
def comoving_distance_gpc(
    z: float,
    H0_km_s_Mpc: float = 70.0,
    Omega_m: float = 0.3,
) -> float:
    """
    Comoving distance in Gpc for flat ΛCDM, cached by (z, H0, Omega_m).

    This avoids recomputing the integral on every Dash callback. If you later
    add a z_Euc slider, the value will be computed only for new z values.
    """
    if z < 0:
        raise ValueError("z must be >= 0")

    Omega_L = 1.0 - Omega_m

    # Use a grid dense enough for z~2 accuracy but still cheap.
    # You can tune N later if you like.
    N = 2000
    z_grid = np.linspace(0.0, z, N)
    Ez = np.sqrt(Omega_m * (1.0 + z_grid) ** 3 + Omega_L)

    # NumPy compatibility: `np.trapezoid` exists in newer versions, while
    # `np.trapz` is the long-standing implementation.
    trapz = getattr(np, "trapezoid", np.trapz)
    integral = trapz(1.0 / Ez, z_grid)

    D_c_Mpc = (C_KM_S / H0_km_s_Mpc) * integral
    return D_c_Mpc / 1000.0  # Mpc -> Gpc



# -------------------------------------------------------
# Physical parameters
# -------------------------------------------------------

@dataclass(frozen=True)
class AfterglowPhysicalParams:
    """
    Physical parameters for the analytic afterglow model.

    D_euc_cm is derived from z_Euc via comoving distance.
    R_int_yr is derived from volumetric GRB rate density.
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

    # Cosmological calibration
    z_Euc: float = 2.0
    H0_km_s_Mpc: float = 70.0
    Omega_m: float = 0.3

    # Volumetric GRB rate
    rho_grb_gpc3_yr: float = 260.0  # [Gpc^-3 yr^-1]

    # # Derived fields
    # D_euc_cm: float = field(init=False)
    # R_int_yr: float = field(init=False)
    #
    # def __post_init__(self) -> None:
    #
    #     # --- Comoving distance ---
    #     z_key = float(round(self.z_Euc, 6))
    #     H0_key = float(round(self.H0_km_s_Mpc, 6))
    #     Om_key = float(round(self.Omega_m, 6))
    #
    #     D_c_gpc = comoving_distance_gpc(z_key, H0_key, Om_key)
    #     D_c_cm = D_c_gpc * GPC_TO_CM
    #     object.__setattr__(self, "D_euc_cm", D_c_cm)
    #
    #     # --- Intrinsic GRB rate ---
    #     volume_gpc3 = (4.0 / 3.0) * math.pi * (D_c_gpc ** 3)
    #     R_int = self.rho_grb_gpc3_yr * volume_gpc3
    #
    #     object.__setattr__(self, "R_int_yr", R_int)
    D_euc_cm: float = 1.63e28
    R_int_yr: float = (4.0 / 3.0) * math.pi * rho_grb_gpc3_yr * (CM_TO_GPC * D_euc_cm) ** 3


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
class SurveyInstrumentParams:

    omega_exp_sr: float = 47.0 * DEG2_TO_SR
    omega_survey_max_sr: float = 27500.0 * DEG2_TO_SR

    f_live: float = 0.1
    t_overhead_s: float = 0.0

    t_exp_ref_s: float = 30.0
    F_lim_ref_Jy: float = 10 ** (-4.68)

    alpha: float = 0.5


# -------------------------------------------------------
# Strategy
# -------------------------------------------------------

@dataclass(frozen=True)
class SurveyStrategy:

    N_exp: float
    t_cad_s: float
