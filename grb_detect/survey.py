"""Survey/instrument modeling.

This module defines the mapping between the survey strategy variables
(N_exp, t_cad) and instrumental quantities (t_exp, F_lim, sky coverage).

Definitions
-----------
- N_exp: number of exposures per cadence cycle (proxy for sky area covered).
  Covered solid angle: Omega_srv = N_exp * Omega_exp, with hard cap
  Omega_srv <= Omega_srv,max.

- t_cad: cadence, time between consecutive visits to the same field.

Given live fraction f_live and overhead per exposure t_overhead, the exposure time
per visit is:

    t_exp = f_live * t_cad / N_exp - t_overhead.

The limiting flux is modeled as:

    F_lim(t_exp) = F_ref * (t_exp / t_ref)^(-alpha)

with alpha = 1/2 for background-limited observations.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import FOUR_PI
from .params import SurveyInstrumentParams, SurveyStrategy


def exposure_time_s(strategy: SurveyStrategy, instrument: SurveyInstrumentParams) -> float:
    """Compute exposure time per field/visit.

    Parameters
    ----------
    strategy:
        Survey strategy (N_exp, t_cad).
    instrument:
        Survey/instrument parameters.

    Returns
    -------
    float
        Exposure time t_exp [s].
    """

    if strategy.N_exp <= 0:
        raise ValueError("N_exp must be > 0")

    return instrument.f_live * strategy.t_cad_s / strategy.N_exp - instrument.t_overhead_s


def limiting_flux_Jy(t_exp_s: float, instrument: SurveyInstrumentParams) -> float:
    """Power-law limiting flux model used in the analytic survey model.

    F_lim(t_exp) = F_ref * (t_exp/t_ref)^(-alpha)

    Parameters
    ----------
    t_exp_s:
        Exposure time [s].
    instrument:
        Survey/instrument parameters.

    Returns
    -------
    float
        Limiting flux density [Jy].
    """

    if t_exp_s <= 0:
        return np.inf

    return instrument.F_lim_ref_Jy * (t_exp_s / instrument.t_exp_ref_s) ** (-instrument.alpha)


def sky_fraction(strategy: SurveyStrategy, instrument: SurveyInstrumentParams) -> float:
    """Fraction of the full sky covered by the survey strategy."""

    omega_srv = strategy.N_exp * instrument.omega_exp_sr
    return float(omega_srv / FOUR_PI)


def N_exp_max(instrument: SurveyInstrumentParams) -> float:
    """Maximum N_exp allowed by the surveyable area cap."""

    return instrument.omega_survey_max_sr / instrument.omega_exp_sr


def is_strategy_physical(strategy: SurveyStrategy, instrument: SurveyInstrumentParams) -> bool:
    """Basic physical constraints used in the interactive surface.

    Constraints implemented (matching the surface mask A0(N_exp, t_cad)):
    - 1 <= N_exp <= Omega_srv,max / Omega_exp
    - t_exp > 0
    """

    if strategy.N_exp < 1:
        return False

    if strategy.N_exp > N_exp_max(instrument) + 1e-12:
        return False

    t_exp = exposure_time_s(strategy, instrument)
    if not np.isfinite(t_exp) or t_exp <= 0:
        return False

    return True
