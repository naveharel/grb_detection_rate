"""GRB afterglow detection-rate modelling.

This package is a refactor/re-implementation of the analytic framework
introduced in the Bachelor's project (Tier 1) and extended in the Desmos
prototype for survey-strategy optimisation.

The public API is intentionally small at this stage; most users will want:

- :class:`grb_detect.detection_rate.DesmosRateModel`

which reproduces the Desmos detection-rate surface R_det(N_exp, t_cad).
"""

from .params import AfterglowPhysicalParams, MicrophysicsParams, SurveyInstrumentParams, SurveyStrategy
from .pls import PLSG, PLSH
from .detection_rate import DesmosRateModel

__all__ = [
    "AfterglowPhysicalParams",
    "MicrophysicsParams",
    "SurveyInstrumentParams",
    "SurveyStrategy",
    "PLSG",
    "PLSH",
    "DesmosRateModel",
]
