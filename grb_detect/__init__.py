"""GRB afterglow detection-rate modelling.

This package implements an analytic framework (based on the Bachelor's project)
for computing the detection-rate surface R_det(N_exp, t_cad) under simplified
assumptions (ISM external medium, Euclidean rate treatment, chosen PLS model).

Most users will interact with:
- DetectionRateModel: evaluates the piecewise log-rate surface
- plot3d_core: numerical helpers used by the interactive Dash app
"""

from .params import (
    AfterglowPhysicalParams,
    MicrophysicsParams,
    SurveyDesignParams,
    SurveyInstrumentParams,
    SurveyStrategy,
    SurveyTelescopeParams,
)
from .pls import PLSG, PLSH
from .detection_rate import DetectionRateModel

__all__ = [
    "AfterglowPhysicalParams",
    "MicrophysicsParams",
    "SurveyTelescopeParams",
    "SurveyDesignParams",
    "SurveyInstrumentParams",
    "SurveyStrategy",
    "PLSG",
    "PLSH",
    "DetectionRateModel",
]
