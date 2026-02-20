"""Constants and unit conversions.

This codebase follows the conventions used in the Bachelor's project (Tier 1):

- time: seconds (s) and days (days)
- distance: centimeters (cm)
- frequency: Hertz (Hz)
- flux density: Jansky (Jy)
- angles: radians (rad)

We keep everything explicit rather than relying on a unit-handling library.
"""

from __future__ import annotations

import numpy as np

# ---- Time ----
DAY_S: float = 86400.0
YEAR_S: float = 365.25 * DAY_S

# ---- Angles and solid angles ----
DEG2RAD: float = np.pi / 180.0
DEG2_TO_SR: float = DEG2RAD**2
FOUR_PI: float = 4.0 * np.pi

# ---- Flux ----
# 1 Jy = 1e-23 erg / (s cm^2 Hz)
JY_TO_CGS: float = 1.0e-23
CGS_TO_JY: float = 1.0 / JY_TO_CGS

# ---- Convenience scalings used by the analytic formulae ----
CM_IN_1e28: float = 1.0e28
ERG_IN_1e52: float = 1.0e52
HZ_IN_1e14: float = 1.0e14

