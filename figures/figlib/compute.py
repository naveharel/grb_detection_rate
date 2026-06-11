"""Thin evaluators over the engine's public methods.

These wrap ``DetectionRateModel`` so figure scripts read cleanly and so the
"app defaults" (analytic medians, no filters) are spelled out in one place. They
accept scalars or arrays for the strategy variables and broadcast like the engine.
"""

from __future__ import annotations

import numpy as np

from grb_detect.params import GPC_TO_CM


def _as_arr(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


# Regime taxonomy (see docs): A1-A3 flux-limited, A4-A6 cadence-limited, A7 doubly.
_REGIME_KEYS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
FLUX_LIMITED = frozenset({1, 2, 3})
CADENCE_LIMITED = frozenset({4, 5, 6})
DOUBLY_LIMITED = frozenset({7})


def regime_id_at(model, N_exp, t_cad_s, i_det: int, *, scalar: bool = True):
    """Active regime id (1..7 for A1..A7; NaN if unphysical) at a strategy point."""
    N = _as_arr(N_exp)
    t = _as_arr(t_cad_s)
    masks = model.region_masks(int(i_det), N, t, include_unphysical=False)
    shape = np.broadcast(N, t).shape
    ids = np.full(shape, np.nan, dtype=float)
    for k, key in enumerate(_REGIME_KEYS, start=1):
        ids[np.asarray(masks[key])] = float(k)
    if scalar and ids.size == 1:
        return float(ids.ravel()[0])
    return ids


def regime_family(regime_id) -> str:
    """Map a regime id to 'flux_limited' / 'cadence_limited' / 'doubly_limited' / 'none'."""
    if regime_id != regime_id:  # NaN
        return "none"
    rid = int(round(regime_id))
    if rid in FLUX_LIMITED:
        return "flux_limited"
    if rid in CADENCE_LIMITED:
        return "cadence_limited"
    if rid in DOUBLY_LIMITED:
        return "doubly_limited"
    return "none"


def q_median_at(
    model,
    N_exp,
    t_cad_s,
    i_det: int,
    *,
    full_integral: bool = False,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_min: float = 0.0,
    s_mode: str = "discrete",
    scalar: bool = True,
):
    """Median viewing-angle parameter q (= 1 + q-tilde) of detected GRBs.

    Mirrors the medians call in ``standalone_bridge._eval_point``. Returns a float when
    ``scalar`` (the default) and inputs are scalar, else a numpy array.
    """
    N = _as_arr(N_exp)
    t = _as_arr(t_cad_s)
    q_med, _ = model.compute_medians(
        int(i_det), N, t,
        full_integral=full_integral,
        q_min=q_min, D_min_cm=D_min_cm, s_min=s_min, s_mode=s_mode,
    )
    if scalar and q_med.size == 1:
        return float(q_med[0])
    return q_med


def medians_at(
    model,
    N_exp,
    t_cad_s,
    i_det: int,
    *,
    full_integral: bool = False,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_min: float = 0.0,
    s_mode: str = "discrete",
    scalar: bool = True,
):
    """(q_median, D_median_Gpc) of detected GRBs. D returned in Gpc."""
    N = _as_arr(N_exp)
    t = _as_arr(t_cad_s)
    q_med, D_med_cm = model.compute_medians(
        int(i_det), N, t,
        full_integral=full_integral,
        q_min=q_min, D_min_cm=D_min_cm, s_min=s_min, s_mode=s_mode,
    )
    D_med_Gpc = D_med_cm / GPC_TO_CM
    if scalar and q_med.size == 1:
        return float(q_med[0]), float(D_med_Gpc[0])
    return q_med, D_med_Gpc


def rate_at(
    model,
    N_exp,
    t_cad_s,
    i_det: int,
    *,
    full_integral: bool = False,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_min: float = 0.0,
    s_mode: str = "discrete",
    scalar: bool = True,
):
    """Detection rate R_det [yr^-1] (NOT log). NaN where the strategy is invalid.

    Note: this is the day-model rate; it does not apply the sub-day optical ``f_night``
    factor (out of scope for the day-model figures, consistent with ``models.py``).
    """
    N = _as_arr(N_exp)
    t = _as_arr(t_cad_s)
    if full_integral:
        logR = model.rate_log10_full_integral(
            int(i_det), N, t, q_min=q_min, D_min_cm=D_min_cm, s_min=s_min, s_mode=s_mode,
        )
    else:
        logR = model.rate_log10(
            int(i_det), N, t, q_min=q_min, D_min_cm=D_min_cm, s_min=s_min, s_mode=s_mode,
        )
    logR = np.asarray(logR, dtype=float)
    R = np.where(np.isfinite(logR), 10.0 ** logR, np.nan)
    if scalar and R.size == 1:
        return float(R.ravel()[0])
    return R
