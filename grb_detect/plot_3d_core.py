# grb_detect/plot_3d_core.py
"""Numerical core for the interactive 3D detection-rate surface.

This module contains no Dash code. It provides the computational helpers used by
the interactive app:
- build a rate model from user-facing survey parameters
- map cadence to an effective cadence for optical surveys
- evaluate the log-rate surface and (optionally) regime identifiers
- grid-refinement maximisation of the surface
"""

from __future__ import annotations

import numpy as np

from .constants import DAY_S, DEG2_TO_SR
from .detection_rate import DetectionRateModel
from .params import AfterglowPhysicalParams, SurveyDesignParams, SurveyInstrumentParams, SurveyTelescopeParams


def make_rate_model(
    *,
    A_log: float,
    f_live: float,
    t_overhead_s: float,
    omega_exp_deg2: float,
    design: SurveyDesignParams | None = None,
) -> DetectionRateModel:
    """Construct a rate model from the survey parameters exposed in the UI."""
    phys = AfterglowPhysicalParams()

    telescope = SurveyTelescopeParams(
        omega_exp_sr=float(omega_exp_deg2) * DEG2_TO_SR,
        F_lim_ref_Jy=10 ** float(A_log),
        f_live=float(f_live),
        t_overhead_s=float(t_overhead_s),
    )
    instrument = SurveyInstrumentParams(
        telescope=telescope,
        design=design if design is not None else SurveyDesignParams(),
    )
    return DetectionRateModel(phys=phys, instrument=instrument)


def optical_survey_tcad_seconds(
    t_cad_s: np.ndarray,
    *,
    i_det: int,
    t_night_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map cadence to an effective cadence for an optical survey.

    Rules:
      - For t_cad >= 1 day: t_eff is quantized to integer multiples of a day (ceil).
      - For t_night <= t_cad < 1 day: cadences are ineffective and mapped to 1 day.
      - For t_cad < t_night: t_eff = t_cad (continuous), but must satisfy i_det * t_cad < t_night.
        Values that violate this are invalid.
    """
    t = np.asarray(t_cad_s, dtype=float)
    t_eff = np.array(t, copy=True)
    valid = np.isfinite(t_eff) & (t_eff > 0.0)

    # >= 1 day: quantize to integer day multiples
    ge_day = valid & (t_eff >= DAY_S)
    if np.any(ge_day):
        t_eff[ge_day] = np.ceil(t_eff[ge_day] / DAY_S) * DAY_S

    # t_night to 1 day: map to 1 day
    gap = valid & (t_eff >= float(t_night_s)) & (t_eff < DAY_S)
    if np.any(gap):
        t_eff[gap] = DAY_S

    # < t_night: continuous but must allow i detections within a night
    sub_night = valid & (t_eff < float(t_night_s))
    if np.any(sub_night):
        valid[sub_night] &= (float(i_det) * t_eff[sub_night] < float(t_night_s))

    return t_eff, valid


def discrete_regime_colorscale() -> tuple[list[list[float | str]], list[str]]:
    """Discrete (step) Plotly colorscale for regime_id in {1..7}."""
    colors = [
        "#6A3D9A",  # A1 purple
        "#1F78B4",  # A2 blue
        "#33A02C",  # A3 green
        "#FF7F00",  # A4 orange
        "#E31A1C",  # A5 red
        "#A6CEE3",  # A6 light blue
        "#B15928",  # A7 brown
    ]
    cs: list[list[float | str]] = []
    for k, c in enumerate(colors, start=1):
        a = (k - 1) / 7.0
        b = k / 7.0
        cs.append([a, c])
        cs.append([b, c])
    return cs, colors


def compute_surface(
    model_day: DetectionRateModel,
    model_night: DetectionRateModel | None,
    i_det: int,
    *,
    optical_survey: bool,
    color_regimes: bool,
    t_night_s: float,
    nx: int = 220,
    ny: int = 260,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute the log-rate surface on a (log N_exp, log t_cad) grid.

    Returns X, Y as LINEAR coordinates (N_exp and t_cad in seconds) so Plotly can
    use true log axes while preserving the surface shape.
    """
    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr
    x_min, x_max = 0.0, np.log10(N_exp_max)

    # Requested: t_cad from 1 s to 1e8 s
    y_min, y_max = 0.0, 8.0

    logN = np.linspace(x_min, x_max, nx)
    logtcad = np.linspace(y_min, y_max, ny)
    X_log, Y_log = np.meshgrid(logN, logtcad)

    N_exp = 10 ** X_log
    t_cad_s = 10 ** Y_log

    if optical_survey:
        t_cad_eff, valid = optical_survey_tcad_seconds(
            t_cad_s,
            i_det=int(i_det),
            t_night_s=float(t_night_s),
        )
        is_subday = t_cad_eff < DAY_S

        Z_day = model_day.rate_log10(i_det=i_det, N_exp=N_exp, t_cad_s=t_cad_eff)

        if model_night is None:
            raise RuntimeError("model_night is required when optical_survey=True")
        Z_night = model_night.rate_log10(i_det=i_det, N_exp=N_exp, t_cad_s=t_cad_eff)

        Z_raw = np.where(is_subday, Z_night, Z_day)
        Z_raw = np.where(valid, Z_raw, np.nan)
    else:
        t_cad_eff = t_cad_s
        Z_raw = model_day.rate_log10(i_det=i_det, N_exp=N_exp, t_cad_s=t_cad_eff)

    # Display mask for the plotted surface only
    Z_plot = np.where(np.isfinite(Z_raw) & (Z_raw >= -1.0), Z_raw, np.nan)

    regime_id: np.ndarray | None = None
    if color_regimes:
        regime_id = np.full(Z_raw.shape, np.nan, dtype=float)

        def fill_regimes(model: DetectionRateModel, sel: np.ndarray) -> None:
            if not np.any(sel):
                return
            masks = model.region_masks(i_det, N_exp, t_cad_eff, include_unphysical=False)
            regime_id[(masks["A1"] & sel)] = 1
            regime_id[(masks["A2"] & sel)] = 2
            regime_id[(masks["A3"] & sel)] = 3
            regime_id[(masks["A4"] & sel)] = 4
            regime_id[(masks["A5"] & sel)] = 5
            regime_id[(masks["A6"] & sel)] = 6
            regime_id[(masks["A7"] & sel)] = 7

        if optical_survey:
            is_subday = t_cad_eff < DAY_S
            fill_regimes(model_day, ~is_subday)
            if model_night is None:
                raise RuntimeError("model_night is required when optical_survey=True")
            fill_regimes(model_night, is_subday)
        else:
            fill_regimes(model_day, np.ones_like(Z_raw, dtype=bool))

    # Return linear coordinates for true log axes
    X_lin = N_exp
    Y_lin = t_cad_s
    return X_lin, Y_lin, Z_plot, Z_raw, regime_id


def maximize_log_surface_iterative(
    model_day: DetectionRateModel,
    model_night: DetectionRateModel | None,
    i_det: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    optical_survey: bool,
    t_night_s: float,
    n0x: int = 180,
    n0y: int = 220,
    n_refine: int = 3,
    zoom: float = 0.18,
    nfx: int = 280,
    nfy: int = 320,
) -> tuple[float, float, float]:
    """Iteratively maximise the log-rate on a rectangular log-grid.

    Returns
    -------
    N_opt:
        Best N_exp (linear scale).
    t_cad_opt_s:
        Best cadence (seconds, linear scale).
    log10_R_opt:
        log10 of the best rate.
    """

    def eval_grid(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int):
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        X, Y = np.meshgrid(xs, ys)
        N = 10 ** X
        t = 10 ** Y

        if optical_survey:
            t_eff, valid = optical_survey_tcad_seconds(
                t,
                i_det=int(i_det),
                t_night_s=float(t_night_s),
            )
            is_subday = t_eff < DAY_S
            if model_night is None:
                return None

            Z_day = model_day.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
            Z_night = model_night.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
            Z = np.where(is_subday, Z_night, Z_day)
            Z = np.where(valid, Z, np.nan)
        else:
            Z = model_day.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t)

        Z = np.where(np.isfinite(Z), Z, np.nan)
        if not np.any(np.isfinite(Z)):
            return None

        k = np.nanargmax(Z)
        ii, jj = np.unravel_index(k, Z.shape)
        return float(X[ii, jj]), float(Y[ii, jj]), float(Z[ii, jj])

    best = eval_grid(x_min, x_max, y_min, y_max, n0x, n0y)
    if best is None:
        return np.nan, np.nan, np.nan
    x0, y0, z0 = best

    for _ in range(n_refine):
        dx = (x_max - x_min) * zoom
        dy = (y_max - y_min) * zoom
        xa0, xa1 = max(x_min, x0 - dx), min(x_max, x0 + dx)
        ya0, ya1 = max(y_min, y0 - dy), min(y_max, y0 + dy)

        best = eval_grid(xa0, xa1, ya0, ya1, nfx, nfy)
        if best is None:
            break
        x0, y0, z0 = best
        x_min, x_max, y_min, y_max = xa0, xa1, ya0, ya1

    return 10 ** x0, 10 ** y0, z0