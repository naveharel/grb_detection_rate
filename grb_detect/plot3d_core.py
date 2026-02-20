"""Numerical core for the interactive 3D detection-rate surface.

This module intentionally contains **no Dash code** so that the heavy numerical
parts can be imported and unit-tested in headless environments.

The implementation is a direct refactor of the original `plot_3d.py` logic.
The goal is *clarity and modularity* while preserving behaviour.

Public functions
----------------
- :func:`make_model` : build a :class:`~grb_detect.detection_rate.DesmosRateModel`.
- :func:`quantize_tcad_seconds` : cadence quantisation used by the Dash app.
- :func:`compute_surface` : compute the log-rate surface and (optionally) regime IDs.
- :func:`maximize_log_surface_iterative` : grid-refinement maximisation (as in the app).
- :func:`discrete_regime_colorscale` : Plotly-compatible step colours for regimes.

Notes
-----
This code is *not* a general-purpose optimiser; it is a pragmatic helper for the
interactive surface exploration.
"""

from __future__ import annotations

import numpy as np

from .constants import DAY_S, DEG2_TO_SR
from .detection_rate import DesmosRateModel
from .params import AfterglowPhysicalParams, SurveyInstrumentParams


def make_model(
    A_log: float,
    f_live: float,
    t_overhead_s: float,
    omega_exp_deg2: float,
) -> DesmosRateModel:
    """Construct the Desmos/Tier-1 rate model used by the Dash app.

    Parameters
    ----------
    A_log:
        log10(F_lim_ref / Jy) used in the limiting-flux model.
    f_live:
        Live fraction entering the exposure-time relation.
    t_overhead_s:
        Overhead per exposure (readout, slew, etc.), in seconds.
    omega_exp_deg2:
        Field-of-view per exposure, in deg^2.

    Returns
    -------
    DesmosRateModel
        Fully initialised rate model (with cached derived scales).
    """

    phys = AfterglowPhysicalParams()
    instrument = SurveyInstrumentParams(
        omega_exp_sr=float(omega_exp_deg2) * DEG2_TO_SR,
        F_lim_ref_Jy=10 ** float(A_log),
        f_live=float(f_live),
        t_overhead_s=float(t_overhead_s),
    )
    return DesmosRateModel(phys=phys, instrument=instrument)


def quantize_tcad_seconds(t_s: np.ndarray) -> np.ndarray:
    """Quantise cadence values to the discrete set used by the Dash app.

    Behaviour (preserved from the original app):

    - If t >= 1 day: ceil to an *integer number of days* (1d, 2d, 3d, ...).
    - If t < 1 day: allowed values are DAY / 2^k (k = 1,2,3,...) and we round to
      the closest **not larger than 1 day** implied by the original logic.

    Parameters
    ----------
    t_s:
        Cadence in seconds (array-like).

    Returns
    -------
    ndarray
        Quantised cadence in seconds.
    """

    t = np.asarray(t_s, dtype=float)
    out = np.empty_like(t)

    ge_day = t >= DAY_S
    out[ge_day] = np.ceil(t[ge_day] / DAY_S) * DAY_S

    lt_day = ~ge_day
    if np.any(lt_day):
        ratio = DAY_S / np.maximum(t[lt_day], 1e-300)
        k = np.floor(np.log2(ratio)).astype(int)
        k = np.maximum(k, 1)
        out[lt_day] = DAY_S / (2.0 ** k)

    return out


def discrete_regime_colorscale() -> tuple[list[list[float | str]], list[str]]:
    """Return a discrete (step) Plotly colorscale for regime_id in {1..7}."""

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
        # Duplicate stops to avoid interpolation.
        cs.append([a, c])
        cs.append([b, c])

    return cs, colors


def compute_surface(
    model_day: DesmosRateModel,
    model_night: DesmosRateModel | None,
    i_det: int,
    *,
    quantize_cadence: bool,
    color_regimes: bool,
    nx: int = 220,
    ny: int = 260,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute the log-rate surface on a (log N_exp, log t_cad) grid.

    This is a refactor of the original `compute_surface` in `plot_3d.py`.

    Returns
    -------
    X_log, Y_log:
        Meshgrids of log10(N_exp) and log10(t_cad / s).
    Z_plot:
        Masked log10(rate) for display (NaN for invalid/too-low values).
    Z_raw:
        Unmasked log10(rate).
    regime_id:
        Array with values {1..7} (float), NaN where undefined; None if
        `color_regimes=False`.
    """

    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr
    x_min, x_max = 0.0, np.log10(N_exp_max)
    y_min, y_max = -8.0, 8.0

    logN = np.linspace(x_min, x_max, nx)
    logtcad = np.linspace(y_min, y_max, ny)
    X_log, Y_log = np.meshgrid(logN, logtcad)

    N_exp = 10 ** X_log
    t_cad_s = 10 ** Y_log

    if quantize_cadence:
        t_cad_eff = quantize_tcad_seconds(t_cad_s)
        is_subday = t_cad_eff < DAY_S

        Z_day = model_day.rate_log10(i_det=i_det, N_exp=N_exp, t_cad_s=t_cad_eff)

        if model_night is None:
            raise RuntimeError("model_night is required when quantize_cadence=True")
        Z_night = model_night.rate_log10(i_det=i_det, N_exp=N_exp, t_cad_s=t_cad_eff)

        Z_raw = np.where(is_subday, Z_night, Z_day)
    else:
        t_cad_eff = t_cad_s
        Z_raw = model_day.rate_log10(i_det=i_det, N_exp=N_exp, t_cad_s=t_cad_eff)

    # Display mask for the plotted surface only.
    Z_plot = np.where(np.isfinite(Z_raw) & (Z_raw >= -1.0), Z_raw, np.nan)

    regime_id: np.ndarray | None = None
    if color_regimes:
        regime_id = np.full(Z_raw.shape, np.nan, dtype=float)

        def fill_regimes(model: DesmosRateModel, sel: np.ndarray) -> None:
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

        if quantize_cadence:
            is_subday = t_cad_eff < DAY_S
            fill_regimes(model_day, ~is_subday)
            if model_night is None:
                raise RuntimeError("model_night is required when quantize_cadence=True")
            fill_regimes(model_night, is_subday)
        else:
            fill_regimes(model_day, np.ones_like(Z_raw, dtype=bool))

    return X_log, Y_log, Z_plot, Z_raw, regime_id


def maximize_log_surface_iterative(
    model_day: DesmosRateModel,
    model_night: DesmosRateModel | None,
    i_det: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    quantize_cadence: bool,
    n0x: int = 180,
    n0y: int = 220,
    n_refine: int = 3,
    zoom: float = 0.18,
    nfx: int = 280,
    nfy: int = 320,
) -> tuple[float, float, float]:
    """Iteratively maximise the log-rate on a rectangular log-grid.

    This is a refactor of the original `maximize_log_surface_iterative` function.

    Returns
    -------
    N_opt:
        Best N_exp (linear scale).
    t_cad_opt_s:
        Best cadence (seconds, linear scale).
    log10_R_opt:
        log10 of the best rate.
    """

    def eval_grid(
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        nx: int,
        ny: int,
    ) -> tuple[float, float, float] | None:
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        X, Y = np.meshgrid(xs, ys)
        N = 10 ** X
        t = 10 ** Y

        if quantize_cadence:
            t_eff = quantize_tcad_seconds(t)
            is_subday = t_eff < DAY_S
            if model_night is None:
                return None
            Z_day = model_day.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
            Z_night = model_night.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
            Z = np.where(is_subday, Z_night, Z_day)
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
