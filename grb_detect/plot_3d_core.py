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

# Minimum log10 R_det shown on the surface (rates below this are clipped)
ZMIN_DISPLAY_LOG10: float = -1.0

# Fraction of optical cadence grid rows allocated to continuous / gap bands
_OPTICAL_GRID_FRAC_CONT: float = 0.45
_OPTICAL_GRID_FRAC_GAP: float = 0.10


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


def _is_integer_day_multiple(t_s: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Return mask for times that are (within tol) integer multiples of 1 day."""
    x = np.asarray(t_s, dtype=float) / float(DAY_S)
    r = np.rint(x)
    return np.isfinite(x) & (np.abs(x - r) <= tol * np.maximum(1.0, np.abs(x)))


def optical_survey_tcad_seconds(
    t_cad_s: np.ndarray,
    *,
    i_det: int,
    t_night_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Optical-survey cadence validity without rounding.

    Physical/operational constraints:
      1) Sub-night sampling must allow i_det visits in a single night:
            i_det * t_cad < t_night
         This defines the continuous, allowed region.

      2) Between t_night/i_det and 1 day, we disallow cadences:
         you cannot obtain i_det detections within one night, but cadence is still
         shorter than the day-night cycle. Those points are invalid (gap).

      3) For t_cad >= 1 day, observations are restricted to discrete nights:
         allowed cadences are integer multiples of 1 day:
            t_cad = n * 1 day,  n ∈ ℕ.
         We enforce this as a validity constraint, not by rounding.

    Returns
    -------
    t_eff:
        Effective cadence in seconds (here equal to the requested cadence, no rounding).
    valid:
        Boolean mask of physically allowed points.
    """
    t = np.asarray(t_cad_s, dtype=float)
    t_eff = np.array(t, copy=True)

    valid = np.isfinite(t_eff) & (t_eff > 0.0)

    t_cont_max = float(t_night_s) / float(i_det)

    # Continuous allowed region (strict inequality)
    cont = valid & (t_eff < t_cont_max)
    if np.any(cont):
        valid[cont] &= (float(i_det) * t_eff[cont] < float(t_night_s))

    # Gap region: (t_night/i_det) <= t < 1 day is invalid for i_det >= 2.
    # Note: for i_det = 1, this gap should not apply (cadences up to 1 day are valid
    # since a single detection fits in any sub-day window). Fix this when i_det = 1
    # support is added.
    gap = valid & (t_eff >= t_cont_max) & (t_eff < float(DAY_S))
    if np.any(gap):
        valid[gap] = False

    # Discrete day multiples region: t >= 1 day and t is an integer number of days
    ge_day = valid & (t_eff >= float(DAY_S))
    if np.any(ge_day):
        valid[ge_day] &= _is_integer_day_multiple(t_eff[ge_day])

    return t_eff, valid


def _build_optical_tcad_grid(
    *,
    i_det: int,
    t_night_s: float,
    t_min_s: float,
    t_max_s: float,
    ny: int,
    n_days: int | None = None,
) -> np.ndarray:
    """Piecewise cadence grid for optical surveys.

    Below t_night/i_det: continuous (log-spaced).
    Between t_night/i_det and 1 day: include a sparse log-spaced set, but it will be invalid and plot as a gap.
    Above 1 day: only integer multiples of a day (n*DAY_S), sampled via log-spacing in n (still exact integers).
    """
    t_min_s = float(t_min_s)
    t_max_s = float(t_max_s)

    t_cont_max = float(t_night_s) / float(i_det)

    # Allocate rows: prioritize continuous part + day-multiples part, keep a small "gap" band
    n_gap = max(10, int(_OPTICAL_GRID_FRAC_GAP * ny))
    n_cont = max(40, int(_OPTICAL_GRID_FRAC_CONT * ny))
    n_disc = max(40, ny - n_cont - n_gap)

    # Continuous part (cap at t_cont_max, and ensure strictly below it)
    t_cont_hi = min(t_cont_max * 0.999, t_max_s)
    if t_cont_hi <= t_min_s:
        t_cont = np.array([t_min_s], dtype=float)
    else:
        t_cont = np.logspace(np.log10(t_min_s), np.log10(t_cont_hi), n_cont)

    # Gap sampling (will be invalid, but creates a clean "hole" in the surface)
    # Only include if there is a gap interval in range.
    t_gap_lo = max(t_cont_max * 1.001, t_min_s)
    t_gap_hi = min(float(DAY_S) * 0.999, t_max_s)
    if t_gap_hi > t_gap_lo:
        t_gap = np.logspace(np.log10(t_gap_lo), np.log10(t_gap_hi), n_gap)
    else:
        t_gap = np.array([], dtype=float)

    # Discrete day multiples (exact integers n*DAY_S)
    t_disc_lo = max(float(DAY_S), t_min_s)
    if t_max_s >= t_disc_lo:
        max_days = int(np.floor(t_max_s / float(DAY_S)))
        if max_days < 1:
            t_days = np.array([], dtype=float)
        else:
            if n_days is None:
                n_days = n_disc
            # Sample integers in n using log spacing, then unique+sorted, and multiply by DAY_S
            n_vals = np.unique(np.clip(np.rint(np.logspace(0.0, np.log10(max_days), n_days)), 1, max_days).astype(int))
            t_days = n_vals.astype(float) * float(DAY_S)
    else:
        t_days = np.array([], dtype=float)

    t_all = np.concatenate([t_cont, t_gap, t_days])
    t_all = np.unique(t_all[np.isfinite(t_all)])
    t_all = t_all[(t_all > 0.0) & (t_all <= t_max_s)]
    t_all.sort()

    # Safety: ensure we do not return an empty grid
    if t_all.size == 0:
        t_all = np.array([t_min_s], dtype=float)

    return t_all


def discrete_regime_colorscale() -> tuple[list[list[float | str]], list[str]]:
    """Discrete (step) Plotly colorscale for regime_id in {1..7}."""
    # Warm (orange/red) = flux-limited (A1–A3, q_Euc > q_i — going deeper helps)
    # Cool (teal/blue)  = cadence-limited (A4–A6, q_i > q_Euc — faster cadence helps)
    # Neutral slate     = doubly limited (A7)
    colors = [
        "#FF1744",  # A1 flux-limited · decel. (Material red accent)
        "#FF9100",  # A2 flux-limited · post-jet (Material deep orange accent)
        "#FFD740",  # A3 flux-limited · pre-jet (Material amber accent)
        "#1DE9B6",  # A4 cadence-limited · decel. (Material teal accent)
        "#00E5FF",  # A5 cadence-limited · post-jet (Material cyan accent)
        "#2979FF",  # A6 cadence-limited · pre-jet (Material blue accent)
        "#9E9E9E",  # A7 doubly limited (neutral gray)
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

    # Cadence range requested by the app: 1 s to 1e8 s
    t_min_s = 1.0
    t_max_s = 1e8

    logN = np.linspace(x_min, x_max, nx)
    N_exp_1d = 10 ** logN

    if optical_survey:
        t_cad_1d = _build_optical_tcad_grid(
            i_det=int(i_det),
            t_night_s=float(t_night_s),
            t_min_s=t_min_s,
            t_max_s=t_max_s,
            ny=ny,
        )
        # Meshgrid with "xy" style (rows correspond to cadence, cols to N_exp)
        N_exp, t_cad_s = np.meshgrid(N_exp_1d, t_cad_1d)
    else:
        logtcad = np.linspace(np.log10(t_min_s), np.log10(t_max_s), ny)
        t_cad_1d = 10 ** logtcad
        N_exp, t_cad_s = np.meshgrid(N_exp_1d, t_cad_1d)

    if optical_survey:
        t_cad_eff, valid = optical_survey_tcad_seconds(
            t_cad_s,
            i_det=int(i_det),
            t_night_s=float(t_night_s),
        )
        is_subday = t_cad_eff < float(DAY_S)

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
    Z_plot = np.where(np.isfinite(Z_raw) & (Z_raw >= ZMIN_DISPLAY_LOG10), Z_raw, np.nan)

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
            # is_subday already computed above in the optical_survey block
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

    For optical surveys:
      - Search the continuous region t_cad < t_night/i_det with standard refinement.
      - Separately search the discrete region t_cad = n*DAY_S (n integer), and take the best.
    """

    def eval_grid_continuous(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int):
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        X, Y = np.meshgrid(xs, ys)
        N = 10 ** X
        t = 10 ** Y

        t_eff, valid = optical_survey_tcad_seconds(
            t,
            i_det=int(i_det),
            t_night_s=float(t_night_s),
        )
        if model_night is None:
            return None

        is_subday = t_eff < float(DAY_S)
        Z_day = model_day.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
        Z_night = model_night.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
        Z = np.where(is_subday, Z_night, Z_day)
        Z = np.where(valid, Z, np.nan)

        if not np.any(np.isfinite(Z)):
            return None

        k = np.nanargmax(Z)
        ii, jj = np.unravel_index(k, Z.shape)
        return float(X[ii, jj]), float(Y[ii, jj]), float(Z[ii, jj])

    def eval_grid_discrete_days(x0: float, x1: float, t_min: float, t_max: float, nx: int, n_days: int):
        # Candidate integer days in [t_min, t_max]
        t_lo = max(float(DAY_S), float(t_min))
        t_hi = float(t_max)
        if t_hi < t_lo:
            return None

        max_days = int(np.floor(t_hi / float(DAY_S)))
        min_days = int(np.ceil(t_lo / float(DAY_S)))
        if max_days < min_days:
            return None

        n_vals = np.unique(
            np.clip(
                np.rint(np.logspace(np.log10(max(1, min_days)), np.log10(max_days), n_days)),
                min_days,
                max_days,
            ).astype(int)
        )
        t_days = n_vals.astype(float) * float(DAY_S)

        xs = np.linspace(x0, x1, nx)
        N = 10 ** xs  # (nx,)
        # Meshgrid over (t_days, N)
        N2, t2 = np.meshgrid(N, t_days)

        Z = model_day.rate_log10(i_det=i_det, N_exp=N2, t_cad_s=t2)
        if not np.any(np.isfinite(Z)):
            return None

        k = np.nanargmax(Z)
        ii, jj = np.unravel_index(k, Z.shape)
        return float(xs[jj]), float(np.log10(t_days[ii])), float(Z[ii, jj])

    if not optical_survey:
        # Original behaviour for non-optical surveys
        def eval_grid_generic(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int):
            xs = np.linspace(x0, x1, nx)
            ys = np.linspace(y0, y1, ny)
            X, Y = np.meshgrid(xs, ys)
            N = 10 ** X
            t = 10 ** Y
            Z = model_day.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t)
            Z = np.where(np.isfinite(Z), Z, np.nan)
            if not np.any(np.isfinite(Z)):
                return None
            k = np.nanargmax(Z)
            ii, jj = np.unravel_index(k, Z.shape)
            return float(X[ii, jj]), float(Y[ii, jj]), float(Z[ii, jj])

        best = eval_grid_generic(x_min, x_max, y_min, y_max, n0x, n0y)
        if best is None:
            return np.nan, np.nan, np.nan
        x0, y0, z0 = best

        for _ in range(n_refine):
            dx = (x_max - x_min) * zoom
            dy = (y_max - y_min) * zoom
            xa0, xa1 = max(x_min, x0 - dx), min(x_max, x0 + dx)
            ya0, ya1 = max(y_min, y0 - dy), min(y_max, y0 + dy)

            best = eval_grid_generic(xa0, xa1, ya0, ya1, nfx, nfy)
            if best is None:
                break
            x0, y0, z0 = best
            x_min, x_max, y_min, y_max = xa0, xa1, ya0, ya1

        return 10 ** x0, 10 ** y0, z0

    # Optical survey: continuous search region is bounded above by log10(t_night/i_det)
    t_cont_max = float(t_night_s) / float(i_det)
    y_cont_max = np.log10(max(1.0, t_cont_max * 0.999))
    y0_min = y_min
    y0_max = min(y_max, y_cont_max)

    best_cont = None
    if y0_max > y0_min:
        best_cont = eval_grid_continuous(x_min, x_max, y0_min, y0_max, n0x, n0y)
        if best_cont is not None:
            xc, yc, zc = best_cont
            for _ in range(n_refine):
                dx = (x_max - x_min) * zoom
                dy = (y0_max - y0_min) * zoom
                xa0, xa1 = max(x_min, xc - dx), min(x_max, xc + dx)
                ya0, ya1 = max(y0_min, yc - dy), min(y0_max, yc + dy)

                refined = eval_grid_continuous(xa0, xa1, ya0, ya1, nfx, nfy)
                if refined is None:
                    break
                xc, yc, zc = refined
            best_cont = (xc, yc, zc)

    # Optical discrete day multiples region
    t_min_disc = max(float(DAY_S), 10 ** y_min)
    t_max_disc = 10 ** y_max
    best_disc = eval_grid_discrete_days(x_min, x_max, t_min_disc, t_max_disc, nx=nfx, n_days=max(80, int(0.6 * nfy)))

    # Choose best
    candidates = []
    if best_cont is not None and np.isfinite(best_cont[2]):
        candidates.append(best_cont)
    if best_disc is not None and np.isfinite(best_disc[2]):
        candidates.append(best_disc)

    if len(candidates) == 0:
        return np.nan, np.nan, np.nan

    x0, y0, z0 = max(candidates, key=lambda t: t[2])
    return 10 ** x0, 10 ** y0, z0