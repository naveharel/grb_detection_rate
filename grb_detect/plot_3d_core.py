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
ZMIN_DISPLAY_LOG10: float = -2.0

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
        "#FF1744",  # A1 Saturated · Range IV           (strongest warm, deep red)
        "#FF9100",  # A2 Distance-limited · Range III    (medium warm, orange)
        "#FFD740",  # A3 Distance-limited · Range II     (muted warm, amber)
        "#2979FF",  # A4 Cadence-limited · Range IV      (strongest cool, deep blue)
        "#00E5FF",  # A5 Cadence-limited · Range III     (medium cool, cyan)
        "#1DE9B6",  # A6 Cadence-limited · Range II      (muted cool, teal)
        "#9E9E9E",  # A7 Flux-limited · Range I          (neutral gray — D_dec limited)
    ]
    cs: list[list[float | str]] = []
    for k, c in enumerate(colors, start=1):
        a = (k - 1) / 7.0
        b = k / 7.0
        cs.append([a, c])
        cs.append([b, c])
    return cs, colors


def _build_adaptive_N_exp_grid(
    t_cad_1d: np.ndarray,
    nx: int,
    x_min: float,
    x_max: float,
    f_live: float,
    t_overhead_s: float,
) -> np.ndarray:
    """Return 2D N_exp array (ny × nx) with per-row refinement near the t_OH validity boundary.

    For rows where N_boundary = f_live * t_cad / t_overhead_s lies within [N_min, N_max],
    ~20% of the nx columns are reallocated to densely sample the approach to the boundary
    (showing the smooth R_det → 0 decline near t_exp → 0). All other rows use a standard
    uniform log-spaced grid. Physics are unchanged — only sampling density is adapted.
    """
    ny = len(t_cad_1d)
    N_min = 10 ** x_min
    N_max = 10 ** x_max
    N_boundary = f_live * np.asarray(t_cad_1d, dtype=float) / t_overhead_s  # (ny,)

    nx_refine = max(nx // 5, 15)  # columns allocated to near-boundary refinement
    nx_base   = nx - nx_refine    # columns for the main surface region
    f_split   = 0.5               # refinement zone starts at f_split × N_boundary

    result = np.empty((ny, nx), dtype=float)

    for i, N_b in enumerate(N_boundary):
        if N_b >= N_max:
            # Boundary outside grid: standard uniform log-spacing
            result[i] = 10 ** np.linspace(x_min, x_max, nx)
        else:
            N_b_eff      = N_b * 0.999                         # stay just inside boundary
            refine_start = max(N_min * 1.001, f_split * N_b_eff)

            logN_base   = np.linspace(x_min, np.log10(refine_start), nx_base)
            # Skip duplicate at refine_start; produces exactly nx_refine points
            logN_refine = np.linspace(
                np.log10(refine_start), np.log10(N_b_eff), nx_refine + 1
            )[1:]

            result[i] = 10 ** np.concatenate([logN_base, logN_refine])

    return result


def _on_axis_rate_linear(
    model: DetectionRateModel,
    i_det: int,
    N_exp: np.ndarray,
    t_cad_s: np.ndarray,
) -> np.ndarray:
    """Linear rate contribution from on-axis bursts (q < q_dec).

    For q < q_dec the full integral uses D_max(q) = D_dec (constant), so the
    on-axis integral evaluates analytically:

        R_on = 0.5 · fO · θ_j² · q_dec² · min(D_dec/D_euc, D_i/D_euc, 1)³ · R_int

    This is subtracted from R_total to obtain the off-axis-only rate.
    Returns the linear rate (not log10); NaN outside the physical domain.
    """
    t_exp   = model.t_exp_s(N_exp, t_cad_s)
    F_lim   = model.F_lim_Jy(t_exp)
    fO      = model.f_Omega(N_exp)
    D_dec   = model.D_dec(F_lim)
    D_i     = model.D_i(i_det, t_cad_s, F_lim)
    D_euc   = model.phys.D_euc_cm
    theta_j = model.phys.theta_j_rad
    q_dec   = float(model.derived.q_dec)
    R_int   = model.phys.R_int_yr

    # Cap D_dec at D_euc (Euclidean horizon).  We deliberately do NOT apply
    # the cadence distance D_i here: the A7 piecewise formula uses D_dec alone,
    # and mixing in D_i would cause a sign mismatch that leaves spurious positive
    # R_off values in A7 (where R_off should be exactly zero).  In cadence-limited
    # regimes (A4-A6) r_on is slightly over-estimated, giving a conservative
    # (under-counted) off-axis rate — acceptable near the A7 boundary.
    D_eff_on = np.minimum(D_dec / D_euc, 1.0)
    r_on = 0.5 * fO * (theta_j ** 2) * (q_dec ** 2) * (D_eff_on ** 3) * R_int
    return np.where(np.isfinite(t_exp) & (t_exp > 0), r_on, np.nan)


def _rate(
    model: DetectionRateModel,
    i_det: int,
    N_exp: np.ndarray,
    t_cad_s: np.ndarray,
    full_integral: bool,
    *,
    off_axis: bool = False,
) -> np.ndarray:
    """Dispatch to the full-integral or dominant-term rate method.

    When off_axis=True, subtracts the on-axis contribution (q < q_dec) from the
    total rate, yielding only bursts detected from outside the beaming cone.
    """
    if full_integral:
        logR = model.rate_log10_full_integral(i_det, N_exp, t_cad_s)
    else:
        logR = model.rate_log10(i_det, N_exp, t_cad_s)

    if off_axis:
        r_on  = _on_axis_rate_linear(model, i_det, N_exp, t_cad_s)
        R_off = np.where(np.isfinite(logR), 10.0 ** logR, np.nan) - r_on
        with np.errstate(divide="ignore", invalid="ignore"):
            logR = np.where(R_off > 0, np.log10(R_off), np.nan)
    return logR


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
    full_integral: bool = False,
    off_axis: bool = False,
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
        logtcad  = np.linspace(np.log10(t_min_s), np.log10(t_max_s), ny)
        t_cad_1d = 10 ** logtcad
        t_oh = model_day.instrument.t_overhead_s
        f    = model_day.instrument.f_live
        if t_oh > 0:
            # Adaptive 2D N_exp grid: rows near the t_OH validity boundary get denser
            # N_exp sampling to show the smooth R_det → 0 decline (not a physics change).
            N_exp   = _build_adaptive_N_exp_grid(t_cad_1d, nx, x_min, x_max, f, t_oh)
            t_cad_s = np.tile(t_cad_1d[:, np.newaxis], (1, nx))
        else:
            N_exp, t_cad_s = np.meshgrid(N_exp_1d, t_cad_1d)

    if optical_survey:
        t_cad_eff, valid = optical_survey_tcad_seconds(
            t_cad_s,
            i_det=int(i_det),
            t_night_s=float(t_night_s),
        )
        is_subday = t_cad_eff < float(DAY_S)

        Z_day = _rate(model_day, i_det, N_exp, t_cad_eff, full_integral, off_axis=off_axis)

        if model_night is None:
            raise RuntimeError("model_night is required when optical_survey=True")
        Z_night = _rate(model_night, i_det, N_exp, t_cad_eff, full_integral, off_axis=off_axis)

        # Sub-day: only the nighttime fraction of the sky is observable each cadence
        f_night = t_night_s / float(DAY_S)
        Z_raw = np.where(is_subday, Z_night + np.log10(f_night), Z_day)
        Z_raw = np.where(valid, Z_raw, np.nan)
    else:
        t_cad_eff = t_cad_s
        Z_raw = _rate(model_day, i_det, N_exp, t_cad_eff, full_integral, off_axis=off_axis)

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


def _warn_if_invalid(validity_fn, x0: float, y0: float) -> None:
    """Emit a warning if the optimizer result violates validity_fn (internal check)."""
    if validity_fn is None or not (np.isfinite(x0) and np.isfinite(y0)):
        return
    import warnings
    N_ret, t_ret = 10.0 ** x0, 10.0 ** y0
    if not np.all(validity_fn(np.array([N_ret]), np.array([t_ret]))):
        warnings.warn(
            f"maximize_log_surface_iterative: returned point "
            f"(N={N_ret:.3g}, t={t_ret:.3g}s) violates validity_fn — "
            f"validity_fn may not cover the full search domain.",
            stacklevel=3,
        )


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
    validity_fn=None,
) -> tuple[float, float, float]:
    """Iteratively maximise the log-rate on a rectangular log-grid.

    For optical surveys:
      - Search the continuous region t_cad < t_night/i_det with standard refinement.
      - Separately search the discrete region t_cad = n*DAY_S (n integer), and take the best.

    Parameters
    ----------
    validity_fn : callable(N_arr, t_arr) -> bool_array, optional
        If provided, called with linear N_exp and t_cad_s arrays of the same shape as
        the evaluation grid.  Points where the function returns False are masked out
        (set to nan) before argmax.  Use this to restrict the optimizer to a specific
        valid domain (e.g. the approx-mode validity boundary).  Default None preserves
        the existing behaviour.
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
        f_night = t_night_s / float(DAY_S)
        Z = np.where(is_subday, Z_night + np.log10(f_night), Z_day)
        Z = np.where(valid, Z, np.nan)
        if validity_fn is not None:
            Z = np.where(validity_fn(N, t_eff), Z, np.nan)

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
        if validity_fn is not None:
            Z = np.where(validity_fn(N2, t2), Z, np.nan)
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
            if validity_fn is not None:
                Z = np.where(validity_fn(N, t), Z, np.nan)
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

        _warn_if_invalid(validity_fn, x0, y0)
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
    _warn_if_invalid(validity_fn, x0, y0)
    return 10 ** x0, 10 ** y0, z0