# grb_detect/core.py
"""Numerical core for the 3D detection-rate surface.

Pure-Python compute helpers used by the in-browser Pyodide app (via
standalone_bridge.py):
- build a rate model from user-facing survey parameters
- map cadence to an effective cadence for optical surveys
- evaluate the log-rate surface and (optionally) regime identifiers
- grid-refinement maximisation of the surface
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .constants import DAY_S, DEG2_TO_SR
from .detection_rate import DetectionRateModel
from .params import (
    AfterglowPhysicalParams,
    CM_TO_GPC,
    GPC_TO_CM,
    MicrophysicsParams,
    SurveyDesignParams,
    SurveyInstrumentParams,
    SurveyTelescopeParams,
)

# Minimum log10 R_det shown on the surface (rates below this are clipped)
ZMIN_DISPLAY_LOG10: float = -2.0


@lru_cache(maxsize=64)
def _make_rate_model_cached(
    A_log: float,
    f_live: float,
    t_overhead_s: float,
    omega_exp_deg2: float,
    omega_survey_max_sr: float,
    # Physics — all resolved to concrete floats (no None) by the public wrapper
    p: float,
    E_kiso_log10: float,
    n0_log10: float,
    epsilon_e_log10: float,
    epsilon_B_log10: float,
    theta_j_rad: float,
    gamma0_log10: float,
    nu_log10: float,
    D_euc_gpc: float,
    rho_grb_log10: float,
    win_i_minus_one: bool,
    win_from_peak: bool,
    N_v: int,
    dt_v_s: float | None,
) -> DetectionRateModel:
    """Cached model construction — called only when parameters change."""
    p_val      = float(p)
    E_kiso_val = 10.0 ** float(E_kiso_log10)
    n0_val     = 10.0 ** float(n0_log10)
    eps_e_val  = 10.0 ** float(epsilon_e_log10)
    eps_B_val  = 10.0 ** float(epsilon_B_log10)
    theta_j_val = float(theta_j_rad)
    gamma0_val  = 10.0 ** float(gamma0_log10)
    nu_val      = 10.0 ** float(nu_log10)
    D_euc_val   = float(D_euc_gpc) * GPC_TO_CM
    rho_val     = 10.0 ** float(rho_grb_log10)

    # R_int_yr must be recomputed explicitly: the frozen dataclass bakes it in at class-load
    # time from the default rho=260 and D_euc=1.63e28 and does NOT re-evaluate on construction.
    D_euc_gpc_val = D_euc_val * CM_TO_GPC
    R_int_val = (4.0 / 3.0) * math.pi * rho_val * D_euc_gpc_val ** 3

    phys = AfterglowPhysicalParams(
        p=p_val,
        E_kiso_erg=E_kiso_val,
        n0_cm3=n0_val,
        theta_j_rad=theta_j_val,
        gamma0=gamma0_val,
        nu_hz=nu_val,
        D_euc_cm=D_euc_val,
        rho_grb_gpc3_yr=rho_val,
        R_int_yr=R_int_val,
    )
    micro = MicrophysicsParams(epsilon_e=eps_e_val, epsilon_B=eps_B_val)
    telescope = SurveyTelescopeParams(
        omega_exp_sr=float(omega_exp_deg2) * DEG2_TO_SR,
        F_lim_ref_Jy=10 ** float(A_log),
        f_live=float(f_live),
        t_overhead_s=float(t_overhead_s),
    )
    instrument = SurveyInstrumentParams(
        telescope=telescope,
        design=SurveyDesignParams(omega_survey_max_sr=float(omega_survey_max_sr)),
    )
    return DetectionRateModel(
        phys=phys, instrument=instrument, micro=micro,
        win_i_minus_one=bool(win_i_minus_one),
        win_from_peak=bool(win_from_peak),
        N_v=int(N_v), dt_v_s=dt_v_s,
    )


def make_rate_model(
    *,
    A_log: float,
    f_live: float,
    t_overhead_s: float,
    omega_exp_deg2: float,
    design: SurveyDesignParams | None = None,
    # Physics kwargs — all optional; default to AfterglowPhysicalParams / MicrophysicsParams defaults
    p: float | None = None,
    E_kiso_log10: float | None = None,
    n0_log10: float | None = None,
    epsilon_e_log10: float | None = None,
    epsilon_B_log10: float | None = None,
    theta_j_rad: float | None = None,
    gamma0_log10: float | None = None,
    nu_log10: float | None = None,
    D_euc_gpc: float | None = None,
    rho_grb_log10: float | None = None,
    # Detection-window settings (see DetectionRateModel docstring)
    win_i_minus_one: bool = False,
    win_from_peak: bool = False,
    # Night schedule (see DetectionRateModel docstring): N_v=1 (default)
    # reproduces the plain single-cadence model regardless of dt_v_s.
    N_v: int = 1,
    dt_v_s: float | None = None,
) -> DetectionRateModel:
    """Construct a rate model from the survey parameters exposed in the UI.

    Results are cached (up to 64 unique parameter combinations) so repeated
    construction from the same slider values is free.
    """
    _d = AfterglowPhysicalParams()
    _md = MicrophysicsParams()
    _sd = SurveyDesignParams()

    # Resolve optional physics to concrete defaults, then round to 8 sig-fig
    # to ensure cache hits for slider values that only differ in floating-point noise.
    def _r(x: float, n: int = 8) -> float:
        return round(float(x), n)

    omega_max = design.omega_survey_max_sr if design is not None else _sd.omega_survey_max_sr

    # Normalize before the cache key: at N_v <= 1 there is no intra-night
    # schedule, so dt_v_s is irrelevant — force it to None (a no-op inside
    # DetectionRateModel) rather than caching a spurious dependence on it.
    N_v_val = int(N_v)
    dt_v_val = float(dt_v_s) if (N_v_val > 1 and dt_v_s is not None) else None

    return _make_rate_model_cached(
        _r(A_log),
        _r(f_live),
        _r(t_overhead_s),
        _r(omega_exp_deg2),
        _r(omega_max, 12),
        # physics
        _r(p)              if p              is not None else _r(_d.p),
        _r(E_kiso_log10)   if E_kiso_log10   is not None else _r(math.log10(_d.E_kiso_erg)),
        _r(n0_log10)       if n0_log10       is not None else _r(math.log10(_d.n0_cm3)),
        _r(epsilon_e_log10) if epsilon_e_log10 is not None else _r(math.log10(_md.epsilon_e)),
        _r(epsilon_B_log10) if epsilon_B_log10 is not None else _r(math.log10(_md.epsilon_B)),
        _r(theta_j_rad)    if theta_j_rad    is not None else _r(_d.theta_j_rad),
        _r(gamma0_log10)   if gamma0_log10   is not None else _r(math.log10(_d.gamma0)),
        _r(nu_log10)       if nu_log10       is not None else _r(math.log10(_d.nu_hz)),
        _r(D_euc_gpc)      if D_euc_gpc      is not None else _r(_d.D_euc_cm * CM_TO_GPC),
        _r(rho_grb_log10)  if rho_grb_log10  is not None else _r(math.log10(_d.rho_grb_gpc3_yr)),
        bool(win_i_minus_one),
        bool(win_from_peak),
        N_v_val,
        dt_v_val,
    )


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
    """Optical-survey cadence validity: integer day multiples only.

    Observations are possible only at night, so the only physically
    meaningful field-revisit cadences are whole numbers of nights:

        t_cad = n * 1 day,  n ∈ ℕ.

    Sub-day sampling is not a separate cadence regime here — revisiting a
    field within one active night is what the (dtv-window) `N_v`/`dt_v_s`
    schedule parameters on `DetectionRateModel` describe, decoupled from
    `t_cad` (see that class's docstring). This collapses the historical
    continuous/gap bands (dead since no caller can reach sub-day t_cad
    anymore) into a single discrete-day check.

    ``i_det`` and ``t_night_s`` are kept for call-site compatibility; they no
    longer affect the validity rule.

    Returns
    -------
    t_eff:
        Effective cadence in seconds (equal to the requested cadence).
    valid:
        Boolean mask of physically allowed points.
    """
    del i_det, t_night_s
    t = np.asarray(t_cad_s, dtype=float)
    t_eff = np.array(t, copy=True)

    valid = np.isfinite(t_eff) & (t_eff >= float(DAY_S)) & _is_integer_day_multiple(t_eff)
    return t_eff, valid


def _build_optical_tcad_grid(
    *,
    t_min_s: float,
    t_max_s: float,
    ny: int,
) -> np.ndarray:
    """Cadence grid for optical surveys: integer day multiples only.

    Exact integers n*DAY_S, sampled via log-spacing in n (up to ``ny`` unique
    values in [max(1 day, t_min), t_max]).
    """
    t_min_s = float(t_min_s)
    t_max_s = float(t_max_s)

    max_days = int(np.floor(t_max_s / float(DAY_S)))
    min_days = max(1, int(np.ceil(t_min_s / float(DAY_S))))
    if max_days < min_days:
        return np.array([float(min_days) * float(DAY_S)], dtype=float)

    n_vals = np.unique(
        np.clip(
            np.rint(np.logspace(np.log10(min_days), np.log10(max_days), int(ny))),
            min_days, max_days,
        ).astype(int)
    )
    return n_vals.astype(float) * float(DAY_S)


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


def _rate(
    model: DetectionRateModel,
    i_det: int,
    N_exp: np.ndarray,
    t_cad_s: np.ndarray,
    full_integral: bool,
    *,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_fade: float = 0.0,
    s_rise: float = 0.0,
    s_mode: str = "discrete",
    rise_random_start: bool = True,
    fade_random_start: bool = True,
) -> np.ndarray:
    """Dispatch to the full-integral or dominant-term rate method.

    The q_min / D_min_cm / s_fade / s_rise filters are applied inside the chosen rate
    method so R_total and the filters share the same approximation level.  At
    q_min=0, D_min_cm=0, s_fade=0 the result reduces to the unfiltered behavior.

    ``rise_random_start`` / ``fade_random_start`` select, per cut, the
    uniform-start survival weight (True) vs. the best-case hard boundary
    (False) inside the full-integral path; they are inert in the dominant-term
    path (already hard boundaries).
    """
    if full_integral:
        return model.rate_log10_full_integral(
            i_det, N_exp, t_cad_s,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )
    return model.rate_log10(
        i_det, N_exp, t_cad_s,
        q_min=q_min, D_min_cm=D_min_cm,
        s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
        rise_random_start=rise_random_start,
        fade_random_start=fade_random_start,
    )


def compute_surface(
    model: DetectionRateModel,
    i_det: int,
    *,
    optical_survey: bool,
    color_regimes: bool,
    t_night_s: float,
    nx: int = 220,
    ny: int = 260,
    full_integral: bool = False,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_fade: float = 0.0,
    s_rise: float = 0.0,
    s_mode: str = "discrete",
    rise_random_start: bool = True,
    fade_random_start: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Compute the log-rate surface on a (log N_exp, log t_cad) grid.

    Optical mode uses the integer-day cadence grid (any intra-night schedule
    lives in the model itself via N_v/dt_v_s, so a single model serves every
    cell); non-optical mode is the original continuous grid.

    Returns X, Y as LINEAR coordinates (N_exp and t_cad in seconds) so Plotly can
    use true log axes while preserving the surface shape.
    """
    N_exp_max = model.instrument.omega_survey_max_sr / model.instrument.omega_exp_sr
    x_min, x_max = 0.0, np.log10(N_exp_max)

    # Cadence range requested by the app: 1 s to 1e8 s
    t_min_s = 1.0
    t_max_s = 1e8

    logN = np.linspace(x_min, x_max, nx)
    N_exp_1d = 10 ** logN

    if optical_survey:
        t_cad_1d = _build_optical_tcad_grid(t_min_s=t_min_s, t_max_s=t_max_s, ny=ny)
        # Meshgrid with "xy" style (rows correspond to cadence, cols to N_exp)
        N_exp, t_cad_s = np.meshgrid(N_exp_1d, t_cad_1d)
        t_cad_eff, valid = optical_survey_tcad_seconds(
            t_cad_s, i_det=int(i_det), t_night_s=float(t_night_s))
    else:
        logtcad  = np.linspace(np.log10(t_min_s), np.log10(t_max_s), ny)
        t_cad_1d = 10 ** logtcad
        t_oh = model.instrument.t_overhead_s
        f    = model.instrument.f_live
        if t_oh > 0:
            # Adaptive 2D N_exp grid: rows near the t_OH validity boundary get denser
            # N_exp sampling to show the smooth R_det → 0 decline (not a physics change).
            N_exp   = _build_adaptive_N_exp_grid(t_cad_1d, nx, x_min, x_max, f, t_oh)
            t_cad_s = np.tile(t_cad_1d[:, np.newaxis], (1, nx))
        else:
            N_exp, t_cad_s = np.meshgrid(N_exp_1d, t_cad_1d)
        t_cad_eff = t_cad_s
        valid = None

    Z_raw = _rate(model, i_det, N_exp, t_cad_eff, full_integral,
                  q_min=q_min, D_min_cm=D_min_cm,
                  s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
                  rise_random_start=rise_random_start,
                  fade_random_start=fade_random_start)
    if valid is not None:
        Z_raw = np.where(valid, Z_raw, np.nan)

    # Display mask for the plotted surface only
    Z_plot = np.where(np.isfinite(Z_raw) & (Z_raw >= ZMIN_DISPLAY_LOG10), Z_raw, np.nan)

    regime_id: np.ndarray | None = None
    if color_regimes:
        regime_id = np.full(Z_raw.shape, np.nan, dtype=float)
        masks = model.region_masks(i_det, N_exp, t_cad_eff)
        for k, key in enumerate(("A1", "A2", "A3", "A4", "A5", "A6", "A7"), start=1):
            regime_id[masks[key]] = k
        if valid is not None:
            regime_id = np.where(valid, regime_id, np.nan)

    # ── Per-point extras: t_exp, median q, median D ──────────────────────────
    # A single model serves every cell, so rate, t_exp and medians always
    # describe the same population. t_exp uses t_cad_eff to match the rate.
    t_exp_grid = model.t_exp_s(N_exp, t_cad_eff)
    q_med_grid, D_med_cm_grid = model.compute_medians(
        i_det, N_exp, t_cad_eff, full_integral=full_integral,
        q_min=q_min, D_min_cm=D_min_cm,
        s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
        rise_random_start=rise_random_start,
        fade_random_start=fade_random_start,
    )
    if valid is not None:
        t_exp_grid    = np.where(valid, t_exp_grid, np.nan)
        q_med_grid    = np.where(valid, q_med_grid, np.nan)
        D_med_cm_grid = np.where(valid, D_med_cm_grid, np.nan)
    D_med_Gpc_grid  = D_med_cm_grid / GPC_TO_CM

    # Return linear coordinates for true log axes
    X_lin = N_exp
    Y_lin = t_cad_s
    return X_lin, Y_lin, Z_plot, Z_raw, regime_id, t_exp_grid, q_med_grid, D_med_Gpc_grid


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
    model: DetectionRateModel,
    i_det: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    optical_survey: bool,
    t_night_s: float,
    full_integral: bool = False,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_fade: float = 0.0,
    s_rise: float = 0.0,
    s_mode: str = "discrete",
    rise_random_start: bool = True,
    fade_random_start: bool = True,
    n0x: int = 180,
    n0y: int = 220,
    n_refine: int = 3,
    zoom: float = 0.18,
    nfx: int = 280,
    nfy: int = 320,
    validity_fn=None,
) -> tuple[float, float, float]:
    """Iteratively maximise the log-rate on a rectangular log-grid.

    For optical surveys the search domain is the discrete cadence set
    t_cad = n*DAY_S (n integer) — the only valid optical cadences.

    Parameters
    ----------
    full_integral : bool
        If True, use ``rate_log10_full_integral`` instead of the dominant-term
        approximation.  Must match the flag used in ``compute_surface`` so the
        optimizer finds the maximum of the same surface that is displayed.
    q_min, D_min_cm : float, optional
        Lower bounds on q and D.  Must match the values used in ``compute_surface``
        so the optimum lies on the same filtered surface that is displayed.
    validity_fn : callable(N_arr, t_arr) -> bool_array, optional
        If provided, called with linear N_exp and t_cad_s arrays of the same shape as
        the evaluation grid.  Points where the function returns False are masked out
        (set to nan) before argmax.  Use this to restrict the optimizer to a specific
        valid domain (e.g. the approx-mode validity boundary).  Default None preserves
        the existing behaviour.
    """

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

        Z = _rate(model, i_det, N2, t2, full_integral,
                  q_min=q_min, D_min_cm=D_min_cm,
                  s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
                  rise_random_start=rise_random_start,
                  fade_random_start=fade_random_start)
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
            Z = _rate(model, i_det, N, t, full_integral,
                      q_min=q_min, D_min_cm=D_min_cm,
                      s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
                      rise_random_start=rise_random_start,
                      fade_random_start=fade_random_start)
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

    # Optical survey: the only valid cadences are integer day multiples.
    t_min_disc = max(float(DAY_S), 10 ** y_min)
    t_max_disc = 10 ** y_max
    best_disc = eval_grid_discrete_days(x_min, x_max, t_min_disc, t_max_disc, nx=nfx, n_days=max(80, int(0.6 * nfy)))

    if best_disc is None or not np.isfinite(best_disc[2]):
        return np.nan, np.nan, np.nan

    x0, y0, z0 = best_disc
    _warn_if_invalid(validity_fn, x0, y0)
    N_ret = 10.0 ** x0
    t_ret = 10.0 ** y0
    # Snap to exact integer-day multiple if within floating-point tolerance.
    # 10**log10(n*DAY_S) can underestimate by ~1e-11, causing t_ret < DAY_S for
    # a 1-day cadence and triggering the wrong model dispatch in _eval_point.
    ratio = t_ret / float(DAY_S)
    nearest_n = round(ratio)
    if nearest_n >= 1 and abs(ratio - nearest_n) < 1e-9:
        t_ret = float(nearest_n) * float(DAY_S)
    return N_ret, t_ret, z0