"""Standalone bridge for the GRB Detection Rate Pyodide HTML app.

Single public function: compute_all(params) -> dict.
All surface arrays are returned as flat Python lists (row-major), NaN replaced by
None so that Pyodide's .toJs() produces JS null, which Plotly treats as a gap.
"""
from __future__ import annotations

import math

import numpy as np

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.params import GPC_TO_CM, SurveyDesignParams, SurveyStrategy
from grb_detect.plot_3d_core import (
    ZMIN_DISPLAY_LOG10,
    _on_axis_rate_linear,
    _rate,
    compute_surface,
    make_rate_model,
    maximize_log_surface_iterative,
)
from grb_detect.survey import exposure_time_s

ZTF_OMEGA_EXP_DEG2: float = 47.0
OMEGA_SRV_DEFAULT_DEG2: float = 27500.0

# Grid resolutions — must match callbacks/surface.py:36-37
NX_REGIME, NY_REGIME = 200, 240
NX_DEFAULT, NY_DEFAULT = 160, 200


# ── Serialization helpers ────────────────────────────────────────────────────

def _nan_to_none(x):
    if x is None:
        return None
    try:
        f = float(x)
        return None if (f != f) else f  # nan != nan is True
    except (TypeError, ValueError):
        return None


def _array_to_list(arr) -> list:
    """Flatten numpy array to list, replacing NaN with None."""
    flat = np.asarray(arr, dtype=float).ravel()
    return [None if (v != v) else float(v) for v in flat.tolist()]


# ── Regime-id helper (mirrors _masks_1d from callbacks/surface.py) ──────────

def _masks_1d(masks: dict, n: int) -> np.ndarray:
    rid = np.full(n, np.nan, dtype=float)
    for k, key in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
        mk = np.asarray(masks[key]).ravel()
        if mk.size == n:
            rid[mk] = float(k)
    return rid


# ── 1-D rate sweep (mirrors _compute_rate from callbacks/surface.py) ────────

def _compute_rate(
    model,
    i_det: int,
    N_arr: np.ndarray,
    t_arr: np.ndarray,
    *,
    full_integral: bool,
    optical_on: bool,
    model_night,
    t_cad_scalar: float,
    f_night_val: float,
    approx_on: bool,
    f_live: float,
    t_overhead_s: float,
    color_on: bool,
    off_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (R, t_exp, q_med, D_med_Gpc, rid) for a 1-D sweep.

    Verbatim port of callbacks/surface.py:_compute_rate.
    """
    if full_integral:
        Z = model.rate_log10_full_integral(i_det, N_arr, t_arr)
    else:
        Z = model.rate_log10(i_det, N_arr, t_arr)
    R = np.where(np.isfinite(Z), 10.0 ** Z, np.nan)

    if optical_on and model_night is not None and t_cad_scalar < DAY_S:
        R = R * f_night_val

    if approx_on and float(t_overhead_s) > 0:
        R = np.where(float(f_live) * t_arr / N_arr <= float(t_overhead_s), np.nan, R)

    rid = np.full(len(N_arr), np.nan)
    if color_on:
        masks = model.region_masks(i_det, N_arr, t_arr, include_unphysical=False)
        rid = _masks_1d(masks, len(N_arr))

    q_med, D_med_cm = model.compute_medians(
        i_det, N_arr, t_arr, full_integral=full_integral, off_axis=off_axis,
    )
    t_exp = model.t_exp_s(N_arr, t_arr)
    D_med_Gpc = D_med_cm / GPC_TO_CM

    if approx_on and float(t_overhead_s) > 0:
        _inv = float(f_live) * t_arr / N_arr <= float(t_overhead_s)
        t_exp     = np.where(_inv, np.nan, t_exp)
        q_med     = np.where(_inv, np.nan, q_med)
        D_med_Gpc = np.where(_inv, np.nan, D_med_Gpc)

    return R, t_exp, q_med, D_med_Gpc, rid


# ── Point evaluation (mirrors _eval_point from callbacks/surface.py) ────────

def _eval_point(
    N_exp: float,
    t_cad_s: float,
    i_det: int,
    model_day,
    model_night,
    f_live: float,
    f_night: float,
    optical_on: bool,
    approx_on: bool,
    t_overhead_s: float,
    full_integral: bool = False,
    off_axis: bool = False,
) -> tuple[float, float, float, float]:
    """Evaluate (R_det, t_exp_s, q_med, D_med_Gpc) at a single point.

    Mirrors callbacks/surface.py:_eval_point (lines 140-212).
    """
    _nan4 = (math.nan, math.nan, math.nan, math.nan)
    if not (math.isfinite(N_exp) and math.isfinite(t_cad_s) and N_exp > 0 and t_cad_s > 0):
        return _nan4

    # Model dispatch — mirrors compute_surface logic
    if optical_on and model_night is not None and t_cad_s < DAY_S:
        model = model_night
    else:
        model = model_day

    # Rate
    N_arr = np.array([N_exp])
    t_arr = np.array([t_cad_s])
    if full_integral:
        log10R = float(model.rate_log10_full_integral(i_det, N_arr, t_arr)[0])
    else:
        log10R = float(model.rate_log10(i_det, N_arr, t_arr)[0])
    R = 10.0 ** log10R if math.isfinite(log10R) else math.nan
    # Sub-day optical: only the nighttime fraction of detections are accessible
    if optical_on and model_night is not None and t_cad_s < DAY_S:
        R = R * f_night

    # Off-axis correction: subtract on-axis contribution (q < q_dec)
    if off_axis and math.isfinite(R):
        r_on = float(_on_axis_rate_linear(model, i_det, N_arr, t_arr)[0])
        if optical_on and model_night is not None and t_cad_s < DAY_S:
            r_on = r_on * f_night
        R_off = R - r_on if math.isfinite(r_on) else R
        if R_off <= 0:
            return _nan4
        R = R_off

    # t_exp
    try:
        t_exp = float(exposure_time_s(
            SurveyStrategy(N_exp=N_exp, t_cad_s=t_cad_s), model.instrument,
        ))
        if not math.isfinite(t_exp) or t_exp <= 0:
            t_exp = math.nan
    except Exception:
        t_exp = math.nan

    # Approx validity check — same criterion as the surface post-hoc mask
    if approx_on and t_overhead_s > 0:
        if f_live * t_cad_s / N_exp <= t_overhead_s:
            return _nan4

    # Medians — use model_day for consistency with compute_surface
    try:
        qm, dm = model_day.compute_medians(
            i_det, N_arr, t_arr, full_integral=full_integral, off_axis=off_axis,
        )
        q_med     = float(qm[0])
        D_med_Gpc = float(dm[0]) / GPC_TO_CM
    except Exception:
        q_med, D_med_Gpc = math.nan, math.nan

    return R, t_exp, q_med, D_med_Gpc


# ── Discrete-day overlay builder (mirrors components/figures.py:259-379) ────

def _build_day_line_arrays(
    *,
    model_day,
    i_det: int,
    N_cols: np.ndarray,
    t_cad_max_s: float,
    full_integral: bool,
    off_axis: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build flat per-day overlay arrays.

    Returns (day_vals, N_flat, R_flat, rid_flat, t_exp_flat, q_med_flat, D_med_Gpc_flat)
    where each *_flat array has length n_days * n_N in row-major (day, N_exp) order.
    Per-point values outside the display domain (log10R < ZMIN_DISPLAY_LOG10) are set
    to NaN so the JS side can treat them as gaps.

    Mirrors _add_discrete_day_lines in components/figures.py:259-379, but flattens
    the result instead of building Plotly traces.
    """
    N_cols = np.asarray(N_cols, dtype=float)
    n_N = int(N_cols.size)
    empty = np.array([], dtype=float)
    if n_N == 0 or not np.any(np.isfinite(N_cols)):
        return empty, empty, empty, empty, empty, empty, empty

    max_days = int(np.floor(float(t_cad_max_s) / float(DAY_S)))
    if max_days < 1:
        return empty, empty, empty, empty, empty, empty, empty

    n_small = min(30, max_days)
    n_vals_small = np.arange(1, n_small + 1, dtype=int)
    n_vals_large = np.array([], dtype=int)
    if max_days > n_small:
        n_target = 40
        n_vals_large = np.unique(
            np.rint(np.logspace(np.log10(n_small + 1), np.log10(max_days), n_target)).astype(int)
        )
        n_vals_large = n_vals_large[(n_vals_large >= n_small + 1) & (n_vals_large <= max_days)]

    n_vals = np.unique(np.concatenate([n_vals_small, n_vals_large]))
    if n_vals.size == 0:
        return empty, empty, empty, empty, empty, empty, empty

    n_days = int(n_vals.size)
    day_vals = n_vals.astype(float)

    N_line = N_cols[None, :]  # (1, nN)
    N_flat = np.empty((n_days, n_N), dtype=float)
    R_flat = np.empty((n_days, n_N), dtype=float)
    rid_flat = np.full((n_days, n_N), np.nan, dtype=float)
    t_exp_flat = np.empty((n_days, n_N), dtype=float)
    q_med_flat = np.empty((n_days, n_N), dtype=float)
    D_med_Gpc_flat = np.empty((n_days, n_N), dtype=float)

    for i, n in enumerate(n_vals):
        t_s = float(n) * float(DAY_S)
        log10R = _rate(
            model_day, int(i_det), N_line, np.full_like(N_line, t_s),
            full_integral, off_axis=off_axis,
        )
        log10R = np.asarray(log10R).reshape(1, -1).ravel()

        # Match the display-domain gate used for the 3D overlay.
        good = np.isfinite(N_cols) & np.isfinite(log10R) & (log10R >= float(ZMIN_DISPLAY_LOG10))
        R_row = np.where(good, 10.0 ** log10R, np.nan)

        t_arr = np.full_like(N_cols, t_s)
        t_exp_arr = model_day.t_exp_s(N_cols, t_arr)
        q_med_arr, D_med_cm_arr = model_day.compute_medians(
            int(i_det), N_cols, t_arr, full_integral=full_integral, off_axis=off_axis,
        )
        D_med_Gpc_arr = D_med_cm_arr / GPC_TO_CM

        # Regime IDs (always computed; JS can decide whether to colour by them).
        masks = model_day.region_masks(int(i_det), N_line, np.full_like(N_line, t_s),
                                       include_unphysical=False)
        rid_row = np.full(n_N, np.nan, dtype=float)
        for k, key in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
            mk = np.asarray(masks[key]).reshape(1, -1).ravel()
            rid_row[mk] = float(k)

        # Mask t_exp/q/D on non-`good` entries too so JS overlay hover is consistent.
        t_exp_row      = np.where(good, t_exp_arr, np.nan)
        q_med_row      = np.where(good, q_med_arr, np.nan)
        D_med_Gpc_row  = np.where(good, D_med_Gpc_arr, np.nan)
        rid_row        = np.where(good, rid_row, np.nan)

        N_flat[i, :]         = N_cols
        R_flat[i, :]         = R_row
        rid_flat[i, :]       = rid_row
        t_exp_flat[i, :]     = t_exp_row
        q_med_flat[i, :]     = q_med_row
        D_med_Gpc_flat[i, :] = D_med_Gpc_row

    return day_vals, N_flat, R_flat, rid_flat, t_exp_flat, q_med_flat, D_med_Gpc_flat


# ── Shared model builder (used by compute_all, compute_nslice, compute_tslice)

def _build_models(params) -> dict:
    """Parse params and instantiate model_day / model_night plus derived scalars.

    Returns a state dict with every value the main and slice entry points need
    (so no block of setup code is duplicated across the three entry points).
    """
    i_det        = int(params["i_det"])
    A_log        = float(params["A_log"])
    f_live       = float(params["f_live"])
    t_overhead_s = float(params["t_overhead_s"])
    omega_exp    = float(params["omega_exp_deg2"])
    omega_srv    = float(params["omega_srv_deg2"])
    t_night_s    = float(params["t_night_h"]) * 3600.0
    optical_on   = bool(params["optical_survey"])
    color_on     = bool(params["color_regimes"])
    full_on      = bool(params["full_integral"])
    off_axis_on  = bool(params["off_axis"])
    toh_approx   = bool(params.get("toh_approx", False))

    physics_kw = dict(
        p=float(params["p"]),
        nu_log10=float(params["nu_log10"]),
        E_kiso_log10=float(params["E_kiso_log10"]),
        n0_log10=float(params["n0_log10"]),
        epsilon_e_log10=float(params["epsilon_e_log10"]),
        epsilon_B_log10=float(params["epsilon_B_log10"]),
        theta_j_rad=float(params["theta_j_rad"]),
        gamma0_log10=float(params["gamma0_log10"]),
        D_euc_gpc=float(params["D_euc_gpc"]),
        rho_grb_log10=float(params["rho_grb_log10"]),
    )

    # In approx mode, models use t_oh=0; validity boundary enforced post-hoc.
    t_oh_model = 0.0 if toh_approx else t_overhead_s
    design = SurveyDesignParams(omega_survey_max_sr=omega_srv * DEG2_TO_SR)

    model_day = make_rate_model(
        A_log=A_log, f_live=f_live, t_overhead_s=t_oh_model,
        omega_exp_deg2=omega_exp, design=design, **physics_kw,
    )
    model_night = (
        make_rate_model(
            A_log=A_log, f_live=f_live, t_overhead_s=t_oh_model,
            omega_exp_deg2=omega_exp, design=design, **physics_kw,
        )
        if optical_on else None
    )

    f_night   = t_night_s / DAY_S
    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr

    return {
        "i_det":        i_det,
        "f_live":       f_live,
        "t_overhead_s": t_overhead_s,
        "optical_on":   optical_on,
        "color_on":     color_on,
        "full_on":      full_on,
        "off_axis_on":  off_axis_on,
        "toh_approx":   toh_approx,
        "t_night_s":    t_night_s,
        "f_night":      f_night,
        "N_exp_max":    N_exp_max,
        "physics_kw":   physics_kw,
        "design":       design,
        "A_log":        A_log,
        "omega_exp":    omega_exp,
        "model_day":    model_day,
        "model_night":  model_night,
    }


def _cr_kwargs_from_state(state: dict) -> dict:
    """_compute_rate kwargs shared by every slice sweep."""
    return dict(
        full_integral=state["full_on"],
        optical_on=state["optical_on"],
        model_night=state["model_night"],
        approx_on=state["toh_approx"],
        f_live=float(state["f_live"]),
        t_overhead_s=float(state["t_overhead_s"]),
        color_on=state["color_on"],
        off_axis=state["off_axis_on"],
    )


def _compute_nslice_sweep(state: dict, t_cad_fix_s: float) -> dict:
    """N-slice 1-D sweep at user-chosen t_cad. Returns flat payload for JS."""
    i_det        = state["i_det"]
    N_exp_max    = state["N_exp_max"]
    model_day    = state["model_day"]
    model_night  = state["model_night"]
    optical_on   = state["optical_on"]

    N_sweep = np.logspace(0.0, math.log10(N_exp_max), 800)
    t_fixed = np.full_like(N_sweep, float(t_cad_fix_s))
    # Dispatch mirrors _pick_model (callbacks/surface.py:77-81).
    if optical_on and model_night is not None and t_cad_fix_s < DAY_S:
        model_nslice = model_night
    else:
        model_nslice = model_day
    R_n, t_exp_n, q_med_n, D_med_Gpc_n, rid_n = _compute_rate(
        model_nslice, i_det, N_sweep, t_fixed,
        t_cad_scalar=float(t_cad_fix_s),
        f_night_val=state["f_night"], **_cr_kwargs_from_state(state),
    )
    return {
        "N_sweep_flat":           _array_to_list(N_sweep),
        "N_sweep_R_flat":         _array_to_list(R_n),
        "N_sweep_t_exp_flat":     _array_to_list(t_exp_n),
        "N_sweep_q_med_flat":     _array_to_list(q_med_n),
        "N_sweep_D_med_Gpc_flat": _array_to_list(D_med_Gpc_n),
        "N_sweep_regime_flat":    _array_to_list(rid_n),
        "t_cad_fix_s":            float(t_cad_fix_s),
        "t_cad_fix_h":            float(t_cad_fix_s) / 3600.0,
    }


def _compute_tslice_sweep(state: dict, N_fix: float) -> dict:
    """t-slice 1-D sweep(s) at user-chosen N_exp. Returns flat payload for JS."""
    i_det       = state["i_det"]
    model_day   = state["model_day"]
    model_night = state["model_night"]
    optical_on  = state["optical_on"]
    t_night_s   = state["t_night_s"]
    f_night     = state["f_night"]
    cr_kwargs   = _cr_kwargs_from_state(state)

    t_cont_h_flat: list          = []
    t_cont_R_flat: list          = []
    t_cont_t_exp_flat: list      = []
    t_cont_q_med_flat: list      = []
    t_cont_D_med_Gpc_flat: list  = []
    t_cont_regime_flat: list     = []
    t_disc_h_flat: list          = []
    t_disc_R_flat: list          = []
    t_disc_t_exp_flat: list      = []
    t_disc_q_med_flat: list      = []
    t_disc_D_med_Gpc_flat: list  = []
    t_disc_regime_flat: list     = []

    N_fix = float(N_fix)

    if optical_on and model_night is not None:
        # Continuous region (sub-night; force f_night scaling via t_cad_scalar=0.0)
        t_cont_max_s = t_night_s / float(i_det)
        t_cont = np.logspace(
            math.log10(max(10.0, 1.0)),
            math.log10(max(11.0, t_cont_max_s * 0.999)),
            600,
        )
        N_cont = np.full_like(t_cont, N_fix)
        R_c, t_exp_c, q_med_c, D_med_Gpc_c, rid_c = _compute_rate(
            model_night, i_det, N_cont, t_cont,
            t_cad_scalar=0.0,
            f_night_val=f_night, **cr_kwargs,
        )
        t_cont_h_flat         = _array_to_list(t_cont / 3600.0)
        t_cont_R_flat         = _array_to_list(R_c)
        t_cont_t_exp_flat     = _array_to_list(t_exp_c)
        t_cont_q_med_flat     = _array_to_list(q_med_c)
        t_cont_D_med_Gpc_flat = _array_to_list(D_med_Gpc_c)
        t_cont_regime_flat    = _array_to_list(rid_c)

        # Discrete region (integer days; no f_night scaling via t_cad_scalar=DAY_S)
        n_max_days = min(500, int(np.floor(1e8 / DAY_S)))
        t_disc = np.arange(1, n_max_days + 1, dtype=float) * DAY_S
        N_disc = np.full_like(t_disc, N_fix)
        R_d, t_exp_d, q_med_d, D_med_Gpc_d, rid_d = _compute_rate(
            model_day, i_det, N_disc, t_disc,
            t_cad_scalar=float(DAY_S),
            f_night_val=f_night, **cr_kwargs,
        )
        t_disc_h_flat         = _array_to_list(t_disc / 3600.0)
        t_disc_R_flat         = _array_to_list(R_d)
        t_disc_t_exp_flat     = _array_to_list(t_exp_d)
        t_disc_q_med_flat     = _array_to_list(q_med_d)
        t_disc_D_med_Gpc_flat = _array_to_list(D_med_Gpc_d)
        t_disc_regime_flat    = _array_to_list(rid_d)
    else:
        # Non-optical: single logspace sweep, 1500 points, no f_night scaling.
        t_sweep = np.logspace(0.0, 8.0, 1500)
        N_fixed = np.full_like(t_sweep, N_fix)
        R_s, t_exp_s, q_med_s, D_med_Gpc_s, rid_s = _compute_rate(
            model_day, i_det, N_fixed, t_sweep,
            t_cad_scalar=float(DAY_S),
            f_night_val=f_night, **cr_kwargs,
        )
        t_cont_h_flat         = _array_to_list(t_sweep / 3600.0)
        t_cont_R_flat         = _array_to_list(R_s)
        t_cont_t_exp_flat     = _array_to_list(t_exp_s)
        t_cont_q_med_flat     = _array_to_list(q_med_s)
        t_cont_D_med_Gpc_flat = _array_to_list(D_med_Gpc_s)
        t_cont_regime_flat    = _array_to_list(rid_s)
        # t_disc_* stays empty (non-optical has no discrete region)

    return {
        "t_cont_h_flat":          t_cont_h_flat,
        "t_cont_R_flat":          t_cont_R_flat,
        "t_cont_t_exp_flat":      t_cont_t_exp_flat,
        "t_cont_q_med_flat":      t_cont_q_med_flat,
        "t_cont_D_med_Gpc_flat":  t_cont_D_med_Gpc_flat,
        "t_cont_regime_flat":     t_cont_regime_flat,
        "t_disc_h_flat":          t_disc_h_flat,
        "t_disc_R_flat":          t_disc_R_flat,
        "t_disc_t_exp_flat":      t_disc_t_exp_flat,
        "t_disc_q_med_flat":      t_disc_q_med_flat,
        "t_disc_D_med_Gpc_flat":  t_disc_D_med_Gpc_flat,
        "t_disc_regime_flat":     t_disc_regime_flat,
        "N_fix":                  N_fix,
    }


# ── Slice-only entry points (JS calls these on slice-slider drag) ───────────

def compute_nslice(params, t_cad_fix_s) -> dict:
    """Return an N-slice payload only. Called from JS when the user drags the
    nslice-tfix-slider — avoids a full surface+optimizer recompute.
    """
    try:
        state = _build_models(params)
        payload = _compute_nslice_sweep(state, float(t_cad_fix_s))
        payload["error"] = None
        return payload
    except Exception:
        import traceback
        return {"error": traceback.format_exc()}


def compute_tslice(params, N_fix) -> dict:
    """Return a t-slice payload only. Called from JS when the user drags the
    tslice-nfix-slider — avoids a full surface+optimizer recompute.
    """
    try:
        state = _build_models(params)
        payload = _compute_tslice_sweep(state, float(N_fix))
        payload["error"] = None
        return payload
    except Exception:
        import traceback
        return {"error": traceback.format_exc()}


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_all(params) -> dict:
    """Compute the detection-rate surface and derived metrics.

    Called from JS via pyodide: result = compute_all(pyodide.toPy(jsParams)).
    The return dict contains flat Python lists; call result.toJs() on the JS side.
    """
    try:
        state = _build_models(params)
        i_det        = state["i_det"]
        f_live       = state["f_live"]
        t_overhead_s = state["t_overhead_s"]
        optical_on   = state["optical_on"]
        color_on     = state["color_on"]
        full_on      = state["full_on"]
        off_axis_on  = state["off_axis_on"]
        toh_approx   = state["toh_approx"]
        t_night_s    = state["t_night_s"]
        f_night      = state["f_night"]
        N_exp_max    = state["N_exp_max"]
        model_day    = state["model_day"]
        model_night  = state["model_night"]

        # Grid resolution — match Dash (callbacks/surface.py:311-312)
        nx = NX_REGIME if color_on else NX_DEFAULT
        ny = NY_REGIME if color_on else NY_DEFAULT

        # ── Surface ──────────────────────────────────────────────────────────
        X, Y_s, Z_plot, Z_raw, regime_id, t_exp_g, q_med_g, D_med_Gpc_g = compute_surface(
            model_day, model_night, i_det,
            optical_survey=optical_on, color_regimes=color_on,
            t_night_s=t_night_s, nx=nx, ny=ny,
            full_integral=full_on, off_axis=off_axis_on,
        )

        if toh_approx and t_overhead_s > 0:
            budget = f_live * Y_s / X
            invalid = budget <= t_overhead_s
            Z_plot = np.where(invalid, np.nan, Z_plot)
            Z_raw = np.where(invalid, np.nan, Z_raw)
            if regime_id is not None:
                regime_id = np.where(invalid, np.nan, regime_id)
            t_exp_g     = np.where(invalid, np.nan, t_exp_g)
            q_med_g     = np.where(invalid, np.nan, q_med_g)
            D_med_Gpc_g = np.where(invalid, np.nan, D_med_Gpc_g)

        Y_h   = Y_s / 3600.0
        R_lin = np.where(np.isfinite(Z_plot), 10.0 ** Z_plot, np.nan)

        # ── Optimizer ────────────────────────────────────────────────────────
        # Validity constraint for the approx-mode optimizer — see
        # callbacks/surface.py:345-349.
        opt_validity_fn = None
        if toh_approx and t_overhead_s > 0:
            _f0, _toh0 = float(f_live), float(t_overhead_s)
            def opt_validity_fn(N_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
                return _f0 * t_arr / N_arr > _toh0

        N_opt, t_cad_opt_s, log10R_opt = maximize_log_surface_iterative(
            model_day, model_night, i_det,
            x_min=0.0, x_max=math.log10(N_exp_max),
            y_min=0.0, y_max=8.0,
            optical_survey=optical_on, t_night_s=t_night_s,
            full_integral=full_on, off_axis=off_axis_on,
            validity_fn=opt_validity_fn,
        )
        R_opt, t_exp_opt_s, q_med_opt, D_med_Gpc_opt = _eval_point(
            N_opt, t_cad_opt_s, i_det, model_day, model_night,
            f_live, f_night, optical_on, toh_approx, t_overhead_s,
            full_integral=full_on, off_axis=off_axis_on,
        )

        # ── ZTF reference ────────────────────────────────────────────────────
        N_ztf = min(OMEGA_SRV_DEFAULT_DEG2 / ZTF_OMEGA_EXP_DEG2, N_exp_max)
        t_cad_ztf_s = 2.0 * DAY_S
        R_ztf, t_exp_ztf_s, q_med_ztf, D_med_Gpc_ztf = _eval_point(
            N_ztf, t_cad_ztf_s, i_det, model_day, model_night,
            f_live, f_night, optical_on, toh_approx, t_overhead_s,
            full_integral=full_on, off_axis=off_axis_on,
        )

        # ── Slice sweeps at user-chosen positions ────────────────────────────
        # JS passes log10 values from the slice-position sliders. If absent (page
        # bootstrap or very-old callers), fall back to the optimum so the initial
        # render matches the pre-slider behaviour.
        default_tfix_log = math.log10(t_cad_opt_s) if math.isfinite(t_cad_opt_s) and t_cad_opt_s > 0 else math.log10(DAY_S)
        default_nfix_log = math.log10(N_opt) if math.isfinite(N_opt) and N_opt > 0 else 2.0
        nslice_tfix_log = float(params.get("nslice_tfix_log", default_tfix_log))
        tslice_nfix_log = float(params.get("tslice_nfix_log", default_nfix_log))
        t_cad_fix_s = 10.0 ** nslice_tfix_log
        N_fix       = 10.0 ** tslice_nfix_log

        nslice_payload = _compute_nslice_sweep(state, t_cad_fix_s)
        tslice_payload = _compute_tslice_sweep(state, N_fix)

        # ── Derived GRB counts ───────────────────────────────────────────────
        rho      = 10.0 ** float(params["rho_grb_log10"])
        D_gpc    = float(params["D_euc_gpc"])
        theta_j  = float(params["theta_j_rad"])
        R_int_yr = rho * (4.0 / 3.0) * math.pi * D_gpc ** 3
        f_b      = theta_j ** 2 / 2.0
        R_toward_day = R_int_yr * f_b / 365.25

        zmax_log10 = float(np.nanmax(Z_plot)) if np.any(np.isfinite(Z_plot)) else 0.0

        # ── Gap times (optical t-slice; see callbacks/surface.py:501) ────────
        if optical_on:
            gap_lo_h = (t_night_s / 3600.0) / float(i_det)
            gap_hi_h = 24.0
        else:
            gap_lo_h = None
            gap_hi_h = None

        # ── Discrete-day overlay payload (optical only) ──────────────────────
        if optical_on:
            N_cols = np.asarray(X[0, :], dtype=float)
            t_cad_max_s = float(np.nanmax(Y_s[:, 0])) if np.any(np.isfinite(Y_s[:, 0])) else 0.0
            (
                day_vals,
                day_N, day_R, day_rid,
                day_t_exp, day_q_med, day_D_med_Gpc,
            ) = _build_day_line_arrays(
                model_day=model_day, i_det=i_det,
                N_cols=N_cols, t_cad_max_s=t_cad_max_s,
                full_integral=full_on, off_axis=off_axis_on,
            )
            n_days, n_N = (int(day_vals.size),
                           int(day_N.shape[1]) if day_N.ndim == 2 and day_vals.size > 0 else 0)
            day_line_shape = [n_days, n_N]
            day_line_t_cad_days = [float(v) for v in day_vals.tolist()]
            day_line_N_flat          = _array_to_list(day_N)
            day_line_R_flat          = _array_to_list(day_R)
            day_line_regime_flat     = _array_to_list(day_rid)
            day_line_t_exp_flat      = _array_to_list(day_t_exp)
            day_line_q_med_flat      = _array_to_list(day_q_med)
            day_line_D_med_Gpc_flat  = _array_to_list(day_D_med_Gpc)
        else:
            day_line_shape = [0, 0]
            day_line_t_cad_days = []
            day_line_N_flat = []
            day_line_R_flat = []
            day_line_regime_flat = []
            day_line_t_exp_flat = []
            day_line_q_med_flat = []
            day_line_D_med_Gpc_flat = []

        return {
            "X_flat":       _array_to_list(X),
            "Y_flat":       _array_to_list(Y_h),
            "Z_flat":       _array_to_list(R_lin),
            "Z_log_flat":   _array_to_list(Z_plot),
            "regime_flat":  _array_to_list(regime_id) if regime_id is not None else None,
            "t_exp_flat":       _array_to_list(t_exp_g),
            "q_med_flat":       _array_to_list(q_med_g),
            "D_med_Gpc_flat":   _array_to_list(D_med_Gpc_g),
            "shape":        list(X.shape),
            "N_opt":        _nan_to_none(N_opt),
            "t_cad_opt_h":  _nan_to_none(t_cad_opt_s / 3600.0 if math.isfinite(t_cad_opt_s) else math.nan),
            "t_cad_opt_s":  _nan_to_none(t_cad_opt_s),
            "R_opt":        _nan_to_none(R_opt),
            "log10R_opt":   _nan_to_none(log10R_opt),
            "t_exp_opt_s":  _nan_to_none(t_exp_opt_s),
            "q_med_opt":       _nan_to_none(q_med_opt),
            "D_med_Gpc_opt":   _nan_to_none(D_med_Gpc_opt),
            "N_ztf":        float(N_ztf),
            "t_cad_ztf_h":  float(t_cad_ztf_s / 3600.0),
            "t_cad_ztf_s":  float(t_cad_ztf_s),
            "R_ztf":        _nan_to_none(R_ztf),
            "t_exp_ztf_s":  _nan_to_none(t_exp_ztf_s),
            "q_med_ztf":       _nan_to_none(q_med_ztf),
            "D_med_Gpc_ztf":   _nan_to_none(D_med_Gpc_ztf),
            "zmax_log10":   zmax_log10,
            "R_int_yr":     float(R_int_yr),
            "R_toward_day": float(R_toward_day),
            "N_exp_max":    float(N_exp_max),
            "gap_lo_h":     _nan_to_none(gap_lo_h) if gap_lo_h is not None else None,
            "gap_hi_h":     _nan_to_none(gap_hi_h) if gap_hi_h is not None else None,
            "day_line_t_cad_days":     day_line_t_cad_days,
            "day_line_N_flat":         day_line_N_flat,
            "day_line_R_flat":         day_line_R_flat,
            "day_line_regime_flat":    day_line_regime_flat,
            "day_line_t_exp_flat":     day_line_t_exp_flat,
            "day_line_q_med_flat":     day_line_q_med_flat,
            "day_line_D_med_Gpc_flat": day_line_D_med_Gpc_flat,
            "day_line_shape":          day_line_shape,
            # ── Dedicated slice sweeps (at user-chosen slice positions) ────
            "N_sweep_flat":            nslice_payload["N_sweep_flat"],
            "N_sweep_R_flat":          nslice_payload["N_sweep_R_flat"],
            "N_sweep_t_exp_flat":      nslice_payload["N_sweep_t_exp_flat"],
            "N_sweep_q_med_flat":      nslice_payload["N_sweep_q_med_flat"],
            "N_sweep_D_med_Gpc_flat":  nslice_payload["N_sweep_D_med_Gpc_flat"],
            "N_sweep_regime_flat":     nslice_payload["N_sweep_regime_flat"],
            "t_cad_fix_s":             nslice_payload["t_cad_fix_s"],
            "t_cad_fix_h":             nslice_payload["t_cad_fix_h"],
            "t_cont_h_flat":           tslice_payload["t_cont_h_flat"],
            "t_cont_R_flat":           tslice_payload["t_cont_R_flat"],
            "t_cont_t_exp_flat":       tslice_payload["t_cont_t_exp_flat"],
            "t_cont_q_med_flat":       tslice_payload["t_cont_q_med_flat"],
            "t_cont_D_med_Gpc_flat":   tslice_payload["t_cont_D_med_Gpc_flat"],
            "t_cont_regime_flat":      tslice_payload["t_cont_regime_flat"],
            "t_disc_h_flat":           tslice_payload["t_disc_h_flat"],
            "t_disc_R_flat":           tslice_payload["t_disc_R_flat"],
            "t_disc_t_exp_flat":       tslice_payload["t_disc_t_exp_flat"],
            "t_disc_q_med_flat":       tslice_payload["t_disc_q_med_flat"],
            "t_disc_D_med_Gpc_flat":   tslice_payload["t_disc_D_med_Gpc_flat"],
            "t_disc_regime_flat":      tslice_payload["t_disc_regime_flat"],
            "N_fix":                   tslice_payload["N_fix"],
            "error":        None,
        }

    except Exception:
        import traceback
        return {"error": traceback.format_exc(), "X_flat": None, "shape": None}
