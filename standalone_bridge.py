"""Standalone bridge for the GRB Detection Rate Pyodide HTML app.

Single public function: compute_all(params) -> dict.
All surface arrays are returned as flat Python lists (row-major), NaN replaced by
None so that Pyodide's .toJs() produces JS null, which Plotly treats as a gap.
"""
from __future__ import annotations

import copy         # TEMP-FDEC-OVERRIDE
import dataclasses  # TEMP-FDEC-OVERRIDE
import math

import numpy as np

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.params import GPC_TO_CM, SurveyDesignParams, SurveyStrategy
from grb_detect.core import (
    ZMIN_DISPLAY_LOG10,
    _rate,
    compute_surface,
    make_rate_model,
    maximize_log_surface_iterative,
    optical_survey_tcad_seconds,
)
from grb_detect.survey import exposure_time_s

ZTF_OMEGA_EXP_DEG2: float = 47.0

# The two real ZTF observing modes used as reference points on the surface
# (Ho et al. 2022; Andreoni et al. 2021):
#   public all-sky  — ~15,000 deg² every 2 nights (g+r),
#   high-cadence    — ~2,500 deg² partnership/ZUDS, 6 visits per night.
ZTF_PUBLIC_OMEGA_SRV_DEG2: float = 15000.0
ZTF_PUBLIC_T_CAD_S: float = 2.0 * DAY_S
ZTF_HC_OMEGA_SRV_DEG2: float = 2500.0
ZTF_HC_VISITS_PER_NIGHT: int = 6

# Grid resolutions. Regime-colour mode uses the denser grid so the discrete
# boundaries between regimes stay crisp.
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


# ── Regime-id helper ────────────────────────────────────────────────────────

def _masks_1d(masks: dict, n: int) -> np.ndarray:
    rid = np.full(n, np.nan, dtype=float)
    for k, key in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
        mk = np.asarray(masks[key]).ravel()
        if mk.size == n:
            rid[mk] = float(k)
    return rid


# ── 1-D rate sweep ──────────────────────────────────────────────────────────

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
    f_live_night: float,
    t_overhead_s: float,
    color_on: bool,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_fade: float = 0.0,
    s_rise: float = 0.0,
    s_mode: str = "discrete",
    rise_random_start: bool = True,
    fade_random_start: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (R, t_exp, q_med, D_med_Gpc, rid) for a 1-D sweep."""
    if full_integral:
        Z = model.rate_log10_full_integral(
            i_det, N_arr, t_arr,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )
    else:
        Z = model.rate_log10(
            i_det, N_arr, t_arr,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )
    R = np.where(np.isfinite(Z), 10.0 ** Z, np.nan)

    # Optical sub-day branch: rate is reduced by the night-accessibility fraction.
    # The same predicate decides which f_live the validity boundary uses, since
    # model_night was constructed with f_live_eff = f_live / f_night.
    is_subday_optical = optical_on and model_night is not None and t_cad_scalar < DAY_S
    if is_subday_optical:
        R = R * f_night_val
    f_live_validity = float(f_live_night) if is_subday_optical else float(f_live)

    if approx_on and float(t_overhead_s) > 0:
        R = np.where(f_live_validity * t_arr / N_arr <= float(t_overhead_s), np.nan, R)

    rid = np.full(len(N_arr), np.nan)
    if color_on:
        masks = model.region_masks(i_det, N_arr, t_arr, include_unphysical=False)
        rid = _masks_1d(masks, len(N_arr))

    q_med, D_med_cm = model.compute_medians(
        i_det, N_arr, t_arr, full_integral=full_integral,
        q_min=q_min, D_min_cm=D_min_cm,
        s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
        rise_random_start=rise_random_start,
        fade_random_start=fade_random_start,
    )
    t_exp = model.t_exp_s(N_arr, t_arr)
    D_med_Gpc = D_med_cm / GPC_TO_CM

    if approx_on and float(t_overhead_s) > 0:
        _inv = f_live_validity * t_arr / N_arr <= float(t_overhead_s)
        t_exp     = np.where(_inv, np.nan, t_exp)
        q_med     = np.where(_inv, np.nan, q_med)
        D_med_Gpc = np.where(_inv, np.nan, D_med_Gpc)

    return R, t_exp, q_med, D_med_Gpc, rid


# ── Point evaluation ────────────────────────────────────────────────────────

def _eval_point(
    N_exp: float,
    t_cad_s: float,
    i_det: int,
    model_day,
    model_night,
    f_live: float,
    f_live_night: float,
    f_night: float,
    optical_on: bool,
    approx_on: bool,
    t_overhead_s: float,
    full_integral: bool = False,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_fade: float = 0.0,
    s_rise: float = 0.0,
    s_mode: str = "discrete",
    rise_random_start: bool = True,
    fade_random_start: bool = True,
) -> tuple[float, float, float, float]:
    """Evaluate (R_det, t_exp_s, q_med, D_med_Gpc) at a single point."""
    _nan4 = (math.nan, math.nan, math.nan, math.nan)
    if not (math.isfinite(N_exp) and math.isfinite(t_cad_s) and N_exp > 0 and t_cad_s > 0):
        return _nan4

    # Model dispatch — mirrors compute_surface logic
    if optical_on and model_night is not None and t_cad_s < DAY_S:
        model = model_night
    else:
        model = model_day

    # Rate (filter applied inside the rate method, matching compute_surface)
    N_arr = np.array([N_exp])
    t_arr = np.array([t_cad_s])
    if full_integral:
        log10R = float(model.rate_log10_full_integral(
            i_det, N_arr, t_arr,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start)[0])
    else:
        log10R = float(model.rate_log10(
            i_det, N_arr, t_arr,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start)[0])
    R = 10.0 ** log10R if math.isfinite(log10R) else math.nan
    # Sub-day optical: only the nighttime fraction of detections are accessible
    if optical_on and model_night is not None and t_cad_s < DAY_S:
        R = R * f_night

    if not math.isfinite(R) or R <= 0:
        return _nan4

    # t_exp
    try:
        t_exp = float(exposure_time_s(
            SurveyStrategy(N_exp=N_exp, t_cad_s=t_cad_s), model.instrument,
        ))
        if not math.isfinite(t_exp) or t_exp <= 0:
            t_exp = math.nan
    except Exception:
        t_exp = math.nan

    # Approx validity check — same criterion as the surface post-hoc mask.
    # Sub-day optical uses the rescaled f_live (= f_live / f_night).
    f_live_validity = (
        f_live_night
        if (optical_on and model_night is not None and t_cad_s < DAY_S)
        else f_live
    )
    if approx_on and t_overhead_s > 0:
        if f_live_validity * t_cad_s / N_exp <= t_overhead_s:
            return _nan4

    # Medians — same model dispatch as the rate (and compute_surface): on the
    # sub-day optical branch model_night carries f_live = f_live / f_night, so
    # only it describes the detected population behind the reported rate.
    try:
        qm, dm = model.compute_medians(
            i_det, N_arr, t_arr, full_integral=full_integral,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )
        q_med     = float(qm[0])
        D_med_Gpc = float(dm[0]) / GPC_TO_CM
    except Exception:
        q_med, D_med_Gpc = math.nan, math.nan

    return R, t_exp, q_med, D_med_Gpc


# ── Discrete-day overlay builder ────────────────────────────────────────────

def _build_day_line_arrays(
    *,
    model_day,
    i_det: int,
    N_cols: np.ndarray,
    t_cad_max_s: float,
    full_integral: bool,
    q_min: float = 0.0,
    D_min_cm: float = 0.0,
    s_fade: float = 0.0,
    s_rise: float = 0.0,
    s_mode: str = "discrete",
    rise_random_start: bool = True,
    fade_random_start: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build flat per-day overlay arrays.

    Returns (day_vals, N_flat, R_flat, rid_flat, t_exp_flat, q_med_flat, D_med_Gpc_flat)
    where each *_flat array has length n_days * n_N in row-major (day, N_exp) order.
    Per-point values outside the display domain (log10R < ZMIN_DISPLAY_LOG10) are set
    to NaN so the JS side can treat them as gaps.
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
            full_integral, q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )
        log10R = np.asarray(log10R).reshape(1, -1).ravel()

        # Match the display-domain gate used for the 3D overlay.
        good = np.isfinite(N_cols) & np.isfinite(log10R) & (log10R >= float(ZMIN_DISPLAY_LOG10))
        R_row = np.where(good, 10.0 ** log10R, np.nan)

        t_arr = np.full_like(N_cols, t_s)
        t_exp_arr = model_day.t_exp_s(N_cols, t_arr)
        q_med_arr, D_med_cm_arr = model_day.compute_medians(
            int(i_det), N_cols, t_arr, full_integral=full_integral,
            q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
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


# ── TEMP-FDEC-OVERRIDE — begin ───────────────────────────────────────────────
# Inlined equivalent of figures/figlib/overrides.set_F_dec (figures/ is not
# bundled into the Pyodide zip — build_standalone.py zips only grb_detect/ +
# this file). Copy-on-write is mandatory: make_rate_model is lru_cached;
# mutating the cached instance would corrupt every other holder. Scaling
# F_dec/F_j/F_nr by the same factor k is exactly equivalent to a native
# rebuild with A_log -= log10(k) (pinned by tests/test_fdec_override.py).
def _apply_F_dec_override(model, F_dec_Jy: float):
    d = model.derived
    F_cur = float(d.F_dec_Jy)
    if not (F_cur > 0.0):
        return model
    k = float(F_dec_Jy) / F_cur
    m = copy.copy(model)
    m._derived = dataclasses.replace(
        d, F_dec_Jy=d.F_dec_Jy * k, F_j_Jy=d.F_j_Jy * k, F_nr_Jy=d.F_nr_Jy * k)
    return m
# ── TEMP-FDEC-OVERRIDE — end ─────────────────────────────────────────────────


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
    q_min        = float(params.get("qmin", 0.0) or 0.0)
    D_min_cm     = float(params.get("Dmin_cm", 0.0) or 0.0)
    s_fade        = float(params.get("s_fade", 0.0) or 0.0)
    s_rise        = float(params.get("s_rise", 0.0) or 0.0)
    _s_mode_raw  = params.get("s_mode", "discrete")
    s_mode       = str(_s_mode_raw) if _s_mode_raw in ("discrete", "continuous") else "discrete"
    # Per-cut random-start treatment inside the full-integral path: True =
    # uniform-start survival weight, False = best-case hard boundary. Default
    # True preserves the pre-toggle behavior; inert when full_integral is off.
    rise_random_start = bool(params.get("rise_random_start", True))
    fade_random_start = bool(params.get("fade_random_start", True))
    toh_approx   = bool(params.get("toh_approx", False))
    win_iminus1  = bool(params.get("win_iminus1", False))
    win_tp       = bool(params.get("win_tp", False))

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

    f_night = t_night_s / DAY_S
    # Continuous (sub-night) optical branch: within-night live fraction is
    # f_live · 86400 / t_night = f_live / f_night. The UI keeps f_night ≥ f_live
    # via a dynamic t_night floor, but we still guard against a zero denominator.
    f_live_night = f_live / max(f_night, 1e-12)

    model_day = make_rate_model(
        A_log=A_log, f_live=f_live, t_overhead_s=t_oh_model,
        omega_exp_deg2=omega_exp, design=design,
        win_i_minus_one=win_iminus1, win_from_peak=win_tp, **physics_kw,
    )
    model_night = (
        make_rate_model(
            A_log=A_log, f_live=f_live_night, t_overhead_s=t_oh_model,
            omega_exp_deg2=omega_exp, design=design,
            win_i_minus_one=win_iminus1, win_from_peak=win_tp, **physics_kw,
        )
        if optical_on else None
    )

    # TEMP-FDEC-OVERRIDE — begin
    _fdec_ov = params.get("F_dec_override_Jy", None)   # absent or JS null → None
    fdec_override_applied = False
    if _fdec_ov is not None:
        _fdec_ov = float(_fdec_ov)
        if math.isfinite(_fdec_ov) and _fdec_ov > 0.0:
            model_day = _apply_F_dec_override(model_day, _fdec_ov)
            if model_night is not None:
                model_night = _apply_F_dec_override(model_night, _fdec_ov)
            fdec_override_applied = True
    # TEMP-FDEC-OVERRIDE — end

    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr

    return {
        "i_det":        i_det,
        "f_live":       f_live,
        "f_live_night": f_live_night,
        "t_overhead_s": t_overhead_s,
        "optical_on":   optical_on,
        "color_on":     color_on,
        "full_on":      full_on,
        "q_min":        q_min,
        "D_min_cm":     D_min_cm,
        "s_fade":        s_fade,
        "s_rise":        s_rise,
        "s_mode":       s_mode,
        "rise_random_start": rise_random_start,
        "fade_random_start": fade_random_start,
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
        "fdec_override_applied": fdec_override_applied,  # TEMP-FDEC-OVERRIDE
    }


def _cr_kwargs_from_state(state: dict) -> dict:
    """_compute_rate kwargs shared by every slice sweep."""
    return dict(
        full_integral=state["full_on"],
        optical_on=state["optical_on"],
        model_night=state["model_night"],
        approx_on=state["toh_approx"],
        f_live=float(state["f_live"]),
        f_live_night=float(state["f_live_night"]),
        t_overhead_s=float(state["t_overhead_s"]),
        color_on=state["color_on"],
        q_min=state["q_min"],
        D_min_cm=state["D_min_cm"],
        s_fade=state["s_fade"],
        s_rise=state["s_rise"],
        s_mode=state["s_mode"],
        rise_random_start=state["rise_random_start"],
        fade_random_start=state["fade_random_start"],
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
    # Optical sub-day cadences use the night model; everything else uses day.
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


def _compute_qdview_sweep(
    state: dict, N_exp_fix: float, t_cad_fix_s: float
) -> dict:
    """Joint R(q) + R(D) sweep at fixed strategy (N_exp, t_cad).

    Returns flat lists for the q-curve and D-curve, each providing both the
    cumulative and the differential representation (the JS toggle only re-renders;
    no recompute). Mirrors `_compute_nslice_sweep`'s model dispatch, f_night
    scaling and t_OH validity treatment.
    """
    i_det        = state["i_det"]
    model_day    = state["model_day"]
    model_night  = state["model_night"]
    optical_on   = state["optical_on"]
    full_on      = state["full_on"]
    approx_on    = state["toh_approx"]
    f_live_val   = float(state["f_live"])
    f_live_night = float(state["f_live_night"])
    f_night      = float(state["f_night"])
    t_overhead_s = float(state["t_overhead_s"])
    q_min        = float(state["q_min"])
    D_min_cm     = float(state["D_min_cm"])
    s_fade        = float(state["s_fade"])
    s_rise        = float(state["s_rise"])
    s_mode       = str(state["s_mode"])
    rise_random_start = bool(state["rise_random_start"])
    fade_random_start = bool(state["fade_random_start"])

    N_exp_fix   = float(N_exp_fix)
    t_cad_fix_s = float(t_cad_fix_s)

    is_subday_optical = (
        optical_on and model_night is not None and t_cad_fix_s < DAY_S
    )
    model = model_night if is_subday_optical else model_day
    f_night_mul     = f_night if is_subday_optical else 1.0
    f_live_validity = f_live_night if is_subday_optical else f_live_val
    # win_from_peak has no closed dominant-term curves (q-dependent D_eff) —
    # the R(q)/R(D) views fall back to the full-integral branch for it.
    use_full = full_on or bool(getattr(model, "win_from_peak", False))

    # Geometry the JS render layer always needs (drawn even on invalid payloads).
    q_dec_val = float(model.derived.q_dec)
    q_j_val   = float(model.derived.q_j)
    q_nr_val  = float(model.derived.q_nr)
    D_Euc_cm  = float(model.phys.D_euc_cm)
    D_Euc_Gpc = D_Euc_cm / GPC_TO_CM

    N_q_view = 150
    N_D_view = 100
    # Log-spaced view grids match the log x-axes in the renderer. The lower
    # bounds are well below any physically relevant q / D, so cumulative
    # endpoints at the leftmost grid point are R_total to ≤ 1e-4 relative error.
    q_grid     = np.logspace(
        math.log10(max(q_nr_val / 200.0, 1e-3)),
        math.log10(q_nr_val),
        N_q_view,
    )
    D_grid_cm  = np.logspace(
        math.log10(D_Euc_cm / 1000.0),
        math.log10(D_Euc_cm),
        N_D_view,
    )
    D_grid_Gpc = D_grid_cm / GPC_TO_CM

    def _empty_payload() -> dict:
        nans_q = [None] * N_q_view
        nans_D = [None] * N_D_view
        return {
            "qdview_q_grid_flat":       _array_to_list(q_grid),
            "qdview_Rq_cum_flat":       nans_q,
            "qdview_Rq_diff_flat":      nans_q,
            "qdview_D_grid_Gpc_flat":   _array_to_list(D_grid_Gpc),
            "qdview_RD_cum_flat":       nans_D,
            "qdview_RD_diff_flat":      nans_D,
            "qdview_q_dec":             q_dec_val,
            "qdview_q_j":               q_j_val,
            "qdview_q_nr":              q_nr_val,
            "qdview_D_Euc_Gpc":         D_Euc_Gpc,
            "qdview_qmin_sidebar":      q_min,
            "qdview_Dmin_Gpc_sidebar":  D_min_cm / GPC_TO_CM,
            "qdview_total_rate_q":      None,
            "qdview_total_rate_D":      None,
            "qdview_regime_id":         None,
            "qdview_N_fix":             N_exp_fix,
            "qdview_t_cad_fix_s":       t_cad_fix_s,
            "qdview_t_cad_fix_h":       t_cad_fix_s / 3600.0,
        }

    # t_OH approximation validity — same predicate as in _compute_rate.
    if approx_on and t_overhead_s > 0:
        if f_live_validity * t_cad_fix_s / N_exp_fix <= t_overhead_s:
            return _empty_payload()

    # rate_log10 with return_components gives us the active regime + the scalars
    # needed for the dominant-term R(q)/R(D) closed forms (q_E, q_i, D_dec, D_i, fO).
    # We apply the D_min and fading-rate cross-filters here so the regime selection
    # and fO are in the same context as the q-view, but the q_min sweep happens
    # analytically below.
    N_arr = np.array([N_exp_fix])
    t_arr = np.array([t_cad_fix_s])
    _, comps = model.rate_log10(
        i_det, N_arr, t_arr,
        q_min=0.0, D_min_cm=D_min_cm,
        s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
        return_components=True,
    )

    regime_keys = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
    active = None
    for k in regime_keys:
        if bool(np.asarray(comps[k]).ravel()[0]):
            active = k
            break
    if active is None:
        return _empty_payload()

    regime_id_val = float(regime_keys.index(active) + 1)
    fO_val    = float(np.asarray(comps["f_Omega"]).ravel()[0])
    qE_val    = float(np.asarray(comps["q_Euc"]).ravel()[0])
    qi_val    = float(np.asarray(comps["q_i"]).ravel()[0])
    D_dec_val = float(np.asarray(comps["D_dec_cm"]).ravel()[0])
    D_i_val   = float(np.asarray(comps["D_i_cm"]).ravel()[0])

    theta_j = float(model.phys.theta_j_rad)
    R_int   = float(model.phys.R_int_yr)
    base    = 0.5 * fO_val * (theta_j ** 2) * R_int

    q_max_map = {
        "A1": q_nr_val,  "A2": qE_val,    "A3": qE_val,
        "A4": q_nr_val,  "A5": qi_val,    "A6": qi_val,
        "A7": q_dec_val,
    }
    q_max_val_orig = q_max_map[active]

    # Apply the fading-rate cap to the active regime's q_max.  Phase mapping
    # mirrors `rate_log10`: A1/A2/A4/A5 → Phase III, A3/A6/A7 → Phase II.
    q_s_II_arr, q_s_III_arr = model._q_s_fading_caps(
        i_det, np.array([t_cad_fix_s]), s_fade, s_mode,
    )
    q_s_II  = float(q_s_II_arr[0])
    q_s_III = float(q_s_III_arr[0])
    phase_for_regime = {
        "A1": q_s_III, "A2": q_s_III, "A3": q_s_II,
        "A4": q_s_III, "A5": q_s_III, "A6": q_s_II,
        "A7": q_s_II,
    }

    # Rise-rate boundaries (mirrors `rate_log10`): qE_r for the flux-limited
    # A1-A3, q_ri for the cadence-limited A4-A6, D_dec·η^{−1/2} for A7.  The
    # dominant view keeps the separable regime form (single capped q_max and
    # regime-constant D_eff), so the deep-cut on-axis max-term of the rate
    # surface is not reproduced here — use full-integral mode for that corner.
    rise_cap = float("inf")
    D_dec_r_cm = float("inf")
    if s_rise > 0.0:
        F_lim_val = float(np.asarray(comps["F_lim_Jy"]).ravel()[0])
        eta_a, ise_a = model._rise_eta(np.array([t_cad_fix_s]), s_rise)
        with np.errstate(over="ignore", invalid="ignore"):
            qE_r = float(model.q_Euc(np.array([eta_a[0] * F_lim_val]))[0])
            q_ri = float(model.q_Euc(np.array(
                [eta_a[0] * F_lim_val * (D_i_val / D_Euc_cm) ** 2]))[0])
        D_dec_r_cm = D_dec_val * float(ise_a[0])
        rise_cap = {"A1": qE_r, "A2": qE_r, "A3": qE_r,
                    "A4": q_ri, "A5": q_ri, "A6": q_ri,
                    "A7": float("inf")}[active]

    q_max_val = min(q_max_val_orig, phase_for_regime[active], rise_cap)

    if active in ("A1", "A2", "A3"):
        D_norm_cubed_const = 1.0
    elif active in ("A4", "A5", "A6"):
        D_norm_cubed_const = (D_i_val / D_Euc_cm) ** 3
    else:  # A7
        D_norm_cubed_const = (min(D_dec_val, D_i_val, D_dec_r_cm) / D_Euc_cm) ** 3
    D_eff_const_cm = (D_norm_cubed_const ** (1.0 / 3.0)) * D_Euc_cm

    D_min_norm_cubed_sidebar = (D_min_cm / D_Euc_cm) ** 3
    q_min_sq_sidebar         = q_min ** 2

    # ── R(q) cumulative and differential ─────────────────────────────────────
    if use_full:
        q_internal, dRdq_internal = model.dR_dq_full_integral(
            i_det, N_exp_fix, t_cad_fix_s,
            q_min=0.0, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start, N_q=500,
        )
        dq = float(q_internal[1] - q_internal[0])
        # Trapezoidal cumulative from the right: R(x) = ∫_x^{q_nr} dR/dq dq.
        trapz_cells_q = 0.5 * (dRdq_internal[:-1] + dRdq_internal[1:]) * dq
        cum_right_q   = np.zeros_like(dRdq_internal)
        cum_right_q[:-1] = np.flip(np.cumsum(np.flip(trapz_cells_q)))
        Rq_cum  = np.interp(q_grid, q_internal, cum_right_q)
        Rq_diff = np.interp(q_grid, q_internal, dRdq_internal)
    else:
        D_factor_q = max(D_norm_cubed_const - D_min_norm_cubed_sidebar, 0.0)
        Rq_cum  = base * np.maximum(q_max_val ** 2 - q_grid ** 2, 0.0) * D_factor_q
        Rq_diff = np.where(q_grid <= q_max_val,
                           2.0 * q_grid * base * D_factor_q,
                           0.0)

    # ── R(D) cumulative and differential ─────────────────────────────────────
    if use_full:
        D_grid_internal_cm, dRdD_internal = model.dR_dD_full_integral(
            i_det, N_exp_fix, t_cad_fix_s,
            q_min=q_min, D_min_cm=0.0,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start, N_q=500, N_D=200,
        )
        dD = float(D_grid_internal_cm[1] - D_grid_internal_cm[0])
        trapz_cells_D = 0.5 * (dRdD_internal[:-1] + dRdD_internal[1:]) * dD
        cum_right_D   = np.zeros_like(dRdD_internal)
        cum_right_D[:-1] = np.flip(np.cumsum(np.flip(trapz_cells_D)))
        RD_cum         = np.interp(D_grid_cm, D_grid_internal_cm, cum_right_D)
        RD_diff_per_cm = np.interp(D_grid_cm, D_grid_internal_cm, dRdD_internal)
        RD_diff        = RD_diff_per_cm * GPC_TO_CM   # → yr⁻¹ per Gpc
    else:
        q_factor_D     = max(q_max_val ** 2 - q_min_sq_sidebar, 0.0)
        D_norm_cubed_x = (D_grid_cm / D_Euc_cm) ** 3
        RD_cum  = base * q_factor_D * np.maximum(
            D_norm_cubed_const - D_norm_cubed_x, 0.0
        )
        # Dominant-term dR/dD = 3D²/D_Euc³ · base · q_factor, in yr⁻¹/cm;
        # convert to yr⁻¹/Gpc on the way out.
        RD_diff_per_cm = np.where(
            D_grid_cm <= D_eff_const_cm,
            3.0 * (D_grid_cm ** 2) / (D_Euc_cm ** 3) * base * q_factor_D,
            0.0,
        )
        RD_diff = RD_diff_per_cm * GPC_TO_CM

    # f_night scaling — applies uniformly to cumulative and differential.
    if is_subday_optical:
        Rq_cum  = Rq_cum  * f_night_mul
        Rq_diff = Rq_diff * f_night_mul
        RD_cum  = RD_cum  * f_night_mul
        RD_diff = RD_diff * f_night_mul

    total_rate_q = float(Rq_cum[0]) if np.isfinite(Rq_cum[0]) else float("nan")
    total_rate_D = float(RD_cum[0]) if np.isfinite(RD_cum[0]) else float("nan")

    return {
        "qdview_q_grid_flat":       _array_to_list(q_grid),
        "qdview_Rq_cum_flat":       _array_to_list(Rq_cum),
        "qdview_Rq_diff_flat":      _array_to_list(Rq_diff),
        "qdview_D_grid_Gpc_flat":   _array_to_list(D_grid_Gpc),
        "qdview_RD_cum_flat":       _array_to_list(RD_cum),
        "qdview_RD_diff_flat":      _array_to_list(RD_diff),
        "qdview_q_dec":             q_dec_val,
        "qdview_q_j":               q_j_val,
        "qdview_q_nr":              q_nr_val,
        "qdview_D_Euc_Gpc":         D_Euc_Gpc,
        "qdview_qmin_sidebar":      q_min,
        "qdview_Dmin_Gpc_sidebar":  D_min_cm / GPC_TO_CM,
        "qdview_total_rate_q":      _nan_to_none(total_rate_q),
        "qdview_total_rate_D":      _nan_to_none(total_rate_D),
        "qdview_regime_id":         regime_id_val,
        "qdview_N_fix":             N_exp_fix,
        "qdview_t_cad_fix_s":       t_cad_fix_s,
        "qdview_t_cad_fix_h":       t_cad_fix_s / 3600.0,
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


def compute_qdview(params, N_exp_fix, t_cad_fix_s) -> dict:
    """Return a joint R(q) + R(D) payload. Called from JS when the qdview
    big sliders move — avoids a full surface+optimizer recompute.
    """
    try:
        state = _build_models(params)
        payload = _compute_qdview_sweep(
            state, float(N_exp_fix), float(t_cad_fix_s)
        )
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
        f_live_night = state["f_live_night"]
        t_overhead_s = state["t_overhead_s"]
        optical_on   = state["optical_on"]
        color_on     = state["color_on"]
        full_on      = state["full_on"]
        q_min        = state["q_min"]
        D_min_cm     = state["D_min_cm"]
        s_fade        = state["s_fade"]
        s_rise        = state["s_rise"]
        s_mode       = state["s_mode"]
        rise_random_start = state["rise_random_start"]
        fade_random_start = state["fade_random_start"]
        toh_approx   = state["toh_approx"]
        t_night_s    = state["t_night_s"]
        f_night      = state["f_night"]
        N_exp_max    = state["N_exp_max"]
        model_day    = state["model_day"]
        model_night  = state["model_night"]

        # Grid resolution: regime-colouring needs the higher density to keep
        # discrete boundaries clean.
        nx = NX_REGIME if color_on else NX_DEFAULT
        ny = NY_REGIME if color_on else NY_DEFAULT

        # ── Surface ──────────────────────────────────────────────────────────
        X, Y_s, Z_plot, Z_raw, regime_id, t_exp_g, q_med_g, D_med_Gpc_g = compute_surface(
            model_day, model_night, i_det,
            optical_survey=optical_on, color_regimes=color_on,
            t_night_s=t_night_s, nx=nx, ny=ny,
            full_integral=full_on, q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )

        if toh_approx and t_overhead_s > 0:
            # Sub-day optical points use f_live_night in the validity boundary,
            # matching the rescaled t_exp formula t_exp = (f_live/f_night)·t_cad/N_exp.
            if optical_on and model_night is not None:
                f_eff = np.where(Y_s < DAY_S, f_live_night, f_live)
            else:
                f_eff = f_live
            budget = f_eff * Y_s / X
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
        # Validity constraint for the approx-mode optimizer:
        # f_live_eff · t_cad / N_exp must exceed t_OH for the strategy to be feasible.
        # In optical mode the sub-day branch uses f_live_eff = f_live / f_night.
        opt_validity_fn = None
        if toh_approx and t_overhead_s > 0:
            _f_day, _f_night_eff, _toh0 = float(f_live), float(f_live_night), float(t_overhead_s)
            _optical = bool(optical_on and model_night is not None)
            def opt_validity_fn(N_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
                if _optical:
                    f_eff = np.where(t_arr < DAY_S, _f_night_eff, _f_day)
                else:
                    f_eff = _f_day
                return f_eff * t_arr / N_arr > _toh0

        N_opt, t_cad_opt_s, log10R_opt = maximize_log_surface_iterative(
            model_day, model_night, i_det,
            x_min=0.0, x_max=math.log10(N_exp_max),
            y_min=0.0, y_max=8.0,
            optical_survey=optical_on, t_night_s=t_night_s,
            full_integral=full_on, q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
            validity_fn=opt_validity_fn,
        )
        R_opt, t_exp_opt_s, q_med_opt, D_med_Gpc_opt = _eval_point(
            N_opt, t_cad_opt_s, i_det, model_day, model_night,
            f_live, f_live_night, f_night, optical_on, toh_approx, t_overhead_s,
            full_integral=full_on, q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )

        # ── ZTF reference points (the two real observing modes) ─────────────
        # Mode A — public all-sky: ~15,000 deg² every 2 nights.
        N_ztf = min(ZTF_PUBLIC_OMEGA_SRV_DEG2 / ZTF_OMEGA_EXP_DEG2, N_exp_max)
        t_cad_ztf_s = ZTF_PUBLIC_T_CAD_S
        R_ztf, t_exp_ztf_s, q_med_ztf, D_med_Gpc_ztf = _eval_point(
            N_ztf, t_cad_ztf_s, i_det, model_day, model_night,
            f_live, f_live_night, f_night, optical_on, toh_approx, t_overhead_s,
            full_integral=full_on, q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )

        # Mode B — high-cadence partnership/ZUDS: ~2,500 deg², 6 visits/night.
        # t_cad sits just inside the sub-night continuous region so the point
        # stays valid in optical-survey mode (i·t_cad < t_night needs i ≤ 6).
        N_ztf_hc = min(ZTF_HC_OMEGA_SRV_DEG2 / ZTF_OMEGA_EXP_DEG2, N_exp_max)
        t_cad_ztf_hc_s = 0.98 * t_night_s / float(ZTF_HC_VISITS_PER_NIGHT)
        R_ztf_hc, t_exp_ztf_hc_s, q_med_ztf_hc, D_med_Gpc_ztf_hc = _eval_point(
            N_ztf_hc, t_cad_ztf_hc_s, i_det, model_day, model_night,
            f_live, f_live_night, f_night, optical_on, toh_approx, t_overhead_s,
            full_integral=full_on, q_min=q_min, D_min_cm=D_min_cm,
            s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
            rise_random_start=rise_random_start,
            fade_random_start=fade_random_start,
        )
        if optical_on:
            _, _hc_valid = optical_survey_tcad_seconds(
                np.array([t_cad_ztf_hc_s]), i_det=i_det, t_night_s=t_night_s,
            )
            if not bool(_hc_valid[0]):
                # Infeasible schedule (can't fit i visits in one night) — hide.
                R_ztf_hc = math.nan

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

        # qdview shares its strategy with the slice modes' defaults so the first
        # render lines up with the optimum; per-slider drags will override.
        qdview_nfix_log = float(params.get("qdview_nfix_log", default_nfix_log))
        qdview_tfix_log = float(params.get("qdview_tfix_log", default_tfix_log))
        qdview_N_fix    = 10.0 ** qdview_nfix_log
        qdview_t_fix_s  = 10.0 ** qdview_tfix_log
        qdview_payload  = _compute_qdview_sweep(state, qdview_N_fix, qdview_t_fix_s)

        # ── Derived GRB counts ───────────────────────────────────────────────
        rho      = 10.0 ** float(params["rho_grb_log10"])
        D_gpc    = float(params["D_euc_gpc"])
        theta_j  = float(params["theta_j_rad"])
        R_int_yr = rho * (4.0 / 3.0) * math.pi * D_gpc ** 3
        f_b      = theta_j ** 2 / 2.0
        R_toward_day = R_int_yr * f_b / 365.25

        t_dec_s_val  = float(model_day.derived.t_dec_s)
        F_nu_tdec_Jy = float(model_day.derived.F_dec_Jy)

        zmax_log10 = float(np.nanmax(Z_plot)) if np.any(np.isfinite(Z_plot)) else 0.0
        R_surface_max = float(np.nanmax(R_lin)) if np.any(np.isfinite(R_lin)) else 0.0

        # ── Gap times (optical t-slice) ──────────────────────────────────────
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
                full_integral=full_on, q_min=q_min, D_min_cm=D_min_cm,
                s_fade=s_fade, s_rise=s_rise, s_mode=s_mode,
                rise_random_start=rise_random_start,
                fade_random_start=fade_random_start,
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
            "N_ztf_hc":        float(N_ztf_hc),
            "t_cad_ztf_hc_h":  float(t_cad_ztf_hc_s / 3600.0),
            "t_cad_ztf_hc_s":  float(t_cad_ztf_hc_s),
            "R_ztf_hc":        _nan_to_none(R_ztf_hc),
            "t_exp_ztf_hc_s":  _nan_to_none(t_exp_ztf_hc_s),
            "q_med_ztf_hc":       _nan_to_none(q_med_ztf_hc),
            "D_med_Gpc_ztf_hc":   _nan_to_none(D_med_Gpc_ztf_hc),
            "zmax_log10":   zmax_log10,
            "R_surface_max": R_surface_max,
            "R_int_yr":     float(R_int_yr),
            "R_toward_day": float(R_toward_day),
            "t_dec_s":      t_dec_s_val,
            "F_nu_tdec_Jy": F_nu_tdec_Jy,
            "F_dec_override_applied": bool(state["fdec_override_applied"]),  # TEMP-FDEC-OVERRIDE
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
            # ── Joint R(q) + R(D) view payload ─────────────────────────────
            **qdview_payload,
            "error":        None,
        }

    except Exception:
        import traceback
        return {"error": traceback.format_exc(), "X_flat": None, "shape": None}
