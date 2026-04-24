# callbacks/surface.py
"""Main surface-update callback: computes 3D surface, 2D slices, metrics, and export data."""
from __future__ import annotations

import os
import sys

import dash
import numpy as np
import plotly
from dash import Input, Output, State, html

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.params import GPC_TO_CM, SurveyDesignParams, SurveyStrategy
from grb_detect.plot_3d_core import (
    _on_axis_rate_linear,
    compute_surface,
    make_rate_model,
    maximize_log_surface_iterative,
)
from grb_detect.survey import exposure_time_s

from components.figures import (
    ZMIN_DISPLAY_LOG10,
    _empty_figure as build_empty_figure,
    build_3d_figure,
    build_metrics_bar,
    build_nslice_figure,
    build_tslice_figure,
)

# ── App constants ────────────────────────────────────────────────────────────
ZTF_OMEGA_EXP_DEG2: float = 47.0
OMEGA_SRV_DEFAULT_DEG2: float = 27500.0

NX_REGIME, NY_REGIME = 200, 240
NX_DEFAULT, NY_DEFAULT = 160, 200

RENDER_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "local")
RUNTIME_INFO = (
    f"commit={RENDER_COMMIT} | "
    f"python={sys.version.split()[0]} | "
    f"dash={dash.__version__} | "
    f"plotly={plotly.__version__} | "
    f"numpy={np.__version__}"
)
DEBUG_MODE: bool = os.environ.get("GRB_DEBUG", "0").lower() in ("1", "true", "yes")


# ── Helper: convert region_masks to 1-D regime-ID array ────────────────────

def _masks_1d(masks: dict, n: int) -> np.ndarray:
    rid = np.full(n, np.nan, dtype=float)
    for k, key in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
        mk = np.asarray(masks[key]).ravel()
        if mk.size == n:
            rid[mk] = float(k)
    return rid


# ── Helper: build surface CSV content for export ────────────────────────────

def _build_csv(X: np.ndarray, Y_s: np.ndarray, Z_raw: np.ndarray, regime_id) -> str:
    lines = ["N_exp,t_cad_s,t_cad_h,log10_R_det,regime_id"]
    flat_N = X.ravel()
    flat_t = Y_s.ravel()
    flat_z = Z_raw.ravel()
    flat_r = regime_id.ravel() if regime_id is not None else np.full(flat_N.size, np.nan)
    for n, t, z, r in zip(flat_N, flat_t, flat_z, flat_r):
        rid_str = f"{int(r)}" if np.isfinite(r) else ""
        lines.append(f"{n:.6g},{t:.6g},{t/3600:.6g},{z:.6g},{rid_str}")
    return "\n".join(lines)


# ── Model dispatch helper ────────────────────────────────────────────────────

def _pick_model(t_cad_s: float, optical_on: bool, model_day, model_night):
    """Return model_night for sub-day optical cadences, model_day otherwise."""
    if optical_on and model_night is not None and t_cad_s < DAY_S:
        return model_night
    return model_day


# ── Rate + mask computation helper ──────────────────────────────────────────

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

    Applies f_night scaling for sub-day optical cadences, the approx validity
    mask, and optional regime-ID assignment.  All arrays share the same shape
    as N_arr / t_arr.
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

    q_med, D_med_cm = model.compute_medians(i_det, N_arr, t_arr, full_integral=full_integral, off_axis=off_axis)
    t_exp = model.t_exp_s(N_arr, t_arr)
    D_med_Gpc = D_med_cm / GPC_TO_CM

    if approx_on and float(t_overhead_s) > 0:
        _inv = float(f_live) * t_arr / N_arr <= float(t_overhead_s)
        t_exp     = np.where(_inv, np.nan, t_exp)
        q_med     = np.where(_inv, np.nan, q_med)
        D_med_Gpc = np.where(_inv, np.nan, D_med_Gpc)

    return R, t_exp, q_med, D_med_Gpc, rid


# ── Point evaluation helper ──────────────────────────────────────────────────

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

    Mirrors the model dispatch and approx validity mask used in compute_surface +
    the post-hoc mask, so every point evaluation is automatically consistent with
    what the surface shows.  Returns (nan, nan, nan, nan) if outside the valid domain.
    """
    _nan4 = (np.nan, np.nan, np.nan, np.nan)
    if not (np.isfinite(N_exp) and np.isfinite(t_cad_s) and N_exp > 0 and t_cad_s > 0):
        return _nan4

    # Model dispatch — mirrors compute_surface logic
    model = _pick_model(t_cad_s, optical_on, model_day, model_night)

    # Rate
    N_arr = np.array([N_exp])
    t_arr = np.array([t_cad_s])
    if full_integral:
        log10R = float(model.rate_log10_full_integral(i_det, N_arr, t_arr)[0])
    else:
        log10R = float(model.rate_log10(i_det, N_arr, t_arr)[0])
    R = 10.0 ** log10R if np.isfinite(log10R) else np.nan
    # Sub-day optical: only the nighttime fraction of detections are accessible
    if optical_on and model_night is not None and t_cad_s < DAY_S:
        R = R * f_night

    # Off-axis correction: subtract on-axis contribution (q < q_dec)
    if off_axis and np.isfinite(R):
        r_on = float(_on_axis_rate_linear(model, i_det, N_arr, t_arr)[0])
        if optical_on and model_night is not None and t_cad_s < DAY_S:
            r_on = r_on * f_night
        R_off = R - r_on if np.isfinite(r_on) else R
        if R_off <= 0:
            return _nan4
        R = R_off

    # t_exp
    try:
        t_exp = float(exposure_time_s(
            SurveyStrategy(N_exp=N_exp, t_cad_s=t_cad_s), model.instrument,
        ))
        if not np.isfinite(t_exp) or t_exp <= 0:
            t_exp = np.nan
    except Exception:
        t_exp = np.nan

    # Approx validity check — same criterion as the surface post-hoc mask
    if approx_on and t_overhead_s > 0:
        if f_live * t_cad_s / N_exp <= t_overhead_s:
            return _nan4

    # Medians — use model_day for consistency with compute_surface
    try:
        qm, dm = model_day.compute_medians(i_det, N_arr, t_arr, full_integral=full_integral, off_axis=off_axis)
        q_med     = float(qm[0])
        D_med_Gpc = float(dm[0]) / GPC_TO_CM
    except Exception:
        q_med, D_med_Gpc = np.nan, np.nan

    return R, t_exp, q_med, D_med_Gpc


# ── Main registration ────────────────────────────────────────────────────────

def register(app: dash.Dash) -> None:
    @app.callback(
        Output("graph-3d", "figure"),
        Output("graph-nslice", "figure"),
        Output("graph-tslice", "figure"),
        Output("metrics-bar", "children"),
        Output("status", "children"),
        Output("surface-store", "data"),
        Input("i_slider", "value"),
        Input("Alog_slider", "value"),
        Input("omegaexp_slider", "value"),
        Input("flive_slider", "value"),
        Input("toh_slider", "value"),
        Input("optical-switch", "value"),
        Input("toh-approx-switch", "value"),
        Input("regime-color-switch", "value"),
        Input("tnight_slider", "value"),
        Input("omega_srv_slider", "value"),
        Input("theme-store", "data"),
        Input("full-integral-switch", "value"),
        Input("off-axis-switch", "value"),
        Input("p_slider", "value"),
        Input("nu_log_slider", "value"),
        Input("Ekiso_log_slider", "value"),
        Input("n0_log_slider", "value"),
        Input("epse_slider", "value"),
        Input("epsB_slider", "value"),
        Input("thetaj_slider", "value"),
        Input("gamma0_log_slider", "value"),
        Input("deuc_slider", "value"),
        Input("rho_grb_log_slider", "value"),
        Input("nslice-tfix-slider", "value"),
        Input("tslice-nfix-slider", "value"),
        Input("view-store", "data"),
    )
    def update_all_visuals(
        i_det, A_log, omega_exp_deg2, f_live, t_overhead_s,
        optical_switch, toh_approx_switch, color_switch, tnight_hours, omega_srv_deg2, theme,
        full_integral_switch, off_axis_switch,
        p_val, nu_log, ekiso_log, n0_log, epse_val, epsB_val,
        thetaj_val, gamma0_log, deuc_gpc, rho_grb_log,
        nslice_tfix_log, tslice_nfix_log, view_mode,
    ):
        optical_on = bool(optical_switch)
        approx_on = bool(toh_approx_switch)
        color_on = bool(color_switch)
        full_integral_on = bool(full_integral_switch)
        off_axis_on = bool(off_axis_switch)

        survey_defaults = SurveyDesignParams()
        t_night_s = float(tnight_hours) * 3600.0 if optical_on else survey_defaults.t_night_s
        design = SurveyDesignParams(omega_survey_max_sr=float(omega_srv_deg2) * DEG2_TO_SR)

        # ── Build rate models ────────────────────────────────────────────────
        # In approx mode, pass t_overhead_s=0 so t_exp = f_live·t_cad/N_exp.
        # The validity domain (f_live·t_cad/N_exp > t_OH) is enforced by the
        # post-surface mask and by _eval_point() for individual evaluations.
        t_oh_model = 0.0 if approx_on else float(t_overhead_s)
        f_night_val = t_night_s / DAY_S  # fractional night length (rate multiplier for sub-day)

        physics_kwargs = dict(
            p=float(p_val),
            nu_log10=float(nu_log),
            E_kiso_log10=float(ekiso_log),
            n0_log10=float(n0_log),
            epsilon_e_log10=float(epse_val),
            epsilon_B_log10=float(epsB_val),
            theta_j_rad=float(thetaj_val),
            gamma0_log10=float(gamma0_log),
            D_euc_gpc=float(deuc_gpc),
            rho_grb_log10=float(rho_grb_log),
        )

        model_night = None
        if optical_on:
            model_day = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=t_oh_model,
                omega_exp_deg2=float(omega_exp_deg2), design=design,
                **physics_kwargs,
            )
            model_night = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=t_oh_model,
                omega_exp_deg2=float(omega_exp_deg2), design=design,
                **physics_kwargs,
            )
        else:
            model_day = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=t_oh_model,
                omega_exp_deg2=float(omega_exp_deg2), design=design,
                **physics_kwargs,
            )

        # ── Compute surface ──────────────────────────────────────────────────
        nx = NX_REGIME if color_on else NX_DEFAULT
        ny = NY_REGIME if color_on else NY_DEFAULT
        X, Y_s, Z_plot, Z_raw, regime_id, t_exp_grid, q_med_grid, D_med_Gpc_grid = compute_surface(
            model_day, model_night, int(i_det),
            optical_survey=optical_on, color_regimes=color_on,
            t_night_s=t_night_s, nx=nx, ny=ny,
            full_integral=full_integral_on, off_axis=off_axis_on,
        )

        # In approx mode, mask points where f_live·t_cad/N_exp ≤ t_OH
        # (the validity boundary that the approximation cannot remove).
        if approx_on and float(t_overhead_s) > 0:
            _f = float(f_live)
            _toh = float(t_overhead_s)
            _budget = _f * Y_s / X   # f_night is a rate multiplier; f_live budget unchanged
            _invalid = _budget <= _toh
            Z_plot = np.where(_invalid, np.nan, Z_plot)
            Z_raw = np.where(_invalid, np.nan, Z_raw)
            if regime_id is not None:
                regime_id = np.where(_invalid, np.nan, regime_id)
            t_exp_grid    = np.where(_invalid, np.nan, t_exp_grid)
            q_med_grid    = np.where(_invalid, np.nan, q_med_grid)
            D_med_Gpc_grid = np.where(_invalid, np.nan, D_med_Gpc_grid)

        Y_h = Y_s / 3600.0
        R_plot = np.where(np.isfinite(Z_plot), 10 ** Z_plot, np.nan)

        # ── Optimisation ─────────────────────────────────────────────────────
        N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr

        # In approx mode the models use t_oh=0 (no t_exp penalty), so the optimizer
        # can freely explore the region where budget = f_eff·t_cad/N_exp ≤ t_OH —
        # which is masked on the displayed surface.  Pass a validity_fn so the
        # optimizer only considers the same valid domain the surface shows.
        opt_validity_fn = None
        if approx_on and float(t_overhead_s) > 0:
            _f0, _toh0 = float(f_live), float(t_overhead_s)
            def opt_validity_fn(N_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
                return _f0 * t_arr / N_arr > _toh0

        N_opt, t_cad_opt_s, log10R_opt = maximize_log_surface_iterative(
            model_day, model_night, int(i_det),
            x_min=0.0, x_max=np.log10(N_exp_max),
            y_min=0.0, y_max=8.0,
            optical_survey=optical_on, t_night_s=t_night_s,
            full_integral=full_integral_on, off_axis=off_axis_on,
            validity_fn=opt_validity_fn,
        )
        t_cad_opt_hr = t_cad_opt_s / 3600.0 if np.isfinite(t_cad_opt_s) else np.nan
        R_opt, t_exp_opt_s, q_med_opt, D_med_Gpc_opt = _eval_point(
            N_opt, t_cad_opt_s, int(i_det),
            model_day, model_night,
            float(f_live), f_night_val,
            optical_on, approx_on, float(t_overhead_s),
            full_integral=full_integral_on, off_axis=off_axis_on,
        )

        # ── ZTF reference point ──────────────────────────────────────────────
        N_ztf = OMEGA_SRV_DEFAULT_DEG2 / ZTF_OMEGA_EXP_DEG2
        N_ztf = min(N_ztf, N_exp_max)  # guard: rounding in make_rate_model can push N_ztf just above N_exp_max
        t_cad_ztf_s = 2.0 * DAY_S
        t_cad_ztf_hr = t_cad_ztf_s / 3600.0
        R_ztf, t_exp_ztf_s, q_med_ztf, D_med_Gpc_ztf = _eval_point(
            N_ztf, t_cad_ztf_s, int(i_det),
            model_day, model_night,
            float(f_live), f_night_val,
            optical_on, approx_on, float(t_overhead_s),
            full_integral=full_integral_on, off_axis=off_axis_on,
        )

        # ── Surface for 3D figure ─────────────────────────────────────────────
        if optical_on:
            y_rows = np.asarray(Y_s[:, 0], dtype=float)
            keep_rows = np.where(y_rows < float(DAY_S))[0]
        else:
            keep_rows = np.arange(Y_s.shape[0], dtype=int)

        Xs = X[keep_rows, :]
        Ys_h = Y_h[keep_rows, :]
        Rs = R_plot[keep_rows, :]
        Z_plot_sub = Z_plot[keep_rows, :]
        regime_sub = regime_id[keep_rows, :] if regime_id is not None else None
        t_exp_sub      = t_exp_grid[keep_rows, :]
        q_med_sub      = q_med_grid[keep_rows, :]
        D_med_Gpc_sub  = D_med_Gpc_grid[keep_rows, :]

        N_cols = np.asarray(X[0, :], dtype=float)
        t_cad_max_s = float(np.nanmax(Y_s[:, 0])) if np.any(np.isfinite(Y_s[:, 0])) else 0.0

        R_candidates = []
        if np.any(np.isfinite(R_plot)):
            R_candidates.append(float(np.nanmax(R_plot)))
        for r in [R_opt, R_ztf]:
            if np.isfinite(r):
                R_candidates.append(float(r))
        Rmax = max(R_candidates) if R_candidates else 1.0

        # ── Build 3D figure ───────────────────────────────────────────────────
        if view_mode != "3d":
            fig_3d = dash.no_update
        else:
            fig_3d = build_3d_figure(
                surface_Xs=Xs,
                surface_Ys_h=Ys_h,
                surface_Rs=Rs,
                surface_Z_plot=Z_plot_sub,
                surface_regime_id=regime_sub,
                surface_t_exp=t_exp_sub,
                surface_q_med=q_med_sub,
                surface_D_med_Gpc=D_med_Gpc_sub,
                color_on=color_on,
                optical_on=optical_on,
                full_integral_on=full_integral_on,
                off_axis_on=off_axis_on,
                model_day=model_day,
                model_night=model_night,
                i_det=int(i_det),
                day_line_N_cols=N_cols,
                day_line_t_max_s=t_cad_max_s,
                N_opt=N_opt, t_cad_opt_hr=t_cad_opt_hr, R_opt=R_opt, t_exp_opt_s=t_exp_opt_s,
                q_med_opt=q_med_opt, D_med_Gpc_opt=D_med_Gpc_opt,
                N_ztf=N_ztf, t_cad_ztf_hr=t_cad_ztf_hr, R_ztf=R_ztf, t_exp_ztf_s=t_exp_ztf_s,
                q_med_ztf=q_med_ztf, D_med_Gpc_ztf=D_med_Gpc_ztf,
                Rmax=Rmax, theme=theme,
            )

        # ── N-slice (R vs N_exp at slider-controlled t_cad) ─────────────────
        if view_mode != "nslice":
            fig_nslice = dash.no_update
        else:
            t_cad_fix_s  = 10 ** float(nslice_tfix_log)
            t_cad_fix_hr = t_cad_fix_s / 3600.0
            N_sweep = np.logspace(0, np.log10(N_exp_max), 800)
            t_fixed = np.full_like(N_sweep, t_cad_fix_s)
            model_nslice = _pick_model(t_cad_fix_s, optical_on, model_day, model_night)
            R_n, t_exp_n, q_med_n, D_med_Gpc_n, rid_n = _compute_rate(
                model_nslice, int(i_det), N_sweep, t_fixed,
                full_integral=full_integral_on,
                optical_on=optical_on, model_night=model_night,
                t_cad_scalar=t_cad_fix_s, f_night_val=f_night_val,
                approx_on=approx_on, f_live=float(f_live),
                t_overhead_s=float(t_overhead_s), color_on=color_on,
                off_axis=off_axis_on,
            )
            fig_nslice = build_nslice_figure(
                N_sweep=N_sweep, R_n=R_n, rid_n=rid_n,
                t_exp_n=t_exp_n, q_med_n=q_med_n, D_med_Gpc_n=D_med_Gpc_n,
                N_opt=N_opt, R_ztf=R_ztf, N_ztf=N_ztf,
                t_cad_fix_hr=t_cad_fix_hr, t_cad_opt_hr=t_cad_opt_hr,
                color_on=color_on, theme=theme,
            )

        # ── t-slice (R vs t_cad at slider-controlled N_exp) ─────────────────
        if view_mode != "tslice":
            fig_tslice = dash.no_update
        else:
            N_fix = 10 ** float(tslice_nfix_log)
            _cr_kwargs = dict(
                full_integral=full_integral_on,
                optical_on=optical_on, model_night=model_night,
                approx_on=approx_on, f_live=float(f_live),
                t_overhead_s=float(t_overhead_s), color_on=color_on,
                off_axis=off_axis_on,
            )
            if optical_on and model_night is not None:
                # Continuous region (sub-night, uses model_night)
                t_cont_max_s = t_night_s / float(i_det)
                t_cont = np.logspace(
                    np.log10(max(10.0, 1.0)),
                    np.log10(max(11.0, t_cont_max_s * 0.999)),
                    600,
                )
                N_cont = np.full_like(t_cont, N_fix)
                R_cont, t_exp_cont, q_med_cont, D_med_Gpc_cont, rid_cont = _compute_rate(
                    model_night, int(i_det), N_cont, t_cont,
                    t_cad_scalar=0.0,  # always sub-day for cont region → force f_night scaling
                    f_night_val=f_night_val, **_cr_kwargs,
                )

                # Discrete region (integer days, uses model_day)
                n_max_days = min(500, int(np.floor(1e8 / DAY_S)))
                t_disc = np.arange(1, n_max_days + 1, dtype=float) * DAY_S
                N_disc = np.full_like(t_disc, N_fix)
                R_disc, t_exp_disc, q_med_disc, D_med_Gpc_disc, rid_disc = _compute_rate(
                    model_day, int(i_det), N_disc, t_disc,
                    t_cad_scalar=float(DAY_S),  # multi-day → no f_night scaling
                    f_night_val=f_night_val, **_cr_kwargs,
                )

                fig_tslice = build_tslice_figure(
                    t_cont_h=t_cont / 3600.0, R_cont=R_cont, rid_cont=rid_cont,
                    t_exp_cont=t_exp_cont, q_med_cont=q_med_cont, D_med_Gpc_cont=D_med_Gpc_cont,
                    t_disc_h=t_disc / 3600.0, R_disc=R_disc, rid_disc=rid_disc,
                    t_exp_disc=t_exp_disc, q_med_disc=q_med_disc, D_med_Gpc_disc=D_med_Gpc_disc,
                    gap_lo_h=t_cont_max_s / 3600.0, gap_hi_h=24.0,
                    t_opt_h=t_cad_opt_hr, t_ztf_h=t_cad_ztf_hr,
                    R_ztf=R_ztf, R_opt=R_opt, N_fix=N_fix, N_opt=N_opt,
                    color_on=color_on, theme=theme, optical_on=True,
                )
            else:
                t_sweep = np.logspace(0, 8, 1500)
                N_fixed = np.full_like(t_sweep, N_fix)
                R_sweep, t_exp_sw, q_med_sw, D_med_Gpc_sw, rid_sweep = _compute_rate(
                    model_day, int(i_det), N_fixed, t_sweep,
                    t_cad_scalar=float(DAY_S),  # non-optical: no f_night scaling
                    f_night_val=f_night_val, **_cr_kwargs,
                )
                _empty_arr = np.array([], dtype=float)
                fig_tslice = build_tslice_figure(
                    t_cont_h=t_sweep / 3600.0, R_cont=R_sweep, rid_cont=rid_sweep,
                    t_exp_cont=t_exp_sw, q_med_cont=q_med_sw, D_med_Gpc_cont=D_med_Gpc_sw,
                    t_disc_h=_empty_arr, R_disc=_empty_arr, rid_disc=_empty_arr,
                    t_exp_disc=_empty_arr, q_med_disc=_empty_arr, D_med_Gpc_disc=_empty_arr,
                    gap_lo_h=None, gap_hi_h=None,
                    t_opt_h=t_cad_opt_hr, t_ztf_h=t_cad_ztf_hr,
                    R_ztf=R_ztf, R_opt=R_opt, N_fix=N_fix, N_opt=N_opt,
                    color_on=color_on, theme=theme, optical_on=False,
                )

        # ── Metrics strip ─────────────────────────────────────────────────────
        metrics_children = build_metrics_bar(
            R_opt=R_opt, R_ztf=R_ztf,
            t_cad_opt_s=t_cad_opt_s, N_opt=N_opt, t_exp_opt_s=t_exp_opt_s,
            t_cad_ztf_s=t_cad_ztf_s, N_ztf=N_ztf, t_exp_ztf_s=t_exp_ztf_s,
        )

        # ── Status bar (debug) ────────────────────────────────────────────────
        omega_exp_sr = float(omega_exp_deg2) * DEG2_TO_SR
        debug_items = [
            html.Div(RUNTIME_INFO),
            html.Div(
                f"i={int(i_det)} | A_log={float(A_log):.2f} | "
                f"Ω_exp={omega_exp_deg2:.1f} deg² ({omega_exp_sr:.4g} sr) | "
                f"f_live={float(f_live):.3f} | t_oh={float(t_overhead_s):.1f} s | "
                f"optical={optical_on} | color={color_on}"
            ),
        ]
        if optical_on:
            debug_items.append(html.Div(
                f"t_night={t_night_s / 3600.0:.3g} h | "
                f"f_night={f_night_val:.4g}"
            ))

        status_children: list = []
        if DEBUG_MODE:
            status_children.append(html.Details(
                [html.Summary("Debug info"), *debug_items],
                style={"fontSize": "0.82em", "color": "#888", "marginBottom": "4px"},
            ))

        # Store surface data for CSV export (avoids recomputation on export click)
        surface_data = {
            "N": X.tolist(), "t_s": Y_s.tolist(),
            "Z_raw": np.where(np.isfinite(Z_raw), Z_raw, None).tolist(),
            "regime_id": (np.where(np.isfinite(regime_id), regime_id, None).tolist()
                          if regime_id is not None else None),
        }

        return fig_3d, fig_nslice, fig_tslice, metrics_children, status_children, surface_data

    # ── GRB count display (derived from D_Euc, ℛ, θ_j) ─────────────────────
    @app.callback(
        Output("grb-ntotal-display", "children"),
        Output("grb-ntoward-display", "children"),
        Input("deuc_slider", "value"),
        Input("rho_grb_log_slider", "value"),
        Input("thetaj_slider", "value"),
    )
    def update_grb_counts(deuc_gpc, rho_log, theta_j):
        import math as _math
        rho = 10.0 ** float(rho_log)
        D = float(deuc_gpc)
        V = (4.0 / 3.0) * _math.pi * D ** 3  # Gpc³
        N_total = rho * V
        f_b = float(theta_j) ** 2 / 2.0
        N_toward = N_total * f_b

        def _fmt(x: float) -> str:
            if x >= 1_000_000:
                return f"{x/1_000_000:.1f}M"
            if x >= 1_000:
                return f"{x/1_000:.1f}k"
            return f"{x:.1f}"

        ntotal_children = [
            "R", html.Sub("int"), f" = {_fmt(N_total)} yr⁻¹",
        ]
        N_toward_day = N_toward / 365.25
        ntoward_children = [
            "f", html.Sub("b"), "R", html.Sub("int"), f" = {_fmt(N_toward_day)} day⁻¹",
        ]
        return ntotal_children, ntoward_children

    # ── Dedicated CSV export (reads from surface-store — no recomputation) ──────
    @app.callback(
        Output("download-data", "data"),
        Input("export-btn", "n_clicks"),
        State("surface-store", "data"),
        prevent_initial_call=True,
    )
    def export_csv(n_clicks, surface_data):
        if surface_data is None:
            return dash.no_update
        N = np.array(surface_data["N"], dtype=float)
        t_s = np.array(surface_data["t_s"], dtype=float)
        Z_raw = np.array(
            [[v if v is not None else np.nan for v in row] for row in surface_data["Z_raw"]],
            dtype=float,
        )
        rid_raw = surface_data.get("regime_id")
        if rid_raw is not None:
            regime_id = np.array(
                [[v if v is not None else np.nan for v in row] for row in rid_raw],
                dtype=float,
            )
        else:
            regime_id = None
        csv_data = _build_csv(N, t_s, Z_raw, regime_id)
        return dict(content=csv_data, filename="grb_detection_surface.csv", type="text/csv")
