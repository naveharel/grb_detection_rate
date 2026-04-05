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
from grb_detect.params import SurveyDesignParams, SurveyStrategy
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
) -> tuple[float, float]:
    """Evaluate (R_det, t_exp_s) at a single point with the same pipeline as the surface.

    Mirrors the model dispatch and approx validity mask used in compute_surface +
    the post-hoc mask, so every point evaluation is automatically consistent with
    what the surface shows.  Returns (nan, nan) if the point is outside the valid
    domain (physically invalid, or excluded by the approx validity boundary).
    """
    if not (np.isfinite(N_exp) and np.isfinite(t_cad_s) and N_exp > 0 and t_cad_s > 0):
        return np.nan, np.nan

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
            return np.nan, np.nan
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
            return np.nan, np.nan

    return R, t_exp


# ── Main registration ────────────────────────────────────────────────────────

def register(app: dash.Dash) -> None:
    @app.callback(
        Output("graph-3d", "figure"),
        Output("graph-nslice", "figure"),
        Output("graph-tslice", "figure"),
        Output("metrics-bar", "children"),
        Output("status", "children"),
        Input("i_slider", "value"),
        Input("Alog_slider", "value"),
        Input("omegaexp_slider", "value"),
        Input("flive_slider", "value"),
        Input("toh_slider", "value"),
        Input("optical-switch", "value"),
        Input("toh-approx-switch", "value"),
        Input("color-switch", "value"),
        Input("tnight_slider", "value"),
        Input("omega_srv_slider", "value"),
        Input("theme-store", "data"),
        Input("full-integral-switch", "value"),
        Input("off-axis-switch", "value"),
    )
    def update_all_visuals(
        i_det, A_log, omega_exp_deg2, f_live, t_overhead_s,
        optical_switch, toh_approx_switch, color_switch, tnight_hours, omega_srv_deg2, theme,
        full_integral_switch, off_axis_switch,
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

        model_night = None
        if optical_on:
            model_day = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=t_oh_model,
                omega_exp_deg2=float(omega_exp_deg2), design=design,
            )
            model_night = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=t_oh_model,
                omega_exp_deg2=float(omega_exp_deg2), design=design,
            )
        else:
            model_day = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=t_oh_model,
                omega_exp_deg2=float(omega_exp_deg2), design=design,
            )

        # ── Compute surface ──────────────────────────────────────────────────
        nx = NX_REGIME if color_on else NX_DEFAULT
        ny = NY_REGIME if color_on else NY_DEFAULT
        X, Y_s, Z_plot, Z_raw, regime_id = compute_surface(
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
            validity_fn=opt_validity_fn,
        )
        t_cad_opt_hr = t_cad_opt_s / 3600.0 if np.isfinite(t_cad_opt_s) else np.nan
        R_opt, t_exp_opt_s = _eval_point(
            N_opt, t_cad_opt_s, int(i_det),
            model_day, model_night,
            float(f_live), f_night_val,
            optical_on, approx_on, float(t_overhead_s),
            full_integral=full_integral_on, off_axis=off_axis_on,
        )

        # ── ZTF reference point ──────────────────────────────────────────────
        N_ztf = OMEGA_SRV_DEFAULT_DEG2 / ZTF_OMEGA_EXP_DEG2
        t_cad_ztf_s = 2.0 * DAY_S
        t_cad_ztf_hr = t_cad_ztf_s / 3600.0
        R_ztf, t_exp_ztf_s = _eval_point(
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
        fig_3d = build_3d_figure(
            surface_Xs=Xs,
            surface_Ys_h=Ys_h,
            surface_Rs=Rs,
            surface_Z_plot=Z_plot_sub,
            surface_regime_id=regime_sub,
            color_on=color_on,
            optical_on=optical_on,
            model_day=model_day,
            model_night=model_night,
            i_det=int(i_det),
            day_line_N_cols=N_cols,
            day_line_t_max_s=t_cad_max_s,
            N_opt=N_opt, t_cad_opt_hr=t_cad_opt_hr, R_opt=R_opt, t_exp_opt_s=t_exp_opt_s,
            N_ztf=N_ztf, t_cad_ztf_hr=t_cad_ztf_hr, R_ztf=R_ztf, t_exp_ztf_s=t_exp_ztf_s,
            Rmax=Rmax, theme=theme,
        )

        # ── N-slice (R vs N_exp at optimal t_cad) ────────────────────────────
        if np.isfinite(N_opt) and np.isfinite(t_cad_opt_s):
            N_sweep = np.logspace(0, np.log10(N_exp_max), 800)
            t_fixed = np.full_like(N_sweep, t_cad_opt_s)
            model_nslice = (
                model_night
                if (optical_on and model_night is not None and t_cad_opt_s < DAY_S)
                else model_day
            )
            if full_integral_on:
                Z_n = model_nslice.rate_log10_full_integral(int(i_det), N_sweep, t_fixed)
            else:
                Z_n = model_nslice.rate_log10(int(i_det), N_sweep, t_fixed)
            R_n = np.where(np.isfinite(Z_n), 10 ** Z_n, np.nan)
            if optical_on and model_night is not None and t_cad_opt_s < DAY_S:
                R_n = R_n * f_night_val  # sub-day: scale by nighttime fraction
            if approx_on and float(t_overhead_s) > 0:
                R_n = np.where(float(f_live) * t_cad_opt_s / N_sweep <= float(t_overhead_s), np.nan, R_n)
            rid_n = np.full(len(N_sweep), np.nan)
            if color_on:
                masks_n = model_nslice.region_masks(int(i_det), N_sweep, t_fixed, include_unphysical=False)
                rid_n = _masks_1d(masks_n, len(N_sweep))
            fig_nslice = build_nslice_figure(
                N_sweep=N_sweep, R_n=R_n, rid_n=rid_n,
                N_opt=N_opt, R_opt=R_opt, R_ztf=R_ztf, N_ztf=N_ztf,
                t_cad_opt_hr=t_cad_opt_hr, color_on=color_on, theme=theme,
            )
        else:
            fig_nslice = build_empty_figure("Optimization failed — no optimal point found", theme)

        # ── t-slice (R vs t_cad at optimal N_exp) ────────────────────────────
        if np.isfinite(N_opt) and np.isfinite(t_cad_opt_s):
            if optical_on and model_night is not None:
                # Continuous region
                t_cont_max_s = t_night_s / float(i_det)
                t_cont = np.logspace(
                    np.log10(max(10.0, 1.0)),
                    np.log10(max(11.0, t_cont_max_s * 0.999)),
                    600,
                )
                N_cont = np.full_like(t_cont, float(N_opt))
                if full_integral_on:
                    Z_cont = model_night.rate_log10_full_integral(int(i_det), N_cont, t_cont)
                else:
                    Z_cont = model_night.rate_log10(int(i_det), N_cont, t_cont)
                R_cont = np.where(np.isfinite(Z_cont), 10 ** Z_cont, np.nan)
                R_cont = R_cont * f_night_val  # sub-day: scale by nighttime fraction
                if approx_on and float(t_overhead_s) > 0:
                    R_cont = np.where(float(f_live) * t_cont / N_opt <= float(t_overhead_s), np.nan, R_cont)
                rid_cont = np.full(len(t_cont), np.nan)
                if color_on:
                    masks_c = model_night.region_masks(int(i_det), N_cont, t_cont, include_unphysical=False)
                    rid_cont = _masks_1d(masks_c, len(t_cont))

                # Discrete region (integer days)
                n_max_days = min(500, int(np.floor(1e8 / DAY_S)))
                n_days_arr = np.arange(1, n_max_days + 1)
                t_disc = n_days_arr.astype(float) * DAY_S
                N_disc = np.full_like(t_disc, float(N_opt))
                if full_integral_on:
                    Z_disc = model_day.rate_log10_full_integral(int(i_det), N_disc, t_disc)
                else:
                    Z_disc = model_day.rate_log10(int(i_det), N_disc, t_disc)
                R_disc = np.where(np.isfinite(Z_disc), 10 ** Z_disc, np.nan)
                if approx_on and float(t_overhead_s) > 0:
                    R_disc = np.where(float(f_live) * t_disc / N_opt <= float(t_overhead_s), np.nan, R_disc)
                rid_disc = np.full(len(t_disc), np.nan)
                if color_on:
                    masks_d = model_day.region_masks(int(i_det), N_disc, t_disc, include_unphysical=False)
                    rid_disc = _masks_1d(masks_d, len(t_disc))

                fig_tslice = build_tslice_figure(
                    t_cont_h=t_cont / 3600.0, R_cont=R_cont, rid_cont=rid_cont,
                    t_disc_h=t_disc / 3600.0, R_disc=R_disc, rid_disc=rid_disc,
                    gap_lo_h=t_cont_max_s / 3600.0, gap_hi_h=24.0,
                    t_opt_h=t_cad_opt_hr, t_ztf_h=t_cad_ztf_hr,
                    R_ztf=R_ztf, R_opt=R_opt, N_opt=N_opt,
                    color_on=color_on, theme=theme, optical_on=True,
                )
            else:
                t_sweep = np.logspace(0, 8, 1500)
                N_fixed = np.full_like(t_sweep, float(N_opt))
                if full_integral_on:
                    Z_sweep = model_day.rate_log10_full_integral(int(i_det), N_fixed, t_sweep)
                else:
                    Z_sweep = model_day.rate_log10(int(i_det), N_fixed, t_sweep)
                R_sweep = np.where(np.isfinite(Z_sweep), 10 ** Z_sweep, np.nan)
                if approx_on and float(t_overhead_s) > 0:
                    R_sweep = np.where(float(f_live) * t_sweep / N_opt <= float(t_overhead_s), np.nan, R_sweep)
                rid_sweep = np.full(len(t_sweep), np.nan)
                if color_on:
                    masks_t = model_day.region_masks(int(i_det), N_fixed, t_sweep, include_unphysical=False)
                    rid_sweep = _masks_1d(masks_t, len(t_sweep))
                fig_tslice = build_tslice_figure(
                    t_cont_h=t_sweep / 3600.0, R_cont=R_sweep, rid_cont=rid_sweep,
                    t_disc_h=np.array([]), R_disc=np.array([]), rid_disc=np.array([]),
                    gap_lo_h=None, gap_hi_h=None,
                    t_opt_h=t_cad_opt_hr, t_ztf_h=t_cad_ztf_hr,
                    R_ztf=R_ztf, R_opt=R_opt, N_opt=N_opt,
                    color_on=color_on, theme=theme, optical_on=False,
                )
        else:
            fig_tslice = build_empty_figure("Optimization failed — no optimal point found", theme)

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

        return fig_3d, fig_nslice, fig_tslice, metrics_children, status_children

    # ── Dedicated CSV export (triggered by button only) ──────────────────────
    @app.callback(
        Output("download-data", "data"),
        Input("export-btn", "n_clicks"),
        State("i_slider", "value"),
        State("Alog_slider", "value"),
        State("omegaexp_slider", "value"),
        State("flive_slider", "value"),
        State("toh_slider", "value"),
        State("optical-switch", "value"),
        State("tnight_slider", "value"),
        State("omega_srv_slider", "value"),
        prevent_initial_call=True,
    )
    def export_csv(
        n_clicks,
        i_det, A_log, omega_exp_deg2, f_live, t_overhead_s,
        optical_switch, tnight_hours, omega_srv_deg2,
    ):
        optical_on = bool(optical_switch)
        survey_defaults = SurveyDesignParams()
        t_night_s = float(tnight_hours) * 3600.0 if optical_on else survey_defaults.t_night_s
        design = SurveyDesignParams(omega_survey_max_sr=float(omega_srv_deg2) * DEG2_TO_SR)

        if optical_on:
            f_live_night = float(f_live) * (t_night_s / DAY_S)
            model_exp = make_rate_model(
                A_log=float(A_log), f_live=f_live_night,
                t_overhead_s=float(t_overhead_s),
                omega_exp_deg2=float(omega_exp_deg2), design=design,
            )
            model_night_exp = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=float(t_overhead_s),
                omega_exp_deg2=float(omega_exp_deg2), design=design,
            )
        else:
            model_exp = make_rate_model(
                A_log=float(A_log), f_live=float(f_live),
                t_overhead_s=float(t_overhead_s),
                omega_exp_deg2=float(omega_exp_deg2), design=design,
            )
            model_night_exp = None

        X, Y_s, _, Z_raw, regime_id = compute_surface(
            model_exp, model_night_exp, int(i_det),
            optical_survey=optical_on, color_regimes=True,
            t_night_s=t_night_s, nx=160, ny=200,
        )
        csv_data = _build_csv(X, Y_s, Z_raw, regime_id)
        return dict(content=csv_data, filename="grb_detection_surface.csv", type="text/csv")
