# plot_3d.py
import os
import sys

import numpy as np
import plotly
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.params import SurveyDesignParams, SurveyStrategy
from grb_detect.plot_3d_core import (
    compute_surface,
    discrete_regime_colorscale,
    make_rate_model,
    maximize_log_surface_iterative,
)
from grb_detect.survey import exposure_time_s

RENDER_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "local")
RUNTIME_INFO = (
    f"commit={RENDER_COMMIT} | "
    f"python={sys.version.split()[0]} | "
    f"dash={dash.__version__} | "
    f"plotly={plotly.__version__} | "
    f"numpy={np.__version__}"
)

# Keep discrete-line appearance consistent across modes
DISCRETE_LINE_WIDTH = 5
DISCRETE_MARKER_SIZE = 4

# Hover template matching Plotly's surface-style readout: x, y, z only
XYZ_HOVER = "x: %{x:.6g}<br>y: %{y:.6g}<br>z: %{z:.6g}<extra></extra>"

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def boundary_lines_from_regimes(X, Y, Z, regime_id):
    if regime_id is None:
        return None

    rid = np.asarray(regime_id)
    valid = np.isfinite(rid) & np.isfinite(Z)

    xs, ys, zs = [], [], []

    diff_y = (rid[1:, :] != rid[:-1, :]) & valid[1:, :] & valid[:-1, :]
    ii, jj = np.where(diff_y)
    for i, j in zip(ii, jj):
        xs.extend([X[i, j], X[i + 1, j], None])
        ys.extend([Y[i, j], Y[i + 1, j], None])
        zs.extend([Z[i, j], Z[i + 1, j], None])

    diff_x = (rid[:, 1:] != rid[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    ii, jj = np.where(diff_x)
    for i, j in zip(ii, jj):
        xs.extend([X[i, j], X[i, j + 1], None])
        ys.extend([Y[i, j], Y[i, j + 1], None])
        zs.extend([Z[i, j], Z[i, j + 1], None])

    if len(xs) == 0:
        return None

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.9)", width=4),
        name="Regime boundaries",
        showlegend=False,
        hoverinfo="skip",
    )


def _add_discrete_day_lines(
    fig: go.Figure,
    *,
    model_day,
    i_det: int,
    N_cols: np.ndarray,
    t_cad_max_s: float,
    zmin_plot_log10: float = -1.0,
    color_mode: str = "height",  # "height" or "regime"
    height_cmin: float = -1.0,
    height_cmax: float = 1.0,
    height_colorscale: str = "Plasma",
    regime_colors: list[str] | None = None,  # len=7, A1..A7
) -> None:
    N_cols = np.asarray(N_cols, dtype=float)
    if not np.any(np.isfinite(N_cols)):
        return

    max_days = int(np.floor(float(t_cad_max_s) / float(DAY_S)))
    if max_days < 1:
        return

    # Integer day sampling: all small n, then log-sample larger n
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
        return

    N_line = N_cols[None, :]  # (1, nN)

    # Hover label style: force neutral gray background (not trace color)
    hoverlabel_style = dict(
        bgcolor="rgba(230,230,230,0.92)",
        bordercolor="rgba(160,160,160,1.0)",
        font=dict(color="black"),
    )

    def _regime_id_for_points(t_s: float) -> np.ndarray:
        t_arr = np.full_like(N_line, float(t_s), dtype=float)
        masks = model_day.region_masks(int(i_det), N_line, t_arr, include_unphysical=False)

        rid = np.full(N_cols.shape, np.nan, dtype=float)
        for k, key in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
            mk = np.asarray(masks[key]).reshape(1, -1).ravel()
            rid[mk] = float(k)
        return rid

    for n in n_vals:
        t_s = float(n) * float(DAY_S)

        log10R = model_day.rate_log10(
            i_det=int(i_det),
            N_exp=N_line,
            t_cad_s=np.array([[t_s]], dtype=float),
        )
        log10R = np.asarray(log10R).reshape(1, -1).ravel()

        good = np.isfinite(N_cols) & np.isfinite(log10R) & (log10R >= float(zmin_plot_log10))
        if np.count_nonzero(good) < 2:
            continue

        x = N_cols[good]
        y = np.full_like(x, t_s / 3600.0, dtype=float)  # hours
        z = (10 ** log10R[good]).astype(float)

        if color_mode == "regime" and regime_colors is not None and len(regime_colors) == 7:
            rid_all = _regime_id_for_points(t_s)
            rid = rid_all[good]

            start = 0
            while start < rid.size:
                if not np.isfinite(rid[start]):
                    start += 1
                    continue

                k = int(rid[start])
                end = start + 1
                while end < rid.size and np.isfinite(rid[end]) and int(rid[end]) == k:
                    end += 1

                if end - start >= 2:
                    # Use same alpha in regime mode to avoid perceived width change
                    col = _hex_to_rgba(regime_colors[k - 1], alpha=0.85)

                    fig.add_trace(
                        go.Scatter3d(
                            x=x[start:end],
                            y=y[start:end],
                            z=z[start:end],
                            mode="lines+markers",
                            line=dict(color=col, width=DISCRETE_LINE_WIDTH),
                            marker=dict(size=DISCRETE_MARKER_SIZE, opacity=0.001),
                            showlegend=False,
                            hovertemplate=XYZ_HOVER,
                            hoverlabel=hoverlabel_style,
                        )
                    )

                start = end

        else:
            # Height coloring: keep the connector line *same opacity* as regime mode
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines+markers",
                    line=dict(color="rgba(0,0,0,0.85)", width=DISCRETE_LINE_WIDTH),
                    marker=dict(
                        size=DISCRETE_MARKER_SIZE,
                        color=log10R[good],
                        colorscale=height_colorscale,
                        cmin=float(height_cmin),
                        cmax=float(height_cmax),
                        opacity=1.0,
                        showscale=False,
                    ),
                    showlegend=False,
                    hovertemplate=XYZ_HOVER,
                    hoverlabel=hoverlabel_style,
                )
            )

app = Dash(__name__)
server = app.server

survey_defaults = SurveyDesignParams()
T_NIGHT_DEFAULT_S = survey_defaults.t_night_s

app.layout = html.Div(
    style={
        "width": "98vw",
        "maxWidth": "1800px",
        "margin": "0 auto",
        "paddingTop": "8px",
        "fontFamily": "Arial",
        "overflow": "visible",
    },
    children=[
        html.H2("GRB Detection Rate Surface", style={"margin": "0", "paddingTop": "10px", "paddingBottom": "6px"}),
        dcc.Graph(id="surface", style={"height": "780px", "marginTop": "0px", "paddingTop": "0px"}),
        html.Div(
            style={"display": "flex", "gap": "28px", "alignItems": "center", "marginTop": "6px", "marginBottom": "10px"},
            children=[
                dcc.Checklist(
                    id="optical_checkbox",
                    options=[{"label": "Optical survey", "value": "on"}],
                    value=[],
                    inline=True,
                    style={"userSelect": "none"},
                ),
                dcc.Checklist(
                    id="color_checkbox",
                    options=[{"label": "Color regimes", "value": "on"}],
                    value=[],
                    inline=True,
                    style={"userSelect": "none"},
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(320px, 1fr))", "gap": "12px 18px"},
            children=[
                html.Div(
                    [
                        html.Label("i (detections required)", style={"fontSize": "14px"}),
                        dcc.Slider(
                            2,
                            300,
                            1,
                            value=10,
                            id="i_slider",
                            marks={2: "2", 10: "10", 30: "30", 150: "150", 300: "300"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("log(A [Jy])", style={"fontSize": "14px"}),
                        dcc.Slider(
                            -12,
                            -2,
                            0.01,
                            value=-4.68,
                            id="Alog_slider",
                            marks={-12: "-12", -8: "-8", -4.68: "-4.68", -4: "-4", -2: "-2"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Ω_exp [deg²]", style={"fontSize": "14px"}),
                        dcc.Slider(
                            1,
                            200,
                            1,
                            value=47,
                            id="omegaexp_slider",
                            marks={1: "1", 47: "47", 100: "100", 200: "200"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("f_live", style={"fontSize": "14px"}),
                        dcc.Slider(
                            0.01,
                            1.0,
                            0.01,
                            value=0.1,
                            id="flive_slider",
                            marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2", 0.5: "0.5", 1.0: "1.0"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("t_overhead [s]", style={"fontSize": "14px"}),
                        dcc.Slider(
                            0.0,
                            30.0,
                            0.5,
                            value=0.0,
                            id="toh_slider",
                            marks={0: "0", 10: "10", 15: "15", 20: "20", 30: "30"},
                        ),
                    ]
                ),
                html.Div(
                    id="tnight_container",
                    style={"display": "none"},
                    children=[
                        html.Label("t_night [h]", style={"fontSize": "14px"}),
                        dcc.Slider(
                            4.0,
                            14.0,
                            0.25,
                            value=T_NIGHT_DEFAULT_S / 3600.0,
                            id="tnight_slider",
                            marks={4: "4", 6: "6", 8: "8", 10: "10", 12: "12", 14: "14"},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="status", style={"marginTop": "10px", "opacity": 0.85}),
    ],
)


@app.callback(Output("tnight_container", "style"), Input("optical_checkbox", "value"))
def toggle_tnight_slider(optical_vals):
    optical_on = "on" in (optical_vals or [])
    return {"display": "block"} if optical_on else {"display": "none"}


@app.callback(
    Output("surface", "figure"),
    Output("status", "children"),
    Input("i_slider", "value"),
    Input("Alog_slider", "value"),
    Input("omegaexp_slider", "value"),
    Input("flive_slider", "value"),
    Input("toh_slider", "value"),
    Input("optical_checkbox", "value"),
    Input("color_checkbox", "value"),
    Input("tnight_slider", "value"),
)
def update_surface(i_det, A_log, omega_exp_deg2, f_live, t_overhead_s, optical_vals, color_vals, tnight_hours):
    optical_on = "on" in (optical_vals or [])
    color_on = "on" in (color_vals or [])

    t_night_s = float(tnight_hours) * 3600.0 if optical_on else T_NIGHT_DEFAULT_S

    model_day = make_rate_model(
        A_log=float(A_log),
        f_live=float(f_live),
        t_overhead_s=float(t_overhead_s),
        omega_exp_deg2=float(omega_exp_deg2),
    )

    model_night = None
    if optical_on:
        f_live_night = float(f_live) * (t_night_s / DAY_S)
        model_night = make_rate_model(
            A_log=float(A_log),
            f_live=f_live_night,
            t_overhead_s=float(t_overhead_s),
            omega_exp_deg2=float(omega_exp_deg2),
        )

    X, Y_s, Z_plot, Z_raw, regime_id = compute_surface(
        model_day,
        model_night,
        int(i_det),
        optical_survey=optical_on,
        color_regimes=color_on,
        t_night_s=t_night_s,
        nx=280 if color_on else 220,
        ny=340 if color_on else 260,
    )

    Y_h = Y_s / 3600.0
    R_plot = np.where(np.isfinite(Z_plot), 10 ** Z_plot, np.nan)

    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr
    N_opt, t_cad_opt_s, log10R_opt = maximize_log_surface_iterative(
        model_day,
        model_night,
        int(i_det),
        x_min=0.0,
        x_max=np.log10(N_exp_max),
        y_min=0.0,
        y_max=8.0,
        optical_survey=optical_on,
        t_night_s=t_night_s,
    )
    R_opt = 10 ** log10R_opt if np.isfinite(log10R_opt) else np.nan
    t_cad_opt_hr = t_cad_opt_s / 3600.0 if np.isfinite(t_cad_opt_s) else np.nan

    t_exp_opt_s = np.nan
    if np.isfinite(N_opt) and np.isfinite(t_cad_opt_s):
        try:
            t_exp_opt_s = float(
                exposure_time_s(
                    SurveyStrategy(N_exp=float(N_opt), t_cad_s=float(t_cad_opt_s)),
                    model_day.instrument,
                )
            )
        except Exception:
            t_exp_opt_s = np.nan

    N_ztf = 27500.0 / 47.0
    t_cad_ztf_s = 2.0 * DAY_S
    t_cad_ztf_hr = t_cad_ztf_s / 3600.0
    log10_ztf = float(
        model_day.rate_log10(
            i_det=int(i_det),
            N_exp=np.array([N_ztf]),
            t_cad_s=np.array([t_cad_ztf_s]),
        )[0]
    )
    R_ztf = (10 ** log10_ztf) if np.isfinite(log10_ztf) else np.nan

    t_exp_ztf_s = np.nan
    try:
        t_exp_ztf_s = float(
            exposure_time_s(
                SurveyStrategy(N_exp=float(N_ztf), t_cad_s=float(t_cad_ztf_s)),
                model_day.instrument,
            )
        )
    except Exception:
        t_exp_ztf_s = np.nan

    R_candidates = []
    if np.any(np.isfinite(R_plot)):
        R_candidates.append(float(np.nanmax(R_plot)))
    if np.isfinite(R_opt):
        R_candidates.append(float(R_opt))
    if np.isfinite(R_ztf):
        R_candidates.append(float(R_ztf))
    Rmax = max(R_candidates) if R_candidates else 1.0

    fig = go.Figure()

    # Surface selection:
    # - optical_on: plot only sub-day surface as continuous (discrete part is drawn as lines)
    # - non-optical: plot full surface (no cadence restrictions)
    if optical_on:
        y_rows = np.asarray(Y_s[:, 0], dtype=float)
        keep_rows = np.where(y_rows < float(DAY_S))[0]
    else:
        keep_rows = np.arange(Y_s.shape[0], dtype=int)

    Xs = X[keep_rows, :]
    Ys = Y_h[keep_rows, :]
    Rs = R_plot[keep_rows, :]

    # Discrete day curves only for optical surveys
    if optical_on:
        N_cols = np.asarray(X[0, :], dtype=float)
        t_cad_max_s = float(np.nanmax(Y_s[:, 0])) if np.any(np.isfinite(Y_s[:, 0])) else 0.0

        if color_on and (regime_id is not None):
            _, regime_cols = discrete_regime_colorscale()
            _add_discrete_day_lines(
                fig,
                model_day=model_day,
                i_det=int(i_det),
                N_cols=N_cols,
                t_cad_max_s=t_cad_max_s,
                zmin_plot_log10=-1.0,
                color_mode="regime",
                regime_colors=regime_cols,
            )
        else:
            zmax_color = float(np.nanmax(Z_plot)) if np.any(np.isfinite(Z_plot)) else 0.0
            _add_discrete_day_lines(
                fig,
                model_day=model_day,
                i_det=int(i_det),
                N_cols=N_cols,
                t_cad_max_s=t_cad_max_s,
                zmin_plot_log10=-1.0,
                color_mode="height",
                height_cmin=-1.0,
                height_cmax=zmax_color,
                height_colorscale="Plasma",
            )

    # Surface trace
    if color_on and (regime_id is not None):
        cs, colors = discrete_regime_colorscale()
        Cs = regime_id[keep_rows, :]

        fig.add_trace(
            go.Surface(
                x=Xs,
                y=Ys,
                z=Rs,
                surfacecolor=Cs,
                cmin=1,
                cmax=7,
                colorscale=cs,
                showscale=False,
                connectgaps=False,
                lighting=dict(ambient=0.75, diffuse=0.75, specular=0.08, roughness=0.95),
                lightposition=dict(x=100, y=200, z=0),
                name="",
                showlegend=False,
            )
        )

        bl = boundary_lines_from_regimes(Xs, Ys, Rs, Cs)
        if bl is not None:
            fig.add_trace(bl)

        for idx, col in enumerate(colors, start=1):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=7, color=col),
                    name=f"A{idx}",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
    else:
        zmax_color = float(np.nanmax(Z_plot)) if np.any(np.isfinite(Z_plot)) else 0.0
        Zs = Z_plot[keep_rows, :]

        fig.add_trace(
            go.Surface(
                x=Xs,
                y=Ys,
                z=Rs,
                surfacecolor=Zs,
                cmin=-1.0,
                cmax=zmax_color,
                showscale=True,
                colorscale="Plasma",
                connectgaps=False,
                name="",
                colorbar=dict(title="log10 R_det"),
            )
        )

    # Markers
    if np.isfinite(N_opt) and np.isfinite(t_cad_opt_hr) and np.isfinite(R_opt):
        fig.add_trace(
            go.Scatter3d(
                x=[N_opt],
                y=[t_cad_opt_hr],
                z=[R_opt],
                mode="markers",
                marker=dict(size=7, color="black"),
                name="Grid maximum",
                hovertemplate=(
                    "Grid optimum"
                    + "<br>N_exp=%{x:.4g}"
                    + "<br>t_cad=%{y:.4g} h"
                    + (f"<br>t_exp={t_exp_opt_s:.4g} s" if np.isfinite(t_exp_opt_s) else "<br>t_exp=nan")
                    + "<br>R_det=%{z:.4g} yr⁻¹"
                    + "<extra></extra>"
                ),
            )
        )

    if np.isfinite(R_ztf):
        fig.add_trace(
            go.Scatter3d(
                x=[N_ztf],
                y=[t_cad_ztf_hr],
                z=[R_ztf],
                mode="markers",
                marker=dict(size=7, color="green"),
                name="ZTF (2 d cadence)",
                hovertemplate=(
                    "ZTF strategy"
                    + "<br>N_exp=%{x:.4g}"
                    + "<br>t_cad=%{y:.4g} h"
                    + (f"<br>t_exp={t_exp_ztf_s:.4g} s" if np.isfinite(t_exp_ztf_s) else "<br>t_exp=nan")
                    + "<br>R_det=%{z:.4g} yr⁻¹"
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        uirevision="keep-view-v1",
        scene=dict(
            xaxis=dict(title="N_exp", type="log"),
            yaxis=dict(title="t_cad [h]", type="log"),
            zaxis=dict(title="R_det [yr^-1]", type="log", range=[-1.0, np.log10(Rmax) + 0.05]),
            aspectmode="cube",
        ),
        scene_camera=dict(eye=dict(x=-1.25, y=-1.25, z=0.95), up=dict(x=0, y=0, z=1)),
        margin=dict(l=0, r=0, b=0, t=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02) if color_on else dict(),
    )

    omega_exp_sr = float(omega_exp_deg2) * DEG2_TO_SR

    status_children = [
        html.Div(RUNTIME_INFO),
        html.Div(
            f"i = {int(i_det)} | "
            f"A_log = {float(A_log):.2f} | "
            f"Ω_exp = {omega_exp_deg2:.1f} deg² ({omega_exp_sr:.4g} sr) | "
            f"f_live = {float(f_live):.2f} | "
            f"t_overhead = {float(t_overhead_s):.1f} s | "
            f"optical survey = {optical_on} | "
            f"color regimes = {color_on}"
        ),
    ]

    if optical_on:
        status_children.append(
            html.Div(
                f"t_night = {t_night_s / 3600.0:.3g} h | "
                f"f_live_night = {float(f_live) * (t_night_s / DAY_S):.4g}"
            )
        )

    status_children += [
        html.Br(),
        html.Div(
            [
                html.B("Grid optimum: "),
                f"N_exp = {N_opt:.3g}, ",
                f"t_cad = {t_cad_opt_hr:.3g} h = {t_cad_opt_s:.3g} s, ",
                f"t_exp = {t_exp_opt_s:.3g} s, ",
                f"R_det = {R_opt:.3g} yr⁻¹",
            ]
        ),
        html.Br(),
        html.Div(
            [
                html.B("ZTF strategy: "),
                f"N_exp = {N_ztf:.3g}, ",
                f"t_cad = {t_cad_ztf_hr:.3g} h = {t_cad_ztf_s:.3g} s, ",
                f"t_exp = {t_exp_ztf_s:.3g} s, ",
                f"R_det = {R_ztf:.3g} yr⁻¹",
            ]
        ),
    ]

    return fig, html.Div(status_children)


if __name__ == "__main__":
    app.run(debug=True)