# plot_3d.py
import os
import sys

import numpy as np
import plotly
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State

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
DEBUG_MODE: bool = os.environ.get("GRB_DEBUG", "0").lower() in ("1", "true", "yes")

# ZTF survey parameters (reference strategy)
ZTF_OMEGA_EXP_DEG2: float = 47.0   # ZTF field of view [deg²]

# Grid dimensions for the rate surface
NX_REGIME, NY_REGIME = 280, 340   # higher resolution for regime coloring
NX_DEFAULT, NY_DEFAULT = 220, 260  # default resolution

# Log10 of the minimum displayed detection rate
ZMIN_DISPLAY_LOG10: float = -1.0

# Day-line appearance: regime mode uses wider, more opaque lines; height mode uses thin thread + small markers
_DL_WIDTH_REGIME = 2.5
_DL_WIDTH_HEIGHT = 1.0
_DL_MARKER_HEIGHT = 3  # small markers coloured by rate value in height mode

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
                    col = _hex_to_rgba(regime_colors[k - 1], alpha=0.80)

                    fig.add_trace(
                        go.Scatter3d(
                            x=x[start:end],
                            y=y[start:end],
                            z=z[start:end],
                            mode="lines+markers",
                            line=dict(color=col, width=_DL_WIDTH_REGIME),
                            marker=dict(size=0, opacity=0.001),
                            showlegend=False,
                            hovertemplate=XYZ_HOVER,
                        )
                    )

                start = end

        else:
            # Height coloring: thin transparent connector + small rate-coloured markers
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines+markers",
                    line=dict(color="rgba(255,255,255,0.12)", width=_DL_WIDTH_HEIGHT),
                    marker=dict(
                        size=_DL_MARKER_HEIGHT,
                        color=log10R[good],
                        colorscale=height_colorscale,
                        cmin=float(height_cmin),
                        cmax=float(height_cmax),
                        opacity=1.0,
                        showscale=False,
                    ),
                    showlegend=False,
                    hovertemplate=XYZ_HOVER,
                )
            )

app = Dash(__name__)
server = app.server

survey_defaults = SurveyDesignParams()
T_NIGHT_DEFAULT_S = survey_defaults.t_night_s
OMEGA_SRV_DEFAULT_DEG2 = 27500.0

def _param_block(label, slider_id, input_id, slider_min, slider_max, slider_step,
                 slider_val, marks, input_min=None, input_max=None, input_step=None):
    """Return a labeled slider block with an editable number input."""
    return html.Div(className="slider-wrap", children=[
        html.Div(className="param-row", children=[
            html.Label(label, className="slider-label"),
            dcc.Input(
                id=input_id, type="number", value=slider_val,
                min=input_min if input_min is not None else slider_min,
                max=input_max if input_max is not None else slider_max,
                step=input_step if input_step is not None else slider_step,
                className="param-input", debounce=False,
            ),
        ]),
        dcc.Slider(slider_min, slider_max, slider_step, value=slider_val,
                   id=slider_id, marks=marks),
    ])


app.layout = html.Div(
    id="app-root",
    style={"display": "flex", "flexDirection": "column", "height": "100vh", "width": "100vw", "overflow": "hidden"},
    children=[
        dcc.Store(id="theme-store", data="dark"),
        dcc.Store(id="_dom-theme-dummy"),

        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            className="header-bar",
            children=[
                html.Button("✕", id="sidebar-toggle", className="icon-btn", n_clicks=0),
                html.H2("GRB Detection Rate Surface"),
                html.Button("☾ Dark", id="theme-toggle", className="icon-btn", n_clicks=0),
            ],
        ),

        # ── Body: sidebar + plot area ────────────────────────────────────────
        html.Div(
            style={"display": "flex", "flex": "1", "overflow": "hidden"},
            children=[

                # ── Collapsible sidebar ──────────────────────────────────────
                html.Div(
                    id="sidebar",
                    className="sidebar open",
                    children=[html.Div(
                        className="sidebar-inner",
                        children=[
                            # ── STRATEGY ──────────────────────────────────────
                            html.Div("Strategy", className="slider-group-title"),
                            _param_block("i", "i_slider", "i_input",
                                         2, 300, 1, 10,
                                         {2: "2", 10: "10", 30: "30", 100: "100", 300: "300"}),
                            _param_block("f_live", "flive_slider", "flive_input",
                                         0.01, 1.0, 0.01, 0.2,
                                         {0.01: "0.01", 0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
                                         input_step=0.01),

                            # ── INSTRUMENT ────────────────────────────────────
                            html.Div("Instrument", className="slider-group-title"),
                            _param_block("log A [Jy]", "Alog_slider", "Alog_input",
                                         -12, -2, 0.01, -4.68,
                                         {-12: "-12", -8: "-8", -4.68: "-4.68", -2: "-2"},
                                         input_step=0.01),
                            _param_block("Ω_exp [deg²]", "omegaexp_slider", "omegaexp_input",
                                         1, 200, 1, 47,
                                         {1: "1", 47: "47", 100: "100", 200: "200"}),
                            _param_block("t_overhead [s]", "toh_slider", "toh_input",
                                         0.0, 30.0, 0.5, 0.0,
                                         {0: "0", 10: "10", 20: "20", 30: "30"},
                                         input_step=0.5),

                            # ── CONSTRAINTS ───────────────────────────────────
                            html.Div("Constraints", className="slider-group-title"),
                            _param_block("Ω_srv max [deg²]", "omega_srv_slider", "omega_srv_input",
                                         100, 41253, 100, OMEGA_SRV_DEFAULT_DEG2,
                                         {100: "100", 10000: "10 k", 27500: "27.5 k", 41253: "41.3 k"},
                                         input_step=100),
                            html.Div(
                                id="tnight_container",
                                style={"display": "none"},
                                children=[
                                    _param_block("t_night [h]", "tnight_slider", "tnight_input",
                                                 4.0, 14.0, 0.25, T_NIGHT_DEFAULT_S / 3600.0,
                                                 {4: "4", 6: "6", 8: "8", 10: "10", 12: "12", 14: "14"},
                                                 input_step=0.25),
                                ],
                            ),
                        ],
                    )],
                ),

                # ── Plot area ────────────────────────────────────────────────
                html.Div(
                    style={"flex": "1", "display": "flex", "flexDirection": "column",
                           "overflow": "hidden", "minWidth": "0"},
                    children=[
                        html.Div(
                            className="toolbar-row",
                            children=[
                                dcc.Checklist(
                                    id="optical_checkbox",
                                    options=[{"label": "  Optical survey", "value": "on"}],
                                    value=[], inline=True,
                                ),
                                dcc.Checklist(
                                    id="color_checkbox",
                                    options=[{"label": "  Color regimes", "value": "on"}],
                                    value=[], inline=True,
                                ),
                            ],
                        ),
                        dcc.Graph(id="surface", style={"flex": "1", "minHeight": "0"}),
                        html.Div(id="status", className="status-bar"),
                    ],
                ),
            ],
        ),
    ],
)


# ── Sidebar toggle ────────────────────────────────────────────────────────────
@app.callback(
    Output("sidebar", "className"),
    Output("sidebar-toggle", "children"),
    Input("sidebar-toggle", "n_clicks"),
)
def toggle_sidebar(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return "sidebar", "☰"
    return "sidebar open", "✕"


# ── t_night visibility ────────────────────────────────────────────────────────
@app.callback(Output("tnight_container", "style"), Input("optical_checkbox", "value"))
def toggle_tnight_slider(optical_vals):
    optical_on = "on" in (optical_vals or [])
    return {"display": "block"} if optical_on else {"display": "none"}


# ── Theme toggle (server) ─────────────────────────────────────────────────────
@app.callback(
    Output("theme-store", "data"),
    Output("theme-toggle", "children"),
    Input("theme-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_theme(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return "light", "☀ Light"
    return "dark", "☾ Dark"


# ── Theme apply (clientside) — sets CSS custom properties on <html> ───────────
app.clientside_callback(
    """
    function(theme) {
        document.documentElement.setAttribute("data-theme", theme || "dark");
        return window.dash_clientside.no_update;
    }
    """,
    Output("_dom-theme-dummy", "data"),
    Input("theme-store", "data"),
)


# ── Slider ↔ input sync ───────────────────────────────────────────────────────
# Format: (slider_id, input_id, clamp_min, clamp_max)
_SYNC_PAIRS = [
    ("i_slider",         "i_input",         2,      300),
    ("flive_slider",     "flive_input",     0.01,   1.0),
    ("Alog_slider",      "Alog_input",      -12,    -2),
    ("omegaexp_slider",  "omegaexp_input",  1,      200),
    ("toh_slider",       "toh_input",       0.0,    30.0),
    ("omega_srv_slider", "omega_srv_input", 100,    41253),
    ("tnight_slider",    "tnight_input",    4.0,    14.0),
]

# Slider → input: instant clientside update (no server round-trip)
for _sid, _iid, *_ in _SYNC_PAIRS:
    app.clientside_callback(
        "function(v) { return (v === null || v === undefined) ? window.dash_clientside.no_update : v; }",
        Output(_iid, "value"),
        Input(_sid, "value"),
    )


def _make_input_to_slider_cb(slider_id: str, input_id: str, lo: float, hi: float):
    """Register a callback: input.n_blur → slider.value (with clamping)."""
    @app.callback(
        Output(slider_id, "value"),
        Input(input_id, "n_blur"),
        State(input_id, "value"),
        prevent_initial_call=True,
    )
    def _fn(_, v):
        if v is None:
            return dash.no_update
        return float(np.clip(float(v), lo, hi))


for _sid, _iid, _lo, _hi in _SYNC_PAIRS:
    _make_input_to_slider_cb(_sid, _iid, _lo, _hi)


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
    Input("omega_srv_slider", "value"),
    Input("theme-store", "data"),
)
def update_surface(i_det, A_log, omega_exp_deg2, f_live, t_overhead_s, optical_vals, color_vals, tnight_hours,
                   omega_srv_deg2, theme):
    optical_on = "on" in (optical_vals or [])
    color_on = "on" in (color_vals or [])

    t_night_s = float(tnight_hours) * 3600.0 if optical_on else T_NIGHT_DEFAULT_S
    design = SurveyDesignParams(omega_survey_max_sr=float(omega_srv_deg2) * DEG2_TO_SR)
    plotly_template = "plotly_dark" if (theme or "dark") == "dark" else "plotly_white"

    model_night = None
    if optical_on:
        f_live_night = float(f_live) * (t_night_s / DAY_S)
        # Multi-day cadence (n days): survey spread over n nights → t_exp = n × f_live × t_night / N_exp
        model_day = make_rate_model(
            A_log=float(A_log),
            f_live=f_live_night,
            t_overhead_s=float(t_overhead_s),
            omega_exp_deg2=float(omega_exp_deg2),
            design=design,
        )
        # Sub-day cadence: multiple visits per night → t_exp = f_live × t_cad / N_exp
        model_night = make_rate_model(
            A_log=float(A_log),
            f_live=float(f_live),
            t_overhead_s=float(t_overhead_s),
            omega_exp_deg2=float(omega_exp_deg2),
            design=design,
        )
    else:
        model_day = make_rate_model(
            A_log=float(A_log),
            f_live=float(f_live),
            t_overhead_s=float(t_overhead_s),
            omega_exp_deg2=float(omega_exp_deg2),
            design=design,
        )

    X, Y_s, Z_plot, Z_raw, regime_id = compute_surface(
        model_day,
        model_night,
        int(i_det),
        optical_survey=optical_on,
        color_regimes=color_on,
        t_night_s=t_night_s,
        nx=NX_REGIME if color_on else NX_DEFAULT,
        ny=NY_REGIME if color_on else NY_DEFAULT,
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
            # Select instrument with the correct f_live for the optimal cadence regime
            opt_instrument = (
                model_night.instrument
                if (optical_on and model_night is not None and t_cad_opt_s < DAY_S)
                else model_day.instrument
            )
            t_exp_opt_s = float(
                exposure_time_s(
                    SurveyStrategy(N_exp=float(N_opt), t_cad_s=float(t_cad_opt_s)),
                    opt_instrument,
                )
            )
        except Exception:
            t_exp_opt_s = np.nan

    N_ztf = OMEGA_SRV_DEFAULT_DEG2 / ZTF_OMEGA_EXP_DEG2
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
                zmin_plot_log10=ZMIN_DISPLAY_LOG10,
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
                zmin_plot_log10=ZMIN_DISPLAY_LOG10,
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

        regime_labels = [
            "Flux-limited · decel.",
            "Flux-limited · post-jet",
            "Flux-limited · pre-jet",
            "Cadence-limited · decel.",
            "Cadence-limited · post-jet",
            "Cadence-limited · pre-jet",
            "Doubly limited",
        ]
        for col, label in zip(colors, regime_labels):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=7, color=col),
                    name=label,
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
                colorbar=dict(
                    title=dict(text="log₁₀ R", side="right"),
                    x=0.85, xanchor="left",
                    thickness=14, len=0.65,
                    y=0.45, yanchor="middle",
                ),
            )
        )

    # Markers
    if np.isfinite(N_opt) and np.isfinite(t_cad_opt_hr) and np.isfinite(R_opt):
        fig.add_trace(
            go.Scatter3d(
                x=[N_opt],
                y=[t_cad_opt_hr],
                z=[R_opt],
                mode="markers+text",
                marker=dict(size=10, color="#FFD700", symbol="diamond"),
                text=["Optimum"],
                textposition="top center",
                name="Optimum",
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
                mode="markers+text",
                marker=dict(size=9, color="#FF6B6B", symbol="circle"),
                text=["ZTF"],
                textposition="top center",
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
        template=plotly_template,
        scene=dict(
            xaxis=dict(title="N_exp", type="log"),
            yaxis=dict(
                title="t_cad",
                type="log",
                tickvals=[1 / 3600, 1 / 60, 1, 6, 24, 168, 730, 8760],
                ticktext=["1 s", "1 min", "1 h", "6 h", "1 d", "1 wk", "1 mo", "1 yr"],
            ),
            zaxis=dict(title="R_det [yr^-1]", type="log", range=[ZMIN_DISPLAY_LOG10, np.log10(Rmax) + 0.05]),
            aspectmode="cube",
        ),
        scene_camera=dict(eye=dict(x=-1.25, y=-1.25, z=0.95), up=dict(x=0, y=0, z=1)),
        margin=dict(l=0, r=0, b=0, t=5),
        legend=dict(
            orientation="v",
            x=0.01, y=0.98,
            xanchor="left", yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    omega_exp_sr = float(omega_exp_deg2) * DEG2_TO_SR

    debug_items = [
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
        debug_items.append(
            html.Div(
                f"t_night = {t_night_s / 3600.0:.3g} h | "
                f"f_live_night = {float(f_live) * (t_night_s / DAY_S):.4g}"
            )
        )

    status_children: list = []
    if DEBUG_MODE:
        status_children.append(
            html.Details(
                [html.Summary("Debug info"), *debug_items],
                style={"fontSize": "0.82em", "color": "#888", "marginBottom": "4px"},
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