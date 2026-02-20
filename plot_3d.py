import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from grb_detect.constants import DAY_S, DEG2_TO_SR
from grb_detect.params import SurveyDesignParams
from grb_detect.plot3d_core import (
    compute_surface,
    discrete_regime_colorscale,
    make_rate_model,
    maximize_log_surface_iterative,
)


def boundary_lines_from_regimes(X, Y, Z, regime_id):
    """Boundary overlay as polylines between adjacent grid cells of different regime_id."""
    if regime_id is None:
        return None

    rid = np.asarray(regime_id)
    valid = np.isfinite(rid) & np.isfinite(Z)

    xs, ys, zs = [], [], []

    # boundaries along y direction (adjacent rows)
    diff_y = (rid[1:, :] != rid[:-1, :]) & valid[1:, :] & valid[:-1, :]
    ii, jj = np.where(diff_y)
    for i, j in zip(ii, jj):
        xs.extend([X[i, j], X[i + 1, j], None])
        ys.extend([Y[i, j], Y[i + 1, j], None])
        zs.extend([Z[i, j], Z[i + 1, j], None])

    # boundaries along x direction (adjacent columns)
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


app = Dash(__name__)
server = app.server  # for WSGI servers (gunicorn)

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
        html.H2(
            "GRB Detection Rate Surface",
            style={"margin": "0", "paddingTop": "10px", "paddingBottom": "6px"},
        ),
        dcc.Graph(
            id="surface",
            style={"height": "780px", "marginTop": "0px", "paddingTop": "0px"},
        ),
        html.Div(
            style={
                "display": "flex",
                "gap": "28px",
                "alignItems": "center",
                "marginTop": "6px",
                "marginBottom": "10px",
            },
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
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(320px, 1fr))",
                "gap": "12px 18px",
                "alignItems": "center",
            },
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


@app.callback(
    Output("tnight_container", "style"),
    Input("optical_checkbox", "value"),
)
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

    # Use slider value only when optical survey is enabled.
    t_night_s = float(tnight_hours) * 3600.0 if optical_on else T_NIGHT_DEFAULT_S

    model_day = make_rate_model(
        A_log=float(A_log),
        f_live=float(f_live),
        t_overhead_s=float(t_overhead_s),
        omega_exp_deg2=float(omega_exp_deg2),
    )

    model_night = None
    if optical_on:
        # Effective live fraction within a night, scaled only by t_night / 24 hr.
        f_live_night = float(f_live) * (t_night_s / DAY_S)

        model_night = make_rate_model(
            A_log=float(A_log),
            f_live=f_live_night,
            t_overhead_s=float(t_overhead_s),
            omega_exp_deg2=float(omega_exp_deg2),
        )

    X, Y, Z_plot, Z_raw, regime_id = compute_surface(
        model_day,
        model_night,
        int(i_det),
        optical_survey=optical_on,
        color_regimes=color_on,
        t_night_s=t_night_s,
        nx=280 if color_on else 220,
        ny=340 if color_on else 260,
    )

    # Surface optimum (consistent with the optical-survey mapping if enabled)
    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr
    N_opt, t_cad_opt_s, log10R_opt = maximize_log_surface_iterative(
        model_day,
        model_night,
        int(i_det),
        x_min=0.0,
        x_max=np.log10(N_exp_max),
        y_min=-8.0,
        y_max=8.0,
        optical_survey=optical_on,
        t_night_s=t_night_s,
    )
    R_opt = 10 ** log10R_opt if np.isfinite(log10R_opt) else np.nan
    t_cad_opt_hr = t_cad_opt_s / 3600.0 if np.isfinite(t_cad_opt_s) else np.nan

    # ZTF strategy (fixed point)
    N_ztf = 27500.0 / 47.0
    t_cad_ztf_s = 2.0 * DAY_S
    x_ztf = np.log10(N_ztf)
    y_ztf = np.log10(t_cad_ztf_s)
    z_ztf = float(
        model_day.rate_log10(
            i_det=int(i_det),
            N_exp=np.array([N_ztf]),
            t_cad_s=np.array([t_cad_ztf_s]),
        )[0]
    )

    z_candidates = []
    if np.any(np.isfinite(Z_plot)):
        z_candidates.append(float(np.nanmax(Z_plot)))
    if np.isfinite(log10R_opt):
        z_candidates.append(float(log10R_opt))
    if np.isfinite(z_ztf):
        z_candidates.append(float(z_ztf))
    zmax = max(z_candidates) if z_candidates else -1.0
    zmax_plot = zmax + 0.06

    fig = go.Figure()

    if color_on and (regime_id is not None):
        cs, colors = discrete_regime_colorscale()

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z_plot,
                surfacecolor=regime_id,
                cmin=1,
                cmax=7,
                colorscale=cs,
                showscale=False,
                connectgaps=False,
                lighting=dict(ambient=0.75, diffuse=0.75, specular=0.08, roughness=0.95),
                lightposition=dict(x=100, y=200, z=0),
                name="Regimes",
            )
        )

        bl = boundary_lines_from_regimes(X, Y, Z_plot, regime_id)
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
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z_plot,
                showscale=True,
                colorscale="Plasma",
                connectgaps=False,
                name="log10 R_det",
            )
        )

    if np.isfinite(N_opt) and np.isfinite(t_cad_opt_s) and np.isfinite(log10R_opt):
        fig.add_trace(
            go.Scatter3d(
                x=[np.log10(N_opt)],
                y=[np.log10(t_cad_opt_s)],
                z=[log10R_opt],
                mode="markers",
                marker=dict(size=7, color="black"),
                name="Grid maximum",
            )
        )

    if np.isfinite(z_ztf):
        fig.add_trace(
            go.Scatter3d(
                x=[x_ztf],
                y=[y_ztf],
                z=[z_ztf],
                mode="markers",
                marker=dict(size=7, color="green"),
                name="ZTF (2 d cadence)",
            )
        )

    fig.update_layout(
        uirevision="keep-view-v1",
        scene=dict(
            xaxis_title="log10 N_exp",
            yaxis_title="log10 t_cad [s]",
            zaxis_title="log10 R_det [yr^-1]",
            zaxis=dict(range=[-1, float(zmax_plot)]),
        ),
        scene_camera=dict(
            eye=dict(x=-1.25, y=-1.25, z=0.95),
            up=dict(x=0, y=0, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=5),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.02,
        )
        if color_on
        else dict(),
    )

    omega_exp_sr = float(omega_exp_deg2) * DEG2_TO_SR

    status_children = [
        html.Div(
            f"i = {int(i_det)} | "
            f"A_log = {float(A_log):.2f} | "
            f"Ω_exp = {omega_exp_deg2:.1f} deg² ({omega_exp_sr:.4g} sr) | "
            f"f_live = {float(f_live):.2f} | "
            f"t_overhead = {float(t_overhead_s):.1f} s | "
            f"optical survey = {optical_on} | "
            f"color regimes = {color_on}"
        )
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
                f"R_det = {R_opt:.3g} yr⁻¹",
            ]
        ),
        html.Br(),
        html.Div(
            [
                html.B("ZTF strategy: "),
                f"N_exp = {N_ztf:.3g}, ",
                f"t_cad = {t_cad_ztf_s / 3600.0:.3g} h = {t_cad_ztf_s:.3g} s, ",
                f"R_det = {(10 ** z_ztf):.3g} yr⁻¹",
            ]
        ),
    ]

    return fig, html.Div(status_children)


if __name__ == "__main__":
    app.run(debug=True)
