import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from grb_detect.constants import DAY_S
from grb_detect.detection_rate import DesmosRateModel
from grb_detect.params import AfterglowPhysicalParams, SurveyInstrumentParams

# Keep consistent with params.py
DEG2_TO_SR: float = 0.00030461741978670857  # (pi/180)^2

# Default duration of an astronomical night (used by the optical-survey cadence logic)
T_NIGHT_DEFAULT_S: float = 10.0 * 3600.0


def make_model(A_log: float, f_live: float, t_overhead_s: float, omega_exp_deg2: float) -> DesmosRateModel:
    phys = AfterglowPhysicalParams()
    instrument = SurveyInstrumentParams(
        omega_exp_sr=float(omega_exp_deg2) * DEG2_TO_SR,
        F_lim_ref_Jy=10 ** float(A_log),
        f_live=float(f_live),
        t_overhead_s=float(t_overhead_s),
    )
    return DesmosRateModel(phys=phys, instrument=instrument)


def quantize_tcad_seconds(t_s: np.ndarray) -> np.ndarray:
    """
    Quantize cadence:
      - if >= 1 day: ceil to integer days (1d, 2d, 3d, ...)
      - if < 1 day: allowed values are DAY/2^k (k=1,2,3,...) and we round UP
        (12h, 6h, 3h, 1.5h, ...)
    """
    t = np.asarray(t_s, dtype=float)
    out = np.empty_like(t)

    ge_day = t >= DAY_S
    out[ge_day] = np.ceil(t[ge_day] / DAY_S) * DAY_S

    lt_day = ~ge_day
    if np.any(lt_day):
        ratio = DAY_S / np.maximum(t[lt_day], 1e-300)
        k = np.floor(np.log2(ratio)).astype(int)
        k = np.maximum(k, 1)
        out[lt_day] = DAY_S / (2.0 ** k)

    return out


def optical_survey_tcad_seconds(
    t_cad_s: np.ndarray,
    *,
    i_det: int,
    t_night_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a continuous cadence to an "optical survey" effective cadence.

    Rules:
      - For t_cad >= 1 day: t_eff is quantized to integer multiples of a day (ceil).
      - For t_night <= t_cad < 1 day: cadences are ineffective and mapped to 1 day.
      - For t_cad < t_night: t_eff = t_cad (continuous), but must satisfy i_det * t_cad < t_night.
        Values that violate this are marked invalid.

    Returns:
      (t_eff_s, valid_mask)
    """
    t = np.asarray(t_cad_s, dtype=float)
    t_eff = np.array(t, copy=True)
    valid = np.isfinite(t_eff) & (t_eff > 0.0)

    # >= 1 day: quantize to integer day multiples
    ge_day = valid & (t_eff >= DAY_S)
    if np.any(ge_day):
        t_eff[ge_day] = np.ceil(t_eff[ge_day] / DAY_S) * DAY_S

    # night-to-day gap: map to 1 day
    gap = valid & (t_eff >= t_night_s) & (t_eff < DAY_S)
    if np.any(gap):
        t_eff[gap] = DAY_S

    # < t_night: continuous but must allow i detections within a night
    sub_night = valid & (t_eff < t_night_s)
    if np.any(sub_night):
        valid[sub_night] &= (float(i_det) * t_eff[sub_night] < t_night_s)

    return t_eff, valid


def discrete_regime_colorscale():
    """
    Step colors for regime_id in {1..7}.
    Uses duplicated stops to avoid interpolation.
    """
    colors = [
        "#6A3D9A",  # A1 purple
        "#1F78B4",  # A2 blue
        "#33A02C",  # A3 green
        "#FF7F00",  # A4 orange
        "#E31A1C",  # A5 red
        "#A6CEE3",  # A6 light blue
        "#B15928",  # A7 brown
    ]
    cs = []
    for k, c in enumerate(colors, start=1):
        a = (k - 1) / 7.0
        b = k / 7.0
        cs.append([a, c])
        cs.append([b, c])
    return cs, colors


def compute_surface(
    model_day: DesmosRateModel,
    model_night: DesmosRateModel | None,
    i_det: int,
    *,
    optical_survey: bool,
    color_regimes: bool,
    t_night_s: float,
    nx: int = 220,
    ny: int = 260,
):
    """
    Returns:
      X_log, Y_log,
      Z_plot (masked for display),
      Z_raw (unmasked),
      regime_id (or None)
    """
    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr
    x_min, x_max = 0.0, np.log10(N_exp_max)
    y_min, y_max = -8.0, 8.0

    logN = np.linspace(x_min, x_max, nx)
    logtcad = np.linspace(y_min, y_max, ny)
    X_log, Y_log = np.meshgrid(logN, logtcad)

    N_exp = 10 ** X_log
    t_cad_s = 10 ** Y_log

    if optical_survey:
        t_cad_eff, valid = optical_survey_tcad_seconds(
            t_cad_s,
            i_det=int(i_det),
            t_night_s=float(t_night_s),
        )
        is_subday = t_cad_eff < DAY_S

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
    Z_plot = np.where(np.isfinite(Z_raw) & (Z_raw >= -1.0), Z_raw, np.nan)

    regime_id = None
    if color_regimes:
        regime_id = np.full(Z_raw.shape, np.nan, dtype=float)

        def fill_regimes(model: DesmosRateModel, sel: np.ndarray):
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
            is_subday = t_cad_eff < DAY_S
            fill_regimes(model_day, ~is_subday)
            fill_regimes(model_night, is_subday)
        else:
            fill_regimes(model_day, np.ones_like(Z_raw, dtype=bool))

    return X_log, Y_log, Z_plot, Z_raw, regime_id


def boundary_lines_from_regimes(X, Y, Z, regime_id):
    """
    Boundary overlay as polylines between adjacent grid cells of different regime_id.
    Makes boundaries crisp regardless of surface triangulation.
    """
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
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.9)", width=4),
        name="Regime boundaries",
        showlegend=False,
        hoverinfo="skip",
    )


def maximize_log_surface_iterative(
    model_day: DesmosRateModel,
    model_night: DesmosRateModel | None,
    i_det: int,
    x_min, x_max, y_min, y_max,
    *,
    optical_survey: bool,
    t_night_s: float,
    n0x=180, n0y=220, n_refine=3, zoom=0.18,
    nfx=280, nfy=320,
):
    def eval_grid(x0, x1, y0, y1, nx, ny):
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        X, Y = np.meshgrid(xs, ys)
        N = 10 ** X
        t = 10 ** Y

        if optical_survey:
            t_eff, valid = optical_survey_tcad_seconds(
                t,
                i_det=int(i_det),
                t_night_s=float(t_night_s),
            )
            is_subday = t_eff < DAY_S
            if model_night is None:
                return None
            Z_day = model_day.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
            Z_night = model_night.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t_eff)
            Z = np.where(is_subday, Z_night, Z_day)
            Z = np.where(valid, Z, np.nan)
        else:
            Z = model_day.rate_log10(i_det=i_det, N_exp=N, t_cad_s=t)

        Z = np.where(np.isfinite(Z), Z, np.nan)
        if not np.any(np.isfinite(Z)):
            return None

        k = np.nanargmax(Z)
        ii, jj = np.unravel_index(k, Z.shape)
        return float(X[ii, jj]), float(Y[ii, jj]), float(Z[ii, jj])

    best = eval_grid(x_min, x_max, y_min, y_max, n0x, n0y)
    if best is None:
        return np.nan, np.nan, np.nan
    x0, y0, z0 = best

    for _ in range(n_refine):
        dx = (x_max - x_min) * zoom
        dy = (y_max - y_min) * zoom
        xa0, xa1 = max(x_min, x0 - dx), min(x_max, x0 + dx)
        ya0, ya1 = max(y_min, y0 - dy), min(y_max, y0 + dy)

        best = eval_grid(xa0, xa1, ya0, ya1, nfx, nfy)
        if best is None:
            break
        x0, y0, z0 = best
        x_min, x_max, y_min, y_max = xa0, xa1, ya0, ya1

    return 10 ** x0, 10 ** y0, z0


app = Dash(__name__)

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

        # tick boxes row
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
                    id="quantize_checkbox",
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

        # sliders
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(320px, 1fr))",
                "gap": "12px 18px",
                "alignItems": "center",
            },
            children=[
                html.Div([
                    html.Label("i (detections required)", style={"fontSize": "14px"}),
                    dcc.Slider(2, 300, 1, value=10, id="i_slider",
                               marks={2: "2", 10: "10", 30: "30", 150: "150", 300: "300"}),
                ]),
                html.Div([
                    html.Label("log(A [Jy])", style={"fontSize": "14px"}),
                    dcc.Slider(-12, -2, 0.01, value=-4.68, id="Alog_slider",
                               marks={-12: "-12", -8: "-8", -4.68: "-4.68", -4: "-4", -2: "-2"}),
                ]),
                html.Div([
                    html.Label("Ω_exp [deg²]", style={"fontSize": "14px"}),
                    dcc.Slider(1, 200, 1, value=47, id="omegaexp_slider",
                               marks={1: "1", 47: "47", 100: "100", 200: "200"}),
                ]),
                html.Div([
                    html.Label("f_live", style={"fontSize": "14px"}),
                    dcc.Slider(0.01, 1.0, 0.01, value=0.1, id="flive_slider",
                               marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2", 0.5: "0.5", 1.0: "1.0"}),
                ]),
                html.Div([
                    html.Label("t_overhead [s]", style={"fontSize": "14px"}),
                    dcc.Slider(0.0, 30.0, 0.5, value=0.0, id="toh_slider",
                               marks={0: "0", 10: "10", 15: "15", 20: "20", 30: "30"}),
                ]),

                # t_night slider container (shown only if Optical survey is on)
                html.Div(
                    id="tnight_container",
                    style={"display": "none"},
                    children=[
                        html.Label("t_night [h]", style={"fontSize": "14px"}),
                        dcc.Slider(
                            4.0, 14.0, 0.25,
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
    Input("quantize_checkbox", "value"),
)
def toggle_tnight_slider(quantize_vals):
    optical_on = "on" in (quantize_vals or [])
    if optical_on:
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("surface", "figure"),
    Output("status", "children"),
    Input("i_slider", "value"),
    Input("Alog_slider", "value"),
    Input("omegaexp_slider", "value"),
    Input("flive_slider", "value"),
    Input("toh_slider", "value"),
    Input("quantize_checkbox", "value"),
    Input("color_checkbox", "value"),
    Input("tnight_slider", "value"),
)
def update_surface(i_det, A_log, omega_exp_deg2, f_live, t_overhead_s, quantize_vals, color_vals, tnight_hours):
    optical_on = "on" in (quantize_vals or [])
    color_on = "on" in (color_vals or [])

    # Use slider value only when optical survey is enabled.
    # Otherwise, keep the default, but it will not affect anything because optical_survey=False.
    t_night_s = float(tnight_hours) * 3600.0 if optical_on else T_NIGHT_DEFAULT_S

    model_day = make_model(
        A_log=float(A_log),
        f_live=float(f_live),
        t_overhead_s=float(t_overhead_s),
        omega_exp_deg2=float(omega_exp_deg2),
    )

    model_night = None
    if optical_on:
        # Requested rule: only factor is t_night / 24 hr.
        f_live_night = float(f_live) * (t_night_s / DAY_S)

        model_night = make_model(
            A_log=float(A_log),
            f_live=f_live_night,
            t_overhead_s=float(t_overhead_s),
            omega_exp_deg2=float(omega_exp_deg2),
        )

    # Higher resolution when coloring regimes (improves boundary geometry)
    X, Y, Z_plot, Z_raw, regime_id = compute_surface(
        model_day, model_night, int(i_det),
        optical_survey=optical_on,
        color_regimes=color_on,
        t_night_s=t_night_s,
        nx=280 if color_on else 220,
        ny=340 if color_on else 260,
    )

    # Optimum consistent with optical survey option
    N_exp_max = model_day.instrument.omega_survey_max_sr / model_day.instrument.omega_exp_sr
    N_opt, t_cad_opt_s, log10R_opt = maximize_log_surface_iterative(
        model_day, model_night, int(i_det),
        x_min=0.0, x_max=np.log10(N_exp_max),
        y_min=-8.0, y_max=8.0,
        optical_survey=optical_on,
        t_night_s=t_night_s,
    )
    R_opt = 10 ** log10R_opt if np.isfinite(log10R_opt) else np.nan
    t_cad_opt_hr = t_cad_opt_s / 3600.0 if np.isfinite(t_cad_opt_s) else np.nan

    # ZTF strategy (fixed)
    N_ztf = 27500.0 / 47.0
    t_cad_ztf_s = 2.0 * DAY_S
    x_ztf = np.log10(N_ztf)
    y_ztf = np.log10(t_cad_ztf_s)
    z_ztf = float(model_day.rate_log10(
        i_det=int(i_det),
        N_exp=np.array([N_ztf]),
        t_cad_s=np.array([t_cad_ztf_s]),
    )[0])

    # Robust z-axis top (include markers)
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

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_plot,
            surfacecolor=regime_id,
            cmin=1, cmax=7,
            colorscale=cs,
            showscale=False,
            connectgaps=False,
            lighting=dict(ambient=0.75, diffuse=0.75, specular=0.08, roughness=0.95),
            lightposition=dict(x=100, y=200, z=0),
            name="Regimes",
        ))

        bl = boundary_lines_from_regimes(X, Y, Z_plot, regime_id)
        if bl is not None:
            fig.add_trace(bl)

        for idx, col in enumerate(colors, start=1):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=7, color=col),
                name=f"A{idx}",
                showlegend=True,
                hoverinfo="skip",
            ))
    else:
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_plot,
            showscale=True,
            colorscale="Plasma",
            connectgaps=False,
            name="log10 R_det",
        ))

    if np.isfinite(N_opt) and np.isfinite(t_cad_opt_s) and np.isfinite(log10R_opt):
        fig.add_trace(go.Scatter3d(
            x=[np.log10(N_opt)],
            y=[np.log10(t_cad_opt_s)],
            z=[log10R_opt],
            mode="markers",
            marker=dict(size=7, color="black"),
            name="Grid maximum",
        ))

    if np.isfinite(z_ztf):
        fig.add_trace(go.Scatter3d(
            x=[x_ztf], y=[y_ztf], z=[z_ztf],
            mode="markers",
            marker=dict(size=7, color="green"),
            name="ZTF (2 d cadence)",
        ))

    fig.update_layout(
        uirevision="keep-view-v5",
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
        ) if color_on else dict(),
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
        status_children.append(html.Div(
            f"t_night = {t_night_s / 3600.0:.3g} h | "
            f"f_live_night = {float(f_live) * (t_night_s / DAY_S):.4g}"
        ))

    status_children += [
        html.Br(),
        html.Div([
            html.B("Grid optimum: "),
            f"N_exp = {N_opt:.3g}, ",
            f"t_cad = {t_cad_opt_hr:.3g} h = {t_cad_opt_s:.3g} s, ",
            f"R_det = {R_opt:.3g} yr⁻¹"
        ]),
        html.Br(),
        html.Div([
            html.B("ZTF strategy: "),
            f"N_exp = {N_ztf:.3g}, ",
            f"t_cad = {t_cad_ztf_s / 3600.0:.3g} h = {t_cad_ztf_s:.3g} s, ",
            f"R_det = {(10 ** z_ztf):.3g} yr⁻¹"
        ]),
    ]

    return fig, html.Div(status_children)


if __name__ == "__main__":
    app.run(debug=True)
