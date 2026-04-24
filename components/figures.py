# components/figures.py
"""Plotly figure builders for the GRB Detection Rate Explorer.

Contains all figure-construction logic (3D surface + 2D slices) with
zero Dash callback code. Physics is imported read-only; nothing here
modifies the grb_detect layer.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html

from grb_detect.constants import DAY_S
from grb_detect.plot_3d_core import ZMIN_DISPLAY_LOG10, _rate
# ── Regime palette ───────────────────────────────────────────────────────────
REGIME_HEX: list[str] = [
    "#FF1744",  # A1 Saturated · Range IV           (strongest warm, deep red)
    "#FF9100",  # A2 Distance-limited · Range III    (medium warm, orange)
    "#FFD740",  # A3 Distance-limited · Range II     (muted warm, amber)
    "#2979FF",  # A4 Cadence-limited · Range IV      (strongest cool, deep blue)
    "#00E5FF",  # A5 Cadence-limited · Range III     (medium cool, cyan)
    "#1DE9B6",  # A6 Cadence-limited · Range II      (muted cool, teal)
    "#9E9E9E",  # A7 Flux-limited · Range I          (neutral gray — D_dec limited)
]
REGIME_LABELS: list[str] = [
    "Saturated · Range IV",
    "Distance-limited · Range III",
    "Distance-limited · Range II",
    "Cadence-limited · Range IV",
    "Cadence-limited · Range III",
    "Cadence-limited · Range II",
    "Flux-limited · Range I",
]

# Human-readable t_cad tick labels for the y-axis
TCAD_TICKVALS_H = [1 / 3600, 1 / 60, 1, 6, 24, 168, 730, 8760]
TCAD_TICKTEXT = ["1 sec", "1 min", "1 hr", "6 hr", "1 day", "1 wk", "1 mo", "1 yr"]

# Marker styles
AMBER = "#fbbf24"   # optimal point
CORAL = "#f87171"   # ZTF reference point

# Standardized hover templates — order: N_exp, t_cad, t_exp, q_med, D_med, R_det
# 3D surface: N_exp=x, t_cad=y, R_det=z; customdata=[t_exp, q_med, D_med_Gpc]
XYZ_HOVER = (
    "N<sub>exp</sub> = %{x:.4g}<br>"
    "t<sub>cad</sub> = %{y:.4g} hr<br>"
    "t<sub>exp</sub> = %{customdata[0]:.3g} s<br>"
    "q<sub>med</sub> = %{customdata[1]:.3g}<br>"
    "D<sub>med</sub> = %{customdata[2]:.3g} Gpc<br>"
    "R<sub>det</sub> = %{z:.4g} yr ⁻¹"
    "<extra></extra>"
)
# N-slice: N_exp=x, R_det=y; customdata=[t_cad_hr, t_exp, q_med, D_med_Gpc]
N_HOVER_FULL = (
    "N<sub>exp</sub> = %{x:.4g}<br>"
    "t<sub>cad</sub> = %{customdata[0]:.4g} hr<br>"
    "t<sub>exp</sub> = %{customdata[1]:.3g} s<br>"
    "q<sub>med</sub> = %{customdata[2]:.3g}<br>"
    "D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>"
    "R<sub>det</sub> = %{y:.4g} yr ⁻¹"
    "<extra></extra>"
)
# T-slice: t_cad=x, R_det=y; customdata=[N_exp, t_exp, q_med, D_med_Gpc]
T_HOVER_FULL = (
    "N<sub>exp</sub> = %{customdata[0]:.4g}<br>"
    "t<sub>cad</sub> = %{x:.4g} hr<br>"
    "t<sub>exp</sub> = %{customdata[1]:.3g} s<br>"
    "q<sub>med</sub> = %{customdata[2]:.3g}<br>"
    "D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>"
    "R<sub>det</sub> = %{y:.4g} yr ⁻¹"
    "<extra></extra>"
)

# Line widths for day-cadence overlays
_DL_WIDTH_REGIME = 6
_DL_WIDTH_HEIGHT = 1.0
_DL_MARKER_HEIGHT = 2


# ── Utilities ────────────────────────────────────────────────────────────────

def _a2l(arr):
    """Convert numpy array → nested Python list for Plotly JSON serialization.

    Plotly 6 serializes numpy arrays as binary 'bdata' blobs.  The Plotly.js
    bundled with Dash 4.0 does not support this format, so 3-D surfaces (and
    other traces that receive raw numpy arrays) render as blank.  Converting to
    plain Python lists forces JSON-array serialization that Plotly.js can read.
    """
    return arr.tolist() if hasattr(arr, "tolist") else arr


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' → 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(0,0,0,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _masks_to_regime_id(masks: dict, n: int) -> np.ndarray:
    """Convert region_masks dict → 1-D integer regime-ID array (1–7, NaN for unassigned)."""
    rid = np.full(n, np.nan, dtype=float)
    for k, key in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
        mk = np.asarray(masks[key]).ravel()
        rid[mk] = float(k)
    return rid


def _figure_layout(theme: str, **extra) -> dict:
    """Base layout kwargs for 2-D slice figures."""
    dark = (theme or "dark") != "light"
    bg = "rgba(0,0,0,0)"
    grid_col = "rgba(255,255,255,0.10)" if dark else "rgba(0,0,0,0.10)"
    font_col = "#e2e8f0" if dark else "#1e293b"
    return dict(
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="'JetBrains Mono', 'Cascadia Code', monospace", color=font_col, size=12),
        margin=dict(l=64, r=24, b=48, t=40),
        **extra,
    )


def _empty_figure(message: str, theme: str) -> go.Figure:
    """Return a blank figure with a centered annotation (used when optimisation fails)."""
    dark = (theme or "dark") != "light"
    txt_col = "#8ba0c0" if dark else "#4b6080"
    fig = go.Figure()
    fig.update_layout(
        **_figure_layout(theme),
        annotations=[dict(text=message, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color=txt_col))],
    )
    return fig


# ── 2-D segment drawing ──────────────────────────────────────────────────────

def _hover_label(dark: bool) -> dict:
    """Consistent gray hover label for 2-D traces."""
    if dark:
        return dict(bgcolor="#1e2533", font_color="#e2e8f0", bordercolor="rgba(0,0,0,0)")
    return dict(bgcolor="#e8f0f8", font_color="#1e293b", bordercolor="rgba(0,0,0,0)")


def _draw_regime_segments_2d(
    fig: go.Figure,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    regime_ids: np.ndarray,
    *,
    customdata: np.ndarray | None = None,
    line_width: float = 2.5,
    opacity: float = 1.0,
    hovertemplate: str | None = None,
    hoverlabel: dict | None = None,
) -> None:
    """Draw a 2-D scatter line coloured by consecutive regime segments."""
    n = len(x_vals)
    i = 0
    while i < n:
        xi, yi, ri = x_vals[i], y_vals[i], regime_ids[i]
        if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(ri)):
            i += 1
            continue

        cur = int(ri)
        j = i
        while (j < n
               and np.isfinite(x_vals[j]) and np.isfinite(y_vals[j])
               and np.isfinite(regime_ids[j]) and int(regime_ids[j]) == cur):
            j += 1

        seg_x = list(x_vals[i:j])
        seg_y = list(y_vals[i:j])
        bridged = j < n and np.isfinite(x_vals[j]) and np.isfinite(y_vals[j])
        # Bridge to next segment for continuity
        if bridged:
            seg_x.append(x_vals[j])
            seg_y.append(y_vals[j])

        col = _hex_to_rgba(REGIME_HEX[cur - 1], opacity) if 1 <= cur <= 7 else f"rgba(128,128,128,{opacity})"
        trace_kwargs: dict = dict(
            x=seg_x, y=seg_y,
            mode="lines",
            line=dict(color=col, width=line_width),
            showlegend=False,
        )
        if customdata is not None:
            seg_cd = customdata[i:j]
            if bridged:
                seg_cd = np.vstack([seg_cd, customdata[j : j + 1]])
            trace_kwargs["customdata"] = seg_cd
        if hovertemplate is not None:
            trace_kwargs["hovertemplate"] = hovertemplate
            if hoverlabel is not None:
                trace_kwargs["hoverlabel"] = hoverlabel
        else:
            trace_kwargs["hoverinfo"] = "skip"
        fig.add_trace(go.Scatter(**trace_kwargs))
        i = j


def _draw_single_line_2d(
    fig: go.Figure,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    customdata: np.ndarray | None = None,
    color: str = "#6d9eff",
    line_width: float = 2.5,
    name: str = "",
    show_legend: bool = False,
    hovertemplate: str | None = None,
    hoverlabel: dict | None = None,
) -> None:
    trace_kwargs: dict = dict(
        x=_a2l(x_vals), y=_a2l(y_vals),
        mode="lines",
        line=dict(color=color, width=line_width),
        name=name,
        showlegend=show_legend,
    )
    if customdata is not None:
        trace_kwargs["customdata"] = customdata
    if hovertemplate is not None:
        trace_kwargs["hovertemplate"] = hovertemplate
        if hoverlabel is not None:
            trace_kwargs["hoverlabel"] = hoverlabel
    else:
        trace_kwargs["hoverinfo"] = "skip"
    fig.add_trace(go.Scatter(**trace_kwargs))


# ── Regime legend helper (2-D figures) ──────────────────────────────────────

def _add_regime_legend_2d(fig: go.Figure, color_on: bool) -> None:
    """Add phantom legend entries for all 7 regimes (same as 3D, used in 2D slices)."""
    if not color_on:
        return
    for col, label in zip(REGIME_HEX, REGIME_LABELS):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=8, color=col, symbol="square"),
            name=label,
            showlegend=True,
            hoverinfo="skip",
        ))


# ── Discrete-day overlay on 3-D surface ─────────────────────────────────────

def _add_discrete_day_lines(
    fig: go.Figure,
    *,
    model_day,
    i_det: int,
    N_cols: np.ndarray,
    t_cad_max_s: float,
    zmin_plot_log10: float = ZMIN_DISPLAY_LOG10,
    color_mode: str = "height",
    height_cmin: float = -2.0,
    height_cmax: float = 1.0,
    height_colorscale: str = "Plasma",
    regime_colors: list[str] | None = None,
    full_integral: bool = False,
    off_axis: bool = False,
) -> None:
    """Draw discrete integer-day cadence curves on top of a 3-D surface."""
    N_cols = np.asarray(N_cols, dtype=float)
    if not np.any(np.isfinite(N_cols)):
        return

    max_days = int(np.floor(float(t_cad_max_s) / float(DAY_S)))
    if max_days < 1:
        return

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
    _GPC = 3.086e27

    def _regime_id_for_t(t_s: float) -> np.ndarray:
        t_arr = np.full_like(N_line, float(t_s), dtype=float)
        masks = model_day.region_masks(int(i_det), N_line, t_arr, include_unphysical=False)
        rid = np.full(N_cols.shape, np.nan, dtype=float)
        for k, key in enumerate(["A1", "A2", "A3", "A4", "A5", "A6", "A7"], start=1):
            mk = np.asarray(masks[key]).reshape(1, -1).ravel()
            rid[mk] = float(k)
        return rid

    def _customdata_for_t(t_s: float) -> np.ndarray:
        """Return customdata array of shape (nN, 3) = [t_exp_s, q_med, D_med_Gpc]."""
        t_arr = np.full_like(N_cols, float(t_s))
        t_exp_arr = model_day.t_exp_s(N_cols, t_arr)
        q_med_arr, D_med_cm_arr = model_day.compute_medians(
            int(i_det), N_cols, t_arr, full_integral=full_integral, off_axis=off_axis,
        )
        D_med_Gpc_arr = D_med_cm_arr / _GPC
        return np.stack([t_exp_arr, q_med_arr, D_med_Gpc_arr], axis=-1)  # (nN, 3)

    _pending_marker_traces: list[go.Scatter3d] = []

    for n in n_vals:
        t_s = float(n) * float(DAY_S)
        log10R = _rate(
            model_day, int(i_det), N_line, np.full_like(N_line, t_s),
            full_integral, off_axis=off_axis,
        )
        log10R = np.asarray(log10R).reshape(1, -1).ravel()

        good = np.isfinite(N_cols) & np.isfinite(log10R) & (log10R >= float(zmin_plot_log10))
        if np.count_nonzero(good) < 2:
            continue

        x = _a2l(N_cols[good])
        y = _a2l(np.full(len(x), t_s / 3600.0, dtype=float))
        z = _a2l((10 ** log10R[good]).astype(float))
        cd_good = _customdata_for_t(t_s)[good]  # (n_good, 3)

        if color_mode == "regime" and regime_colors is not None and len(regime_colors) == 7:
            rid_all = _regime_id_for_t(t_s)
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
                    fig.add_trace(go.Scatter3d(
                        x=x[start:end], y=y[start:end], z=z[start:end],
                        customdata=_a2l(cd_good[start:end]),
                        mode="lines+markers",
                        line=dict(color=col, width=_DL_WIDTH_REGIME),
                        marker=dict(size=0, opacity=0.001),
                        showlegend=False,
                        hovertemplate=XYZ_HOVER,
                    ))
                start = end
        else:
            # Add line trace now; defer marker trace until after all lines so
            # dots always render on top regardless of camera angle (WebGL depth-sort
            # tie-breaks in favour of later draw calls).
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                customdata=_a2l(cd_good),
                mode="lines",
                line=dict(color="rgba(255,255,255,0.12)", width=_DL_WIDTH_HEIGHT),
                showlegend=False,
                hovertemplate=XYZ_HOVER,
            ))
            _pending_marker_traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                customdata=_a2l(cd_good),
                mode="markers",
                marker=dict(
                    size=_DL_MARKER_HEIGHT,
                    color=_a2l(log10R[good]),
                    colorscale=height_colorscale,
                    cmin=float(height_cmin),
                    cmax=float(height_cmax),
                    opacity=1.0,
                    showscale=False,
                ),
                showlegend=False,
                hovertemplate=XYZ_HOVER,
            ))

    for trace in _pending_marker_traces:
        fig.add_trace(trace)



# ── 3-D surface figure ───────────────────────────────────────────────────────

def build_3d_figure(
    *,
    surface_Xs: np.ndarray,
    surface_Ys_h: np.ndarray,
    surface_Rs: np.ndarray,
    surface_Z_plot: np.ndarray,
    surface_regime_id,
    surface_t_exp: np.ndarray,
    surface_q_med: np.ndarray,
    surface_D_med_Gpc: np.ndarray,
    color_on: bool,
    optical_on: bool,
    full_integral_on: bool = False,
    off_axis_on: bool = False,
    model_day=None,
    model_night=None,
    i_det: int = 1,
    day_line_N_cols: np.ndarray = None,
    day_line_t_max_s: float = 0.0,
    N_opt: float = np.nan,
    t_cad_opt_hr: float = np.nan,
    R_opt: float = np.nan,
    t_exp_opt_s: float = np.nan,
    q_med_opt: float = np.nan,
    D_med_Gpc_opt: float = np.nan,
    N_ztf: float = np.nan,
    t_cad_ztf_hr: float = np.nan,
    R_ztf: float = np.nan,
    t_exp_ztf_s: float = np.nan,
    q_med_ztf: float = np.nan,
    D_med_Gpc_ztf: float = np.nan,
    Rmax: float = 1.0,
    theme: str = "dark",
) -> go.Figure:
    dark = (theme or "dark") != "light"
    plotly_template = "plotly_dark" if dark else "plotly_white"

    fig = go.Figure()

    # Discrete day lines (optical mode only)
    if optical_on:
        if color_on and surface_regime_id is not None:
            _add_discrete_day_lines(
                fig, model_day=model_day, i_det=i_det,
                N_cols=day_line_N_cols, t_cad_max_s=day_line_t_max_s,
                zmin_plot_log10=ZMIN_DISPLAY_LOG10, color_mode="regime", regime_colors=REGIME_HEX,
                full_integral=full_integral_on, off_axis=off_axis_on,
            )
        else:
            zmax_color = float(np.nanmax(surface_Z_plot)) if np.any(np.isfinite(surface_Z_plot)) else 0.0
            _add_discrete_day_lines(
                fig, model_day=model_day, i_det=i_det,
                N_cols=day_line_N_cols, t_cad_max_s=day_line_t_max_s,
                zmin_plot_log10=ZMIN_DISPLAY_LOG10, color_mode="height",
                height_cmin=-2.0, height_cmax=zmax_color, height_colorscale="Plasma",
                full_integral=full_integral_on, off_axis=off_axis_on,
            )

    # Per-point customdata for hover: [t_exp_s, q_med, D_med_Gpc].
    # Plotly's go.Surface looks up customdata at transposed indices [j, i, k] for the
    # grid point at (row i, col j), so we must transpose (ny, nx, 3) → (nx, ny, 3).
    surf_cd = np.stack(
        [np.asarray(surface_t_exp), np.asarray(surface_q_med), np.asarray(surface_D_med_Gpc)],
        axis=-1,
    ).transpose(1, 0, 2)  # (ny, nx, 3) → (nx, ny, 3)

    # Main surface trace
    if color_on and surface_regime_id is not None:
        # One solid-color trace per regime — avoids Plotly's interpolation of
        # surfacecolor values across regime boundaries (which caused color bleed
        # and required explicit black boundary lines).
        rid = np.asarray(surface_regime_id)
        Rs  = np.asarray(surface_Rs)
        for k in range(1, 8):
            mask_k = np.isfinite(rid) & (rid == k) & np.isfinite(Rs)
            if not np.any(mask_k):
                continue
            Z_k = np.where(mask_k, Rs, np.nan)
            col = REGIME_HEX[k - 1]
            fig.add_trace(go.Surface(
                x=_a2l(surface_Xs), y=_a2l(surface_Ys_h), z=_a2l(Z_k),
                customdata=surf_cd,
                colorscale=[[0, col], [1, col]],
                cmin=0, cmax=1,
                showscale=False,
                connectgaps=False,
                hovertemplate=XYZ_HOVER,
                lighting=dict(ambient=0.75, diffuse=0.75, specular=0.08, roughness=0.95),
                lightposition=dict(x=100, y=200, z=0),
                showlegend=False,
            ))
        # Legend entries (regime labels only in color mode)
        for col, label in zip(REGIME_HEX, REGIME_LABELS):
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=7, color=col),
                name=label,
                showlegend=True,
                hoverinfo="skip",
            ))
    else:
        zmax_color = float(np.nanmax(surface_Z_plot)) if np.any(np.isfinite(surface_Z_plot)) else 0.0
        fig.add_trace(go.Surface(
            x=_a2l(surface_Xs), y=_a2l(surface_Ys_h), z=_a2l(surface_Rs),
            surfacecolor=_a2l(surface_Z_plot),
            customdata=surf_cd,
            cmin=ZMIN_DISPLAY_LOG10,
            cmax=zmax_color,
            showscale=True,
            colorscale="Plasma",
            connectgaps=False,
            hovertemplate=XYZ_HOVER,
            colorbar=dict(
                title=dict(text="R<sub>det</sub> [yr ⁻¹]", side="top"),
                x=0.88, xanchor="left",
                thickness=18, len=0.80,
                y=0.5, yanchor="middle",
                tickvals=[-2, -1, 0, 1, 2, 3, 4],
                ticktext=["0.01", "0.1", "1", "10", "100", "1k", "10k"],
            ),
        ))

    def _marker_hover(label: str, t_exp: float, q_med: float, D_med_Gpc: float) -> str:
        """Build standardized hover text for a special marker (optimum / ZTF)."""
        texp_str   = f"<br>t<sub>exp</sub> = {t_exp:.3g} s"   if np.isfinite(t_exp)    else ""
        qmed_str   = f"<br>q<sub>med</sub> = {q_med:.3g}"      if np.isfinite(q_med)    else ""
        dmed_str   = f"<br>D<sub>med</sub> = {D_med_Gpc:.3g} Gpc" if np.isfinite(D_med_Gpc) else ""
        return (
            f"{label}"
            f"<br>N<sub>exp</sub> = %{{x:.4g}}"
            f"<br>t<sub>cad</sub> = %{{y:.4g}} hr"
            f"{texp_str}{qmed_str}{dmed_str}"
            f"<br>R<sub>det</sub> = %{{z:.4g}} yr ⁻¹<extra></extra>"
        )

    # Markers: optimal (amber diamond) and ZTF (coral circle)
    if N_opt is not None and R_opt is not None and np.isfinite(N_opt) and np.isfinite(t_cad_opt_hr) and np.isfinite(R_opt):
        fig.add_trace(go.Scatter3d(
            x=[N_opt], y=[t_cad_opt_hr], z=[R_opt],
            mode="markers+text",
            marker=dict(size=10, color=AMBER, symbol="diamond"),
            text=["Optimum"],
            textposition="top center",
            name="Optimum",
            hovertemplate=_marker_hover("Grid optimum", t_exp_opt_s, q_med_opt, D_med_Gpc_opt),
        ))
    if R_ztf is not None and np.isfinite(R_ztf):
        fig.add_trace(go.Scatter3d(
            x=[N_ztf], y=[t_cad_ztf_hr], z=[R_ztf],
            mode="markers+text",
            marker=dict(size=9, color=CORAL, symbol="circle"),
            text=["ZTF"],
            textposition="top center",
            name="ZTF (2 day cadence)",
            hovertemplate=_marker_hover("ZTF strategy", t_exp_ztf_s, q_med_ztf, D_med_Gpc_ztf),
        ))

    # Layout
    bg_col = "rgba(0,0,0,0)"
    grid_col = "rgba(255,255,255,0.14)" if dark else "rgba(0,0,0,0.12)"
    hl = _hover_label(dark)
    fig.update_layout(
        uirevision="keep-view-v1",
        template=plotly_template,
        paper_bgcolor=bg_col,
        hoverlabel=dict(bgcolor=hl["bgcolor"], font_color=hl["font_color"],
                        bordercolor=hl["bordercolor"]),
        scene=dict(
            bgcolor=bg_col,
            xaxis=dict(
                title="N<sub>exp</sub>", type="log", gridcolor=grid_col,
                showbackground=False,
            ),
            yaxis=dict(
                title="t<sub>cad</sub>",
                type="log",
                tickvals=TCAD_TICKVALS_H,
                ticktext=TCAD_TICKTEXT,
                gridcolor=grid_col,
                showbackground=False,
            ),
            zaxis=dict(
                title="R<sub>det</sub> [yr ⁻¹]",
                type="log",
                range=[ZMIN_DISPLAY_LOG10, np.log10(max(Rmax, 0.11)) + 0.05],
                gridcolor=grid_col,
                showbackground=False,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.9),
        ),
        scene_camera=dict(
            eye=dict(x=-1.48, y=-1.48, z=0.70),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=-0.2),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            orientation="v",
            x=0.01, y=0.98,
            xanchor="left", yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ── N-slice figure (R vs N_exp at optimal t_cad) ─────────────────────────────

def build_nslice_figure(
    *,
    N_sweep: np.ndarray,
    R_n: np.ndarray,
    rid_n: np.ndarray,
    t_exp_n: np.ndarray,
    q_med_n: np.ndarray,
    D_med_Gpc_n: np.ndarray,
    N_opt: float,
    R_ztf: float,
    N_ztf: float,
    t_cad_fix_hr: float,
    t_cad_opt_hr: float,
    color_on: bool,
    theme: str,
) -> go.Figure:
    dark = (theme or "dark") != "light"
    accent = "#6d9eff" if dark else "#3b6fff"
    hl = _hover_label(dark)

    fig = go.Figure()

    # Regime legend entries (same as 3D, placed first so they appear at top of legend)
    _add_regime_legend_2d(fig, color_on)

    valid = np.isfinite(R_n) & (R_n > 0)
    if not np.any(valid):
        return _empty_figure("No valid data in N-slice", theme)

    # customdata: [t_cad_hr (fixed), t_exp, q_med, D_med_Gpc]
    t_cad_col = np.full(len(N_sweep), t_cad_fix_hr)
    cd_n = np.stack([t_cad_col, t_exp_n, q_med_n, D_med_Gpc_n], axis=-1)

    # Main rate curve
    if color_on and np.any(np.isfinite(rid_n)):
        _draw_regime_segments_2d(fig, N_sweep[valid], R_n[valid], rid_n[valid],
                                 customdata=cd_n[valid],
                                 line_width=2.5, hovertemplate=N_HOVER_FULL, hoverlabel=hl)
    else:
        _draw_single_line_2d(fig, N_sweep[valid], R_n[valid], color=accent, line_width=2.5,
                             customdata=cd_n[valid],
                             hovertemplate=N_HOVER_FULL, hoverlabel=hl)

    # ZTF reference line (horizontal dashed)
    if R_ztf is not None and np.isfinite(R_ztf):
        fig.add_hline(
            y=R_ztf,
            line=dict(color=CORAL, width=1.5, dash="dot"),
            annotation=dict(text=f"R<sub>ZTF</sub> = {R_ztf:.2g} yr ⁻¹", font_size=11,
                            font_color=CORAL, xanchor="right", x=0.98),
        )

    # Vertical line at ZTF N
    if N_ztf is not None and np.isfinite(N_ztf):
        fig.add_vline(
            x=N_ztf,
            line=dict(color=CORAL, width=1.2, dash="dot"),
        )
        fig.add_annotation(
            x=np.log10(N_ztf), y=0.04, xref="x", yref="paper",
            text=f"N<sub>ZTF</sub>={N_ztf:.0f}", showarrow=False,
            font=dict(size=10, color=CORAL), xanchor="center",
        )

    # Optimal N_exp marker — drawn at the N_opt position on the current curve
    if N_opt is not None and np.isfinite(N_opt):
        opt_idx = int(np.argmin(np.abs(N_sweep - N_opt)))
        R_at_N_opt = float(R_n[opt_idx]) if np.isfinite(R_n[opt_idx]) else np.nan
        opt_cd = cd_n[opt_idx : opt_idx + 1]
        if np.isfinite(R_at_N_opt) and R_at_N_opt > 0:
            fig.add_trace(go.Scatter(
                x=[N_opt], y=[R_at_N_opt],
                mode="markers",
                marker=dict(size=12, color=AMBER, symbol="diamond", line=dict(width=1.5, color="white")),
                name="Opt. N<sub>exp</sub>",
                customdata=opt_cd,
                hovertemplate=(
                    "N<sub>exp,opt</sub> = %{x:.4g}<br>"
                    "t<sub>cad</sub> = %{customdata[0]:.4g} hr<br>"
                    "t<sub>exp</sub> = %{customdata[1]:.3g} s<br>"
                    "q<sub>med</sub> = %{customdata[2]:.3g}<br>"
                    "D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>"
                    "R<sub>det</sub> = %{y:.4g} yr ⁻¹"
                    "<extra>Opt. N<sub>exp</sub></extra>"
                ),
                hoverlabel=hl,
            ))

    # Optimal t_cad indicator — dashed vertical line when slice is not at the optimum
    if np.isfinite(t_cad_opt_hr) and abs(t_cad_fix_hr - t_cad_opt_hr) / (t_cad_opt_hr + 1e-30) > 0.02:
        fig.add_annotation(
            x=0.99, y=0.98, xref="paper", yref="paper",
            text=f"Opt. t<sub>cad</sub> = {t_cad_opt_hr:.3g} hr",
            showarrow=False, xanchor="right", yanchor="top",
            font=dict(size=10, color=AMBER),
        )

    # Title annotation (compact, top-left)
    title_txt = (
        f"N<sub>exp</sub> slice  |  t<sub>cad</sub> = {t_cad_fix_hr:.3g} hr"
        if np.isfinite(t_cad_fix_hr) else "N<sub>exp</sub> slice"
    )

    fig.update_layout(
        **_figure_layout(theme,
            legend=dict(x=0.01, y=0.98, xanchor="left", yanchor="top",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11))),
        xaxis=dict(
            title="N<sub>exp</sub>",
            type="log",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title="R<sub>det</sub> [yr ⁻¹]",
            type="log",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)",
        ),
        annotations=[dict(
            text=title_txt,
            xref="paper", yref="paper", x=0.01, y=1.0,
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=12, color="#8ba0c0" if dark else "#4b6080"),
        )],
        showlegend=True,
    )
    return fig


# ── t-slice figure (R vs t_cad at optimal N_exp) ─────────────────────────────

def build_tslice_figure(
    *,
    t_cont_h: np.ndarray,
    R_cont: np.ndarray,
    rid_cont: np.ndarray,
    t_exp_cont: np.ndarray,
    q_med_cont: np.ndarray,
    D_med_Gpc_cont: np.ndarray,
    t_disc_h: np.ndarray,
    R_disc: np.ndarray,
    rid_disc: np.ndarray,
    t_exp_disc: np.ndarray,
    q_med_disc: np.ndarray,
    D_med_Gpc_disc: np.ndarray,
    gap_lo_h: float | None,
    gap_hi_h: float | None,
    t_opt_h: float,
    t_ztf_h: float,
    R_ztf: float,
    R_opt: float,
    N_fix: float,
    N_opt: float,
    color_on: bool,
    theme: str,
    optical_on: bool,
) -> go.Figure:
    dark = (theme or "dark") != "light"
    accent = "#6d9eff" if dark else "#3b6fff"
    gap_col = "rgba(255,200,100,0.07)" if dark else "rgba(200,140,0,0.07)"
    hl = _hover_label(dark)

    fig = go.Figure()

    # Regime legend entries (placed first so they appear at top of legend)
    _add_regime_legend_2d(fig, color_on)

    has_cont = len(t_cont_h) > 0 and np.any(np.isfinite(R_cont) & (R_cont > 0))
    has_disc = len(t_disc_h) > 0 and np.any(np.isfinite(R_disc) & (R_disc > 0))

    if not has_cont and not has_disc:
        return _empty_figure("No valid data in t-slice", theme)

    grid_col = "rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)"

    # customdata: [N_exp (fixed), t_exp, q_med, D_med_Gpc]
    N_opt_col_c = np.full(len(t_cont_h), N_opt)
    cd_cont = np.stack([N_opt_col_c, t_exp_cont, q_med_cont, D_med_Gpc_cont], axis=-1)

    N_opt_col_d = np.full(len(t_disc_h), N_opt)
    cd_disc = np.stack([N_opt_col_d, t_exp_disc, q_med_disc, D_med_Gpc_disc], axis=-1)

    # Continuous region
    if has_cont:
        valid_c = np.isfinite(R_cont) & (R_cont > 0)
        if color_on and np.any(np.isfinite(rid_cont)):
            _draw_regime_segments_2d(fig, t_cont_h[valid_c], R_cont[valid_c], rid_cont[valid_c],
                                     customdata=cd_cont[valid_c],
                                     line_width=2.5, hovertemplate=T_HOVER_FULL, hoverlabel=hl)
        else:
            _draw_single_line_2d(fig, t_cont_h[valid_c], R_cont[valid_c], color=accent, line_width=2.5,
                                 customdata=cd_cont[valid_c],
                                 hovertemplate=T_HOVER_FULL, hoverlabel=hl)

    # Gap region shading
    if optical_on and gap_lo_h is not None and gap_hi_h is not None and gap_lo_h < gap_hi_h:
        fig.add_vrect(
            x0=gap_lo_h, x1=gap_hi_h,
            fillcolor=gap_col, layer="below", line_width=0,
            annotation=dict(text="gap", font=dict(size=10, color=AMBER),
                            xanchor="center", yanchor="top", y=0.99, yref="paper"),
        )

    # Discrete region (markers + connecting lines)
    if has_disc:
        valid_d = np.isfinite(R_disc) & (R_disc > 0)
        if color_on and np.any(np.isfinite(rid_disc)):
            _draw_regime_segments_2d(fig, t_disc_h[valid_d], R_disc[valid_d], rid_disc[valid_d],
                                     customdata=cd_disc[valid_d],
                                     line_width=1.5, opacity=0.9,
                                     hovertemplate=T_HOVER_FULL, hoverlabel=hl)
            # Overlay markers
            for idx in np.where(valid_d)[0]:
                rid_val = rid_disc[idx]
                col = REGIME_HEX[int(rid_val) - 1] if (np.isfinite(rid_val) and 1 <= int(rid_val) <= 7) else "#888"
                fig.add_trace(go.Scatter(
                    x=[t_disc_h[idx]], y=[R_disc[idx]],
                    mode="markers",
                    marker=dict(size=6, color=col, line=dict(width=1, color="white")),
                    showlegend=False,
                    customdata=cd_disc[idx : idx + 1],
                    hovertemplate=T_HOVER_FULL,
                    hoverlabel=hl,
                ))
        else:
            fig.add_trace(go.Scatter(
                x=_a2l(t_disc_h[valid_d]), y=_a2l(R_disc[valid_d]),
                mode="markers+lines",
                marker=dict(size=5, color=accent, line=dict(width=1, color="white")),
                line=dict(color=_hex_to_rgba("#6d9eff", 0.4), width=1.5),
                showlegend=False,
                customdata=cd_disc[valid_d],
                hovertemplate=T_HOVER_FULL,
                hoverlabel=hl,
            ))

    # ZTF reference line (horizontal dashed)
    if R_ztf is not None and np.isfinite(R_ztf):
        fig.add_hline(
            y=R_ztf,
            line=dict(color=CORAL, width=1.5, dash="dot"),
            annotation=dict(text=f"R<sub>ZTF</sub> = {R_ztf:.2g} yr ⁻¹", font_size=11,
                            font_color=CORAL, xanchor="right", x=0.98),
        )

    # ZTF cadence vertical marker
    if t_ztf_h is not None and np.isfinite(t_ztf_h):
        fig.add_vline(
            x=t_ztf_h,
            line=dict(color=CORAL, width=1.2, dash="dot"),
        )

    # Optimal t_cad vertical marker
    if t_opt_h is not None and np.isfinite(t_opt_h):
        fig.add_vline(
            x=t_opt_h,
            line=dict(color=AMBER, width=1.8, dash="dash"),
        )
        # Optimal point on slice (find R and customdata at t_opt_h)
        all_t = np.concatenate([t_cont_h, t_disc_h]) if has_disc else t_cont_h
        all_R = np.concatenate([R_cont, R_disc]) if has_disc else R_cont
        all_cd = np.concatenate([cd_cont, cd_disc], axis=0) if has_disc else cd_cont
        if len(all_t) > 0:
            idx_closest = np.nanargmin(np.abs(all_t - t_opt_h))
            R_at_opt = all_R[idx_closest]
            if np.isfinite(R_at_opt) and R_at_opt > 0:
                fig.add_trace(go.Scatter(
                    x=[t_opt_h], y=[R_at_opt],
                    mode="markers",
                    marker=dict(size=12, color=AMBER, symbol="diamond",
                                line=dict(width=1.5, color="white")),
                    name="Optimum",
                    customdata=all_cd[idx_closest : idx_closest + 1],
                    hovertemplate=(
                        "N<sub>exp</sub> = %{customdata[0]:.4g}<br>"
                        "t<sub>cad</sub> = %{x:.4g} hr<br>"
                        "t<sub>exp</sub> = %{customdata[1]:.3g} s<br>"
                        "q<sub>med</sub> = %{customdata[2]:.3g}<br>"
                        "D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>"
                        "R<sub>det</sub> = %{y:.4g} yr ⁻¹"
                        "<extra>Optimum</extra>"
                    ),
                    hoverlabel=hl,
                ))

    title_txt = (
        f"t<sub>cad</sub> slice  |  N<sub>exp</sub> = {N_fix:.0f} fields"
        if N_fix is not None and np.isfinite(N_fix) else "t<sub>cad</sub> slice"
    )

    fig.update_layout(
        **_figure_layout(theme,
            legend=dict(x=0.01, y=0.98, xanchor="left", yanchor="top",
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11))),
        xaxis=dict(
            title="t<sub>cad</sub>",
            type="log",
            tickvals=TCAD_TICKVALS_H,
            ticktext=TCAD_TICKTEXT,
            showgrid=True,
            gridcolor=grid_col,
        ),
        yaxis=dict(
            title="R<sub>det</sub> [yr ⁻¹]",
            type="log",
            showgrid=True,
            gridcolor=grid_col,
        ),
        annotations=[dict(
            text=title_txt,
            xref="paper", yref="paper", x=0.01, y=1.0,
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=12, color="#8ba0c0" if dark else "#4b6080"),
        )],
        showlegend=True,
    )
    return fig


# ── Metrics strip HTML ────────────────────────────────────────────────────────

def build_metrics_bar(
    *,
    R_opt: float,
    R_ztf: float,
    t_cad_opt_s: float,
    N_opt: float,
    t_exp_opt_s: float,
    t_cad_ztf_s: float,
    N_ztf: float,
    t_exp_ztf_s: float,
) -> list:
    """Return a list of html children for the metrics strip."""

    def _fmt_r(r: float) -> str:
        if not np.isfinite(r):
            return "—"
        if r >= 100:
            return f"{r:.0f}"
        if r >= 10:
            return f"{r:.1f}"
        return f"{r:.2f}"

    def _fmt_t(t_s: float) -> str:
        if not np.isfinite(t_s):
            return "—"
        if t_s >= DAY_S:
            return f"{t_s / DAY_S:.2g} day"
        if t_s >= 3600:
            return f"{t_s / 3600:.2g} hr"
        if t_s >= 60:
            return f"{t_s / 60:.2g} min"
        return f"{t_s:.2g} sec"

    gain_str = "—"
    gain_cls = "gain"
    if R_opt is not None and R_ztf is not None and np.isfinite(R_opt) and np.isfinite(R_ztf) and R_ztf > 0:
        gain = R_opt / R_ztf
        gain_str = f"×{gain:.2f}"
        gain_cls = "gain positive" if gain >= 1 else "gain negative"

    def _lbl(children):
        return html.Span(children, className="metric-label")

    # Groups: [optimum group], [ZTF group], [gain]
    # Prominent separators are inserted between groups.
    opt_badges = [
        (_lbl(["R", html.Sub("det,opt")]),  f"{_fmt_r(R_opt)} /yr",                         "amber"),
        (_lbl(["t", html.Sub("cad,opt")]),  _fmt_t(t_cad_opt_s),                             "muted"),
        (_lbl(["N", html.Sub("exp,opt")]),  f"{N_opt:.0f}" if N_opt is not None and np.isfinite(N_opt) else "—",   "muted"),
        (_lbl(["t", html.Sub("exp,opt")]),  _fmt_t(t_exp_opt_s),                             "muted"),
    ]
    ztf_badges = [
        (_lbl(["R", html.Sub("det,ZTF")]),  f"{_fmt_r(R_ztf)} /yr",                           "coral"),
        (_lbl(["t", html.Sub("cad,ZTF")]),  _fmt_t(t_cad_ztf_s),                               "muted"),
        (_lbl(["N", html.Sub("exp,ZTF")]),  f"{N_ztf:.0f}" if N_ztf is not None and np.isfinite(N_ztf) else "—",     "muted"),
        (_lbl(["t", html.Sub("exp,ZTF")]),  _fmt_t(t_exp_ztf_s),                               "muted"),
    ]
    gain_badge = [
        (_lbl("Gain"),  gain_str,  gain_cls),
    ]

    def _render_group(badges):
        out = []
        for i, (label_elem, value, style_cls) in enumerate(badges):
            out.append(html.Div([
                label_elem,
                html.Span(value, className=f"metric-value metric-{style_cls}"),
            ], className="metric-badge"))
            if i < len(badges) - 1:
                out.append(html.Div(className="metric-sep"))
        return out

    children = _render_group(opt_badges)
    children.append(html.Div(className="metric-sep"))
    children.append(html.Div(className="metric-sep-prominent"))
    children += _render_group(ztf_badges)
    children.append(html.Div(className="metric-sep"))
    children.append(html.Div(className="metric-sep-prominent"))
    children += _render_group(gain_badge)

    return children
