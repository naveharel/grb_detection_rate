# components/layout.py
"""Top-level Dash app layout using DBC and custom CSS flexbox."""
from __future__ import annotations

import math

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.sidebar import create_sidebar

# Default initial values for slice position sliders (log₁₀ units)
_NSLICE_TFIX_DEFAULT = round(math.log10(86400), 3)   # 1 day
_TSLICE_NFIX_DEFAULT = 2.0                             # 100 fields


def create_layout() -> html.Div:
    """Return the complete app layout tree."""
    return html.Div(
        id="app-root",
        children=[
            # ── Persistent stores ────────────────────────────────────────────
            dcc.Store(id="theme-store", data="dark"),
            dcc.Store(id="view-store", data="3d"),
            dcc.Store(id="preset-active-store", data=None),  # tracks active preset's parameter values
            dcc.Store(id="surface-store", data=None),        # last computed surface (for CSV export)
            dcc.Store(id="_dom-theme-dummy"),
            dcc.Download(id="download-data"),

            # ── Navbar ───────────────────────────────────────────────────────
            html.Nav(
                className="app-navbar",
                children=[
                    html.Button("☰", id="sidebar-toggle", className="icon-btn", n_clicks=0,
                                title="Toggle parameters sidebar"),
                    html.Span("GRB Detection Rate Explorer", className="app-title"),
                    html.Div(
                        className="navbar-right",
                        children=[
                            # Preset selector
                            dbc.Select(
                                id="preset-select",
                                options=[
                                    {"label": "— Presets —", "value": "none", "disabled": True},
                                    {"label": "ZTF", "value": "ztf"},
                                    {"label": "Rubin LSST", "value": "rubin"},
                                ],
                                value="none",
                                className="preset-select",
                                size="sm",
                            ),
                            # Export button (dcc.Loading shows spinner while computing)
                            dcc.Loading(
                                id="export-loading",
                                type="circle",
                                color="var(--accent)",
                                children=html.Button(
                                    "⬇ CSV", id="export-btn", className="nav-btn",
                                    n_clicks=0, title="Download surface data as CSV",
                                ),
                                style={"display": "inline-flex", "alignItems": "center"},
                                overlay_style={"visibility": "visible"},
                            ),
                            # View-mode segmented control
                            html.Div(
                                className="view-controls",
                                children=[
                                    html.Button("3D", id="btn-view-3d",
                                                className="view-btn active", n_clicks=0,
                                                title="3D surface view"),
                                    html.Button(
                                        ["N", html.Sub("exp"), " slice"],
                                        id="btn-view-nslice",
                                        className="view-btn", n_clicks=0,
                                        title="R vs N_exp at fixed t_cad",
                                    ),
                                    html.Button(
                                        ["t", html.Sub("cad"), " slice"],
                                        id="btn-view-tslice",
                                        className="view-btn", n_clicks=0,
                                        title="R vs t_cad at fixed N_exp",
                                    ),
                                ],
                            ),
                            # Theme toggle
                            html.Button("☾ Dark", id="theme-toggle", className="icon-btn",
                                        n_clicks=0, title="Toggle dark / light mode"),
                        ],
                    ),
                ],
            ),

            # ── Metrics strip ────────────────────────────────────────────────
            html.Div(id="metrics-bar", className="metrics-strip"),

            # ── Body: sidebar + main area ────────────────────────────────────
            html.Div(
                className="app-body",
                children=[
                    create_sidebar(),
                    # Main visualization area
                    html.Main(
                        className="app-main",
                        children=[
                            # 3D view (default visible)
                            html.Div(
                                id="panel-3d",
                                className="view-panel",
                                style={"display": "flex", "flex": "1", "minHeight": "0"},
                                children=[
                                    dcc.Graph(
                                        id="graph-3d",
                                        style={"flex": "1", "minHeight": "0"},
                                        config={"displayModeBar": True, "displaylogo": False,
                                                "modeBarButtonsToRemove": ["resetCameraDefault3d"]},
                                    ),
                                ],
                            ),
                            # N-slice view (hidden)
                            html.Div(
                                id="panel-nslice",
                                className="view-panel",
                                style={"display": "none", "flexDirection": "column"},
                                children=[
                                    dcc.Graph(
                                        id="graph-nslice",
                                        style={"flex": "1", "minHeight": "0"},
                                        config={"displayModeBar": True, "displaylogo": False},
                                    ),
                                    html.Div(className="slice-ctrl", children=[
                                        html.Span(
                                            ["Fixed t", html.Sub("cad")],
                                            className="slice-ctrl-label",
                                        ),
                                        dcc.Slider(
                                            id="nslice-tfix-slider",
                                            min=2, max=8, step=0.05,
                                            value=_NSLICE_TFIX_DEFAULT,
                                            marks={
                                                2: "100 s",
                                                3.56: "1 hr",
                                                4.94: "1 day",
                                                5.78: "1 wk",
                                                7.5: "1 yr",
                                            },
                                            className="param-slider slice-pos-slider",
                                        ),
                                    ]),
                                ],
                            ),
                            # t-slice view (hidden)
                            html.Div(
                                id="panel-tslice",
                                className="view-panel",
                                style={"display": "none", "flexDirection": "column"},
                                children=[
                                    dcc.Graph(
                                        id="graph-tslice",
                                        style={"flex": "1", "minHeight": "0"},
                                        config={"displayModeBar": True, "displaylogo": False},
                                    ),
                                    html.Div(className="slice-ctrl", children=[
                                        html.Span(
                                            ["Fixed N", html.Sub("exp")],
                                            className="slice-ctrl-label",
                                        ),
                                        dcc.Slider(
                                            id="tslice-nfix-slider",
                                            min=0, max=4, step=0.05,
                                            value=_TSLICE_NFIX_DEFAULT,
                                            marks={
                                                0: "1",
                                                1: "10",
                                                2: "100",
                                                3: "1k",
                                                4: "10k",
                                            },
                                            className="param-slider slice-pos-slider",
                                        ),
                                    ]),
                                ],
                            ),
                            # Status / debug bar
                            html.Div(id="status", className="status-bar"),
                        ],
                    ),
                ],
            ),
        ],
    )
