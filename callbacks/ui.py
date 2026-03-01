# callbacks/ui.py
"""UI-state callbacks: theme, sidebar, view-mode switching, presets, t_night visibility."""
from __future__ import annotations

import dash
from dash import Input, Output, State

# Preset configurations (values for all sliders)
_PRESETS: dict[str, dict] = {
    "ztf": {
        "i": 10, "f_live": 0.2, "A_log": -4.68,
        "omega_exp": 47, "t_oh": 15, "omega_srv": 27500,
        "optical": True,
    },
    "rubin": {
        "i": 10, "f_live": 0.7, "A_log": -7.0,
        "omega_exp": 9.6, "t_oh": 30, "omega_srv": 18000,
        "optical": True,
    },
}


def register(app: dash.Dash) -> None:

    # ── Sidebar toggle ───────────────────────────────────────────────────────
    @app.callback(
        Output("app-sidebar", "className"),
        Output("sidebar-toggle", "children"),
        Input("sidebar-toggle", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(n_clicks):
        if n_clicks and n_clicks % 2 == 1:
            return "app-sidebar", "☰"
        return "app-sidebar open", "☰"

    # ── t_night visibility ───────────────────────────────────────────────────
    @app.callback(
        Output("tnight_container", "style"),
        Input("optical-switch", "value"),
    )
    def toggle_tnight(is_optical):
        return {"display": "block"} if is_optical else {"display": "none"}

    # ── Theme: store → button label ──────────────────────────────────────────
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

    # ── Theme: apply to <html> via data-theme attribute ──────────────────────
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

    # ── View-mode store (3D / nslice / tslice) ───────────────────────────────
    app.clientside_callback(
        """
        function(n1, n2, n3) {
            const ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || ctx.triggered.length === 0)
                return "3d";
            const pid = ctx.triggered[0].prop_id;
            if (pid.startsWith("btn-view-nslice")) return "nslice";
            if (pid.startsWith("btn-view-tslice")) return "tslice";
            return "3d";
        }
        """,
        Output("view-store", "data"),
        Input("btn-view-3d", "n_clicks"),
        Input("btn-view-nslice", "n_clicks"),
        Input("btn-view-tslice", "n_clicks"),
    )

    # ── Show/hide view panels based on view-store ────────────────────────────
    app.clientside_callback(
        """
        function(mode) {
            var show = {display: "flex", flex: "1", minHeight: "0"};
            var hide = {display: "none"};
            return [
                mode === "3d"      ? show : hide,
                mode === "nslice"  ? show : hide,
                mode === "tslice"  ? show : hide,
            ];
        }
        """,
        Output("panel-3d", "style"),
        Output("panel-nslice", "style"),
        Output("panel-tslice", "style"),
        Input("view-store", "data"),
    )

    # ── View button active-class highlighting ────────────────────────────────
    app.clientside_callback(
        """
        function(mode) {
            return [
                mode === "3d"      ? "view-btn active" : "view-btn",
                mode === "nslice"  ? "view-btn active" : "view-btn",
                mode === "tslice"  ? "view-btn active" : "view-btn",
            ];
        }
        """,
        Output("btn-view-3d", "className"),
        Output("btn-view-nslice", "className"),
        Output("btn-view-tslice", "className"),
        Input("view-store", "data"),
    )

    # ── Presets: fill all sliders at once ────────────────────────────────────
    @app.callback(
        Output("i_slider", "value", allow_duplicate=True),
        Output("flive_slider", "value", allow_duplicate=True),
        Output("Alog_slider", "value", allow_duplicate=True),
        Output("omegaexp_slider", "value", allow_duplicate=True),
        Output("toh_slider", "value", allow_duplicate=True),
        Output("omega_srv_slider", "value", allow_duplicate=True),
        Output("optical-switch", "value", allow_duplicate=True),
        Input("preset-select", "value"),
        prevent_initial_call=True,
    )
    def apply_preset(preset_key):
        if preset_key not in _PRESETS:
            return (dash.no_update,) * 7
        p = _PRESETS[preset_key]
        return p["i"], p["f_live"], p["A_log"], p["omega_exp"], p["t_oh"], p["omega_srv"], p.get("optical", dash.no_update)

