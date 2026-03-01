# callbacks/sync.py
"""Bidirectional slider ↔ number-input sync callbacks.

All slider→input syncs are clientside (zero server round-trips).
All input→slider syncs are server-side on n_blur with clamping.
"""
from __future__ import annotations

import numpy as np
import dash
from dash import Input, Output, State

# (slider_id, input_id, clamp_min, clamp_max)
_SYNC_PAIRS: list[tuple[str, str, float, float]] = [
    ("i_slider",         "i_input",         2,      300),
    ("flive_slider",     "flive_input",     0.01,   1.0),
    ("Alog_slider",      "Alog_input",      -12,    -2),
    ("omegaexp_slider",  "omegaexp_input",  1,      200),
    ("toh_slider",       "toh_input",       0.0,    30.0),
    ("omega_srv_slider", "omega_srv_input", 100,    41253),
    ("tnight_slider",    "tnight_input",    4.0,    14.0),
]


def register(app: dash.Dash) -> None:
    """Register all slider ↔ input sync callbacks on *app*."""

    # Slider → Input: instant clientside (no server round-trip)
    for sid, iid, *_ in _SYNC_PAIRS:
        app.clientside_callback(
            "function(v) { return (v == null) ? window.dash_clientside.no_update : v; }",
            Output(iid, "value"),
            Input(sid, "value"),
        )

    # Input → Slider: server-side on blur with clamping
    def _make_cb(slider_id: str, input_id: str, lo: float, hi: float):
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

    for sid, iid, lo, hi in _SYNC_PAIRS:
        _make_cb(sid, iid, lo, hi)
