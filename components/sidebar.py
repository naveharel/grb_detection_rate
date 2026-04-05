# components/sidebar.py
"""Sidebar parameter controls: DBC Accordion with styled sliders and inputs."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from grb_detect.params import SurveyDesignParams

_survey_defaults = SurveyDesignParams()
T_NIGHT_DEFAULT_H: float = _survey_defaults.t_night_s / 3600.0
OMEGA_SRV_DEFAULT_DEG2: float = 27500.0


# ── Parameter block helper ───────────────────────────────────────────────────

def _param_block(
    label,          # str or list of Dash children
    tooltip: str,
    slider_id: str,
    input_id: str,
    slider_min: float,
    slider_max: float,
    slider_step: float,
    slider_val: float,
    marks: dict,
    *,
    input_min: float | None = None,
    input_max: float | None = None,
    input_step: float | None = None,
) -> html.Div:
    return html.Div(className="param-block", children=[
        html.Div(className="param-row", children=[
            html.Label(
                label,
                htmlFor=slider_id,
                className="param-label",
                title=tooltip,
            ),
            dcc.Input(
                id=input_id,
                type="number",
                value=slider_val,
                min=input_min if input_min is not None else slider_min,
                max=input_max if input_max is not None else slider_max,
                step=input_step if input_step is not None else slider_step,
                className="param-input",
                debounce=False,
            ),
        ]),
        dcc.Slider(
            slider_min, slider_max, slider_step,
            value=slider_val,
            id=slider_id,
            marks=marks,
            className="param-slider",
        ),
    ])


def _toggle_block(switch_id: str, label: str, hint_children: list, default: bool = False) -> html.Div:
    """A switch with a descriptive hint line below it."""
    return html.Div([
        html.Div(className="param-switch-row", children=[
            dbc.Switch(
                id=switch_id,
                label=label,
                value=default,
                className="param-switch",
            ),
        ]),
        html.P(hint_children, className="toggle-hint"),
    ])


# ── Accordion sections ───────────────────────────────────────────────────────

def _strategy_section() -> list:
    return [
        _param_block(
            "i", "Number of detections required per GRB to count as a detection event",
            "i_slider", "i_input",
            2, 100, 1, 10,
            {2: "2", 10: "10", 30: "30", 100: "100"},
        ),
        _param_block(
            ["f", html.Sub("live")],
            "Fraction of total time the telescope is actually observing (0=never, 1=always)",
            "flive_slider", "flive_input",
            0.01, 1.0, 0.01, 0.2,
            {0.01: "0.01", 0.2: "0.2", 0.5: "0.5", 1.0: "1"},
            input_step=0.01,
        ),
    ]


def _instrument_section() -> list:
    return [
        _param_block(
            "log A [Jy]", "log₁₀ of reference limiting flux in Jansky at t_exp_ref = 30 s",
            "Alog_slider", "Alog_input",
            -12, -2, 0.01, -4.68,
            {-12: "-12", -8: "-8", -4.68: "-4.68", -2: "-2"},
            input_step=0.01,
        ),
        _param_block(
            ["Ω", html.Sub("exp"), " [deg²]"],
            "Single-exposure field of view of the instrument in square degrees",
            "omegaexp_slider", "omegaexp_input",
            1, 200, 1, 47,
            {1: "1", 47: "47", 100: "100", 200: "200"},
        ),
        _param_block(
            ["t", html.Sub("OH"), " [sec]"],
            "Per-exposure overhead (readout, slew, settle) in seconds",
            "toh_slider", "toh_input",
            0.0, 30.0, 0.5, 0.0,
            {0: "0", 15: "15", 30: "30"},
            input_step=0.5,
        ),
    ]


def _constraints_section() -> list:
    return [
        _param_block(
            ["Ω", html.Sub("srv,max"), " [deg²]"],
            "Maximum surveyable sky area per cadence cycle in square degrees",
            "omega_srv_slider", "omega_srv_input",
            100, 41253, 100, OMEGA_SRV_DEFAULT_DEG2,
            {100: "100", 10000: "10k", 27500: "27.5k", 41253: "41k"},
            input_step=100,
        ),
        html.Div(
            id="tnight_container",
            style={"display": "none"},
            children=[
                _param_block(
                    ["t", html.Sub("night"), " [hr]"],
                    "Length of the observable astronomical night in hours",
                    "tnight_slider", "tnight_input",
                    4.0, 14.0, 0.25, T_NIGHT_DEFAULT_H,
                    {4: "4", 6: "6", 8: "8", 10: "10", 12: "12", 14: "14"},
                    input_step=0.25,
                ),
            ],
        ),
    ]


def _settings_section() -> list:
    return [
        _toggle_block(
            "optical-switch",
            "Optical survey mode",
            [
                "Requires ≥ i detections in one night or a single detection each night. "
                "Multi-day effective live fraction is reduced by f",
                html.Sub("night"), " = t", html.Sub("night"), " / 24 h; "
                "the t", html.Sub("night"), " slider becomes active.",
            ],
        ),
        _toggle_block(
            "toh-approx-switch",
            "t_OH approximation",
            [
                "Uses the naive t", html.Sub("exp"), " = f", html.Sub("live"),
                "·t", html.Sub("cad"), "/N", html.Sub("exp"),
                " equation (ignoring t", html.Sub("OH"),
                " in the exposure formula) for fully analytic optimal strategies, "
                "while still enforcing f", html.Sub("live"), "·t", html.Sub("cad"),
                "/N", html.Sub("exp"), " > t", html.Sub("OH"), " as the validity boundary.",
            ],
        ),
        _toggle_block(
            "color-switch",
            "Color by detection regime",
            [
                "Colors the surface by analytical detection regime. "
                "Warm colors (red→amber) indicate flux-limited (D",
                html.Sub("Euc"), ") regimes; cool colors (blue→teal) indicate "
                "cadence-limited (D", html.Sub("i"),
                ") regimes; gray indicates the singly-limited (D",
                html.Sub("dec"), ") regime. "
                "More vibrant colors correspond to higher detection ranges.",
            ],
        ),
        _toggle_block(
            "full-integral-switch",
            "Full integral mode",
            [
                "Computes R", html.Sub("det"),
                " using the exact integral over all viewing angles q ∈ [0, q",
                html.Sub("nr"), "] (thesis Eq. 39), instead of the dominant-term "
                "approximation. Adds the tail contribution beyond q",
                html.Sub("Euc"), " (flux-limited) or q", html.Sub("i"),
                " (cadence-limited). Slower but more accurate, especially near "
                "range boundaries.",
            ],
        ),
        _toggle_block(
            "off-axis-switch",
            "Off-axis detections only",
            [
                "Show only GRBs detected from outside the relativistic beaming cone "
                "(viewing angle q > q", html.Sub("dec"), "). "
                "Subtracts the on-axis contribution (q < q", html.Sub("dec"),
                ") from the rate integral. "
                "Regions where no off-axis detection is possible are masked out.",
            ],
        ),
    ]


# ── Public factory ───────────────────────────────────────────────────────────

def create_sidebar() -> html.Aside:
    """Return the full collapsible sidebar element."""
    accordion = dbc.Accordion(
        id="param-accordion",
        always_open=True,
        active_item=["strategy", "instrument", "constraints", "settings"],
        className="param-accordion",
        children=[
            dbc.AccordionItem(
                _strategy_section(),
                title="Strategy",
                item_id="strategy",
                className="accordion-section",
            ),
            dbc.AccordionItem(
                _instrument_section(),
                title="Instrument",
                item_id="instrument",
                className="accordion-section",
            ),
            dbc.AccordionItem(
                _constraints_section(),
                title="Constraints",
                item_id="constraints",
                className="accordion-section",
            ),
            dbc.AccordionItem(
                _settings_section(),
                title="Settings",
                item_id="settings",
                className="accordion-section",
            ),
        ],
    )

    return html.Aside(
        id="app-sidebar",
        className="app-sidebar open",
        children=[
            html.Div(
                className="sidebar-inner",
                children=[accordion],
            )
        ],
    )
