# components/sidebar.py
"""Sidebar parameter controls: DBC Accordion with styled sliders and inputs."""
from __future__ import annotations

import math

import dash_bootstrap_components as dbc
from dash import dcc, html

from grb_detect.params import AfterglowPhysicalParams, CM_TO_GPC, MicrophysicsParams, SurveyDesignParams

_phys_defaults = AfterglowPhysicalParams()
_micro_defaults = MicrophysicsParams()

_survey_defaults = SurveyDesignParams()
T_NIGHT_DEFAULT_H: float = _survey_defaults.t_night_s / 3600.0
OMEGA_SRV_DEFAULT_DEG2: float = 27500.0


# ── Parameter block helper ───────────────────────────────────────────────────

def _bold_default_mark(marks: dict, slider_val: float) -> dict:
    """Return a copy of *marks* where the entry matching *slider_val* is bolded."""
    result = {}
    for k, v in marks.items():
        lbl = v if isinstance(v, str) else v.get("label", str(v))
        if abs(float(k) - float(slider_val)) < 1e-9:
            result[k] = {"label": lbl, "style": {"fontWeight": "700", "fontSize": "11px"}}
        else:
            result[k] = {"label": lbl, "style": {"fontSize": "11px"}}
    return result


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
    styled_marks = _bold_default_mark(marks, slider_val)
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
            marks=styled_marks,
            className="param-slider",
        ),
    ])


def _toggle_block(switch_id: str, label, hint_children: list, default: bool = False) -> html.Div:
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
        html.Div(id="nexpmax-display"),
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
                html.Div(id="subnight-limit-display"),
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
            ["t", html.Sub("OH"), " approximation"],
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
            "regime-color-switch",
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


def _physics_section() -> list:
    _deuc_gpc_default = round(_phys_defaults.D_euc_cm * CM_TO_GPC, 2)  # cm → Gpc, ≈ 5.28
    _rho_log_default = round(math.log10(_phys_defaults.rho_grb_gpc3_yr), 3)  # ≈ 2.415
    _gamma0_log_default = round(math.log10(_phys_defaults.gamma0), 2)        # = 2.5
    return [
        # ── Physical model ─────────────────────────────────────────
        html.P("Physical model", className="param-group-label"),
        _param_block(
            "p",
            "Electron power-law index (must be > 2). Exponentially amplified in flux; "
            "strongly affects spectral slope and all detection ranges.",
            "p_slider", "p_input",
            2.01, 3.0, 0.01, _phys_defaults.p,
            {2.01: "2", 2.5: "2.5", 3: "3"},
            input_step=0.01,
        ),
        _param_block(
            ["log ν [Hz]"],
            "log₁₀ of observing frequency in Hz. Optical/near-IR window: "
            "10^14.3 Hz (~1 μm, J-band) to 10^15.1 Hz (~80 nm, near-UV). "
            "Default 5×10¹⁴ Hz = optical V-band (550 nm).",
            "nu_log_slider", "nu_log_input",
            14.3, 15.1, 0.01, round(math.log10(_phys_defaults.nu_hz), 1),
            {14.3: "14.3", 14.7: "14.7", 15.1: "15.1"},
            input_step=0.01,
        ),
        _param_block(
            ["log E", html.Sub("k,iso"), " [erg]"],
            "log₁₀ of isotropic-equivalent kinetic energy in erg. "
            "Scales flux as E^{(p+3)/4} and deceleration time as E^{1/3}.",
            "Ekiso_log_slider", "Ekiso_log_input",
            51.0, 55.0, 0.1, 53.0,
            {51: "51", 52: "52", 53: "53", 54: "54", 55: "55"},
            input_step=0.1,
        ),
        _param_block(
            ["log n", html.Sub("0"), " [cm⁻³]"],
            "log₁₀ of ISM number density in cm⁻³. "
            "Affects flux (∝ n^{1/2}) and deceleration time (∝ n^{-1/3}).",
            "n0_log_slider", "n0_log_input",
            -3.0, 2.0, 0.1, round(math.log10(_phys_defaults.n0_cm3), 1),
            {-3: "-3", -2: "-2", -1: "-1", 0: "0", 1: "1", 2: "2"},
            input_step=0.1,
        ),
        _param_block(
            ["log Γ", html.Sub("0")],
            "log₁₀ of initial bulk Lorentz factor. Affects deceleration time (∝ Γ₀^{-8/3}) "
            "and the on-axis beaming cone angle.",
            "gamma0_log_slider", "gamma0_log_input",
            2.0, 3.5, 0.05, _gamma0_log_default,
            {2: "2", 2.5: "2.5", 3: "3", 3.5: "3.5"},
            input_step=0.05,
        ),
        _param_block(
            ["θ", html.Sub("j"), " [rad]"],
            "Jet half-opening angle in radians. Sets the jet-break time and "
            "the beaming fraction f_b = θ_j²/2.",
            "thetaj_slider", "thetaj_input",
            0.01, 0.5, 0.01, _phys_defaults.theta_j_rad,
            {0.01: "0.01", 0.1: "0.1", 0.3: "0.3", 0.5: "0.5"},
            input_step=0.01,
        ),
        _param_block(
            ["log ε", html.Sub("e")],
            "log₁₀ of electron energy fraction. Enters flux as (ε_e·(p−2)/(p−1))^{p−1}.",
            "epse_slider", "epse_input",
            -2.0, -0.3, 0.05, round(math.log10(_micro_defaults.epsilon_e), 2),
            {-2: "-2", -1: "-1", -0.3: "-0.3"},
            input_step=0.05,
        ),
        _param_block(
            ["log ε", html.Sub("B")],
            "log₁₀ of magnetic energy fraction. Enters flux as ε_B^{(p+1)/4}.",
            "epsB_slider", "epsB_input",
            -3.0, -1.0, 0.05, round(math.log10(_micro_defaults.epsilon_B), 2),
            {-3: "-3", -2: "-2", -1: "-1"},
            input_step=0.05,
        ),
        # ── Cosmological model ─────────────────────────────────────
        html.P("Cosmological model", className="param-group-label"),
        _param_block(
            ["D", html.Sub("Euc"), " [Gpc]"],
            "Euclidean calibration distance in Gpc. Sets the volume for the rate integral. "
            "Default 5.28 Gpc ≈ z = 2 in flat ΛCDM.",
            "deuc_slider", "deuc_input",
            1.0, 12.0, 0.01, _deuc_gpc_default,
            {1: "1", 5.28: "5.28", 8: "8", 12: "12"},
            input_step=0.01,
        ),
        _param_block(
            ["log ℛ [Gpc⁻³ yr⁻¹]"],
            "log₁₀ of GRB volumetric rate density. Scales the total detection rate linearly.",
            "rho_grb_log_slider", "rho_grb_log_input",
            1.0, 3.3, 0.005, _rho_log_default,
            {1: "10", 2: "100", 2.415: "260", 3: "1k", 3.3: "2k"},
            input_step=0.005,
        ),
        html.Div(
            className="param-row grb-derived-row",
            children=[
                html.Span(id="grb-ntotal-display", className="grb-derived-val"),
                html.Span(id="grb-ntoward-display", className="grb-derived-val"),
            ],
        ),
    ]


# ── Public factory ───────────────────────────────────────────────────────────

def create_sidebar() -> html.Aside:
    """Return the full collapsible sidebar element."""
    accordion = dbc.Accordion(
        id="param-accordion",
        always_open=True,   # DBC: multiple sections can be open simultaneously; opening one does NOT auto-close others
        active_item=["strategy", "instrument", "constraints"],
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
                _physics_section(),
                title="Parameters",
                item_id="parameters",
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
