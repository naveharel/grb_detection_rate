"""TEMP-FDEC-OVERRIDE — temporary bridge-level tests for the F_dec override slider.

Delete this whole file when the override slider is stripped
(grep for TEMP-FDEC-OVERRIDE).

The bridge applies a copy-on-write F_dec override in ``_build_models`` when
``params["F_dec_override_Jy"]`` is a positive finite number. Scaling
F_dec/F_j/F_nr by one factor k must be exactly equivalent to a native rebuild
with ``A_log -= log10(k)`` (the invariant pinned for the figures-side helper
in tests/test_fdec_override.py).

Run with::

    .venv/Scripts/python -m pytest tests/test_fdec_override_bridge.py -v
"""

from __future__ import annotations

import math

import pytest

import standalone_bridge as sb
from grb_detect.constants import DAY_S


# UI default slider values (web/template.html). t_overhead_s = 0 and optical
# off keep the A_log-equivalence check free of validity-mask complications.
def _params(**overrides) -> dict:
    params = {
        "i_det": 10,
        "A_log": -4.68,
        "f_live": 0.2,
        "t_overhead_s": 0.0,
        "omega_exp_deg2": 47.0,
        "omega_srv_deg2": 27500.0,
        "t_night_h": 10.0,
        "optical_survey": False,
        "color_regimes": False,
        "full_integral": False,
        "qmin": 0.0,
        "Dmin_cm": 0.0,
        "s_fade": 0.0,
        "s_rise": 0.0,
        "s_mode": "discrete",
        "toh_approx": False,
        "p": 2.5,
        "nu_log10": 14.7,
        "E_kiso_log10": 53.0,
        "n0_log10": 0.0,
        "epsilon_e_log10": -1.0,
        "epsilon_B_log10": -2.0,
        "theta_j_rad": 0.1,
        "gamma0_log10": 2.5,
        "D_euc_gpc": 5.28,
        "rho_grb_log10": 2.415,
    }
    params.update(overrides)
    return params


def _rate_at(state: dict, N_exp: float, t_cad_s: float) -> float:
    R, _t_exp, _q_med, _D_med = sb._eval_point(
        N_exp, t_cad_s, state["i_det"],
        state["model_day"], state["model_night"],
        state["f_live"], state["f_live_night"], state["f_night"],
        state["optical_on"], state["toh_approx"], state["t_overhead_s"],
    )
    return R


@pytest.fixture(scope="module")
def F0() -> float:
    """Physics-derived F_dec at UI defaults (no override)."""
    return float(sb._build_models(_params())["model_day"].derived.F_dec_Jy)


def test_override_sets_effective_F_dec(F0):
    k = 10.0
    base = sb._build_models(_params())
    st = sb._build_models(_params(F_dec_override_Jy=k * F0))
    d0, d1 = base["model_day"].derived, st["model_day"].derived

    assert st["fdec_override_applied"] is True
    assert base["fdec_override_applied"] is False
    assert d1.F_dec_Jy == pytest.approx(k * F0, rel=1e-12)
    # Flux-ladder coupling preserved: F_j/F_dec and F_nr/F_dec invariant.
    assert d1.F_j_Jy / d1.F_dec_Jy == pytest.approx(d0.F_j_Jy / d0.F_dec_Jy, rel=1e-12)
    assert d1.F_nr_Jy / d1.F_dec_Jy == pytest.approx(d0.F_nr_Jy / d0.F_dec_Jy, rel=1e-12)
    # Geometric/temporal scales untouched (bit-identical).
    assert d1.t_dec_s == d0.t_dec_s
    assert d1.t_j_s == d0.t_j_s
    assert d1.q_dec == d0.q_dec
    assert d1.q_j == d0.q_j
    assert d1.q_nr == d0.q_nr


def test_override_matches_native_A_log_shift_rate(F0):
    # F_dec -> 10*F0 must reproduce a native rebuild with A_log -= 1.
    st_ov = sb._build_models(_params(F_dec_override_Jy=10.0 * F0))
    st_al = sb._build_models(_params(A_log=-4.68 - 1.0))
    N_exp, t_cad_s = 585.0, 2.0 * DAY_S
    R_ov = _rate_at(st_ov, N_exp, t_cad_s)
    R_al = _rate_at(st_al, N_exp, t_cad_s)
    assert math.isfinite(R_ov) and R_ov > 0
    assert R_ov == pytest.approx(R_al, rel=1e-10)


def test_override_absent_or_null_is_noop(F0):
    st_absent = sb._build_models(_params())
    st_null = sb._build_models(_params(F_dec_override_Jy=None))
    st_ident = sb._build_models(_params(F_dec_override_Jy=F0))

    assert st_absent["fdec_override_applied"] is False
    assert st_null["fdec_override_applied"] is False
    assert st_ident["fdec_override_applied"] is True

    assert st_absent["model_day"].derived.F_dec_Jy == F0
    assert st_null["model_day"].derived.F_dec_Jy == F0
    # Identity override: k = 1.0 multiply is exact.
    assert st_ident["model_day"].derived.F_dec_Jy == F0


def test_no_cache_mutation(F0):
    # Override must never leak into the lru_cached model.
    sb._build_models(_params(F_dec_override_Jy=100.0 * F0))
    st = sb._build_models(_params())
    assert st["model_day"].derived.F_dec_Jy == F0


def test_optical_night_model_also_overridden(F0):
    k = 10.0
    base = sb._build_models(_params(optical_survey=True))
    st = sb._build_models(_params(optical_survey=True, F_dec_override_Jy=k * F0))
    F0_night = float(base["model_night"].derived.F_dec_Jy)
    assert st["model_night"] is not None
    assert st["model_night"].derived.F_dec_Jy == pytest.approx(k * F0_night, rel=1e-12)
