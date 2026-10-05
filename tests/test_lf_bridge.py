"""Bridge tests for scale-free LF parameters and selected-mode distributions."""
from __future__ import annotations

import numpy as np
import pytest
import standalone_bridge as sb


@pytest.fixture(scope="module", autouse=True)
def _small_grids():
    saved = (sb.NX_DEFAULT, sb.NY_DEFAULT, sb.NX_REGIME, sb.NY_REGIME)
    sb.NX_DEFAULT, sb.NY_DEFAULT, sb.NX_REGIME, sb.NY_REGIME = 24, 30, 24, 30
    yield
    sb.NX_DEFAULT, sb.NY_DEFAULT, sb.NX_REGIME, sb.NY_REGIME = saved


def _params(**over):
    params = dict(i_det=2, A_log=-4.68, f_live=0.1, t_overhead_s=0.0,
        omega_exp_deg2=47.0, omega_srv_deg2=27500.0, t_night_h=10.0,
        p=2.2, nu_log10=14.7, E_kiso_log10=53.0, n0_log10=0.0,
        epsilon_e_log10=-1.0, epsilon_B_log10=-4.0, theta_j_rad=0.1,
        gamma0_log10=2.5, D_euc_gpc=4.55, rho_grb_log10=2.415,
        optical_survey=False, color_regimes=False, full_integral=False,
        qmin=0.0, Dmin_cm=0.0)
    params.update(over)
    return params


LF_ON = dict(lf_on=True, lf_alpha=-2.0, lf_log10_A=0.0)


def test_old_params_without_lf_keys_still_work():
    payload = sb.compute_all(_params())
    assert payload["error"] is None
    assert payload["lf_applied"] is False
    assert payload["R_int_yr"] > 0


def test_lf_off_payload_identical_to_absent_keys():
    base = sb.compute_all(_params())
    off = sb.compute_all(_params(lf_on=False, lf_alpha=-2.49, lf_log10_A=3))
    assert base["error"] is None and off["error"] is None
    for key in ("Z_flat", "Z_log_flat", "q_med_flat", "D_med_Gpc_flat",
                "N_opt", "t_cad_opt_s", "R_opt"):
        assert base[key] == off[key], key


@pytest.mark.parametrize("full_on", [False, True], ids=["dominant", "exact"])
@pytest.mark.parametrize("optical", [False, True], ids=["nonopt", "optical"])
def test_lf_on_computes_and_echoes(full_on, optical):
    payload = sb.compute_all(_params(**LF_ON, full_integral=full_on,
        optical_survey=optical, color_regimes=True, s_fade=0.3, s_rise=0.3))
    assert payload["error"] is None
    assert payload["lf_applied"] is True
    assert payload["lf_L_ref_erg_s_Hz"] == 1e32
    for key in ("R_int_yr", "R_toward_day", "F_nu_tdec_Jy",
                "lf_L_med_pop_erg_s", "L0_erg_s"):
        assert payload.get(key) is None, key
    rates = np.array([v for v in payload["Z_flat"] if v is not None])
    assert rates.size and np.all(np.isfinite(rates)) and np.all(rates > 0)
    regimes = np.array([v for v in payload["regime_flat"] if v is not None])
    assert regimes.size and set(np.unique(regimes)).issubset(set(range(1, 8)))


@pytest.mark.parametrize("full_on", [False, True])
def test_bridge_normalization_scales_rates_but_not_medians(full_on):
    params = _params(**LF_ON, full_integral=full_on)
    low = sb.compute_all(params)
    high = sb.compute_all({**params, "lf_log10_A": 1.0})
    assert low["error"] is None and high["error"] is None
    assert high["R_opt"] == pytest.approx(10*low["R_opt"], rel=2e-7)
    assert high["N_opt"] == pytest.approx(low["N_opt"], rel=1e-7)
    assert high["t_cad_opt_s"] == pytest.approx(low["t_cad_opt_s"], rel=1e-7)
    for key in ("q_med_flat", "D_med_Gpc_flat"):
        assert np.allclose(np.asarray(low[key], float), np.asarray(high[key], float),
                           rtol=1e-8, atol=0, equal_nan=True)


@pytest.mark.parametrize("full_on", [False, True])
def test_lf_qdview_matches_selected_rate_and_true_totals(full_on):
    params = _params(**LF_ON, full_integral=full_on, qmin=0.4,
                     Dmin_cm=0.01*sb.GPC_TO_CM)
    N, cadence = 100.0, 86400.0
    payload = sb.compute_qdview(params, N, cadence)
    assert payload["error"] is None
    model = sb._build_models(params)["model_day"]
    rate = model.rate_log10_full_integral if full_on else model.rate_log10
    expected = 10**float(np.asarray(rate(2, N, cadence,
                         q_min=params["qmin"], D_min_cm=params["Dmin_cm"])))
    assert payload["qdview_total_rate_q"] == pytest.approx(expected, rel=2e-7)
    assert payload["qdview_total_rate_D"] == pytest.approx(expected, rel=2e-7)
    for axis in ("q", "D"):
        cumulative = np.asarray(payload[f"qdview_R{axis}_cum_flat"], float)
        differential = np.asarray(payload[f"qdview_R{axis}_diff_flat"], float)
        assert np.all(np.isfinite(cumulative)) and np.all(np.isfinite(differential))
        assert np.all(np.diff(cumulative) <= 1e-9*expected)
        assert cumulative[-1] == pytest.approx(0, abs=1e-10*expected)
        assert np.all(differential >= 0)


def test_qdview_discloses_faint_end_mass_below_plot_range():
    params = _params(lf_on=True, lf_alpha=-2.49, lf_log10_A=0,
                     full_integral=True)
    payload = sb.compute_qdview(params, 100, 86400)
    assert payload["error"] is None
    distance = payload["qdview_D_grid_Gpc_flat"][0]
    expected_fraction = (distance/params["D_euc_gpc"])**0.02
    assert payload["qdview_D_below_plot_fraction"] == pytest.approx(expected_fraction, rel=1e-7)
    assert payload["qdview_D_below_plot_fraction"] > 0.8
    assert payload["qdview_RD_cum_flat"][0]/payload["qdview_total_rate_D"] == pytest.approx(
        1-expected_fraction, rel=1e-7)


@pytest.mark.parametrize("full_on", [False, True])
def test_lf_slices_compute_in_both_modes(full_on):
    params = _params(**LF_ON, full_integral=full_on)
    n = sb.compute_nslice(params, 86400.0)
    t = sb.compute_tslice(params, 100.0)
    assert n["error"] is None and t["error"] is None
    rates = [v for v in n["N_sweep_R_flat"] if v is not None]
    assert rates and all(np.isfinite(v) and v > 0 for v in rates)


@pytest.mark.parametrize("alpha", [-2.5, -1, -3, 0])
def test_nonconvergent_lf_is_rejected_at_bridge(alpha):
    payload = sb.compute_all(_params(lf_on=True, lf_alpha=alpha, lf_log10_A=0))
    assert payload["error"] is not None


def test_inactive_normalizations_and_fdec_override_do_not_change_lf():
    params = _params(**LF_ON, full_integral=True)
    base = sb.compute_all(params)
    changed = sb.compute_all({**params, "rho_grb_log10": 0.5,
        "nu_log10": 15.1, "epsilon_e_log10": -2, "epsilon_B_log10": -1,
        "F_dec_override_Jy": 0.5})
    assert base["error"] is None and changed["error"] is None
    assert changed["F_dec_override_applied"] is False
    for key in ("Z_log_flat", "q_med_flat", "D_med_Gpc_flat"):
        assert np.allclose(np.asarray(base[key], float), np.asarray(changed[key], float),
                           rtol=1e-9, atol=0, equal_nan=True), key


@pytest.mark.parametrize("full_on", [False, True])
def test_batched_lf_day_overlays_match_individual_strategy_evaluation(full_on):
    params = _params(**LF_ON, full_integral=full_on, optical_survey=True,
                     win_tp=full_on, s_rise=0.2, s_fade=0.1)
    state = sb._build_models(params)
    columns = np.array([1.0, 30.0, 100.0, 600.0])
    days, ns, rates, regimes, exposures, qs, ds = sb._build_day_line_arrays(
        model_day=state["model_day"], i_det=2, N_cols=columns,
        t_cad_max_s=4*sb.DAY_S, full_integral=full_on,
        s_rise=0.2, s_fade=0.1)
    for row, day in enumerate(days):
        for col, number in enumerate(columns):
            expected = sb._eval_point(number, day*sb.DAY_S, 2,
                state["model_day"], state["model_night"],
                state["f_live"], state["f_live_night"], state["f_night"],
                True, False, 0, full_integral=full_on, s_rise=0.2, s_fade=0.1)
            if not np.isfinite(expected[0]) or expected[0] < 10**sb.ZMIN_DISPLAY_LOG10:
                expected = [np.nan]*4
            actual = [rates[row, col], exposures[row, col], qs[row, col], ds[row, col]]
            assert np.allclose(actual, expected, rtol=1e-8, atol=0, equal_nan=True)
    assert np.array_equal(ns, np.broadcast_to(columns, ns.shape))
