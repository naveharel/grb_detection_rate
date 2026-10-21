"""Integer strategy sampling and inclusive, option-dependent optical windows."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import grb_detect.core as core
import standalone_bridge as sb


BASE = dict(i_det=2, A_log=-4.68, f_live=0.166, t_overhead_s=15,
            omega_exp_deg2=47, omega_srv_deg2=41253, t_night_h=4,
            optical_survey=True, color_regimes=True, full_integral=False,
            D_euc_gpc=8.66, rho_grb_log10=math.log10(138), theta_j_rad=0.1,
            p=2.2, nu_log10=14.7, E_kiso_log10=53, n0_log10=0,
            epsilon_e_log10=-1, epsilon_B_log10=-4, gamma0_log10=2.5)


@pytest.mark.parametrize("minus_one", [False, True])
@pytest.mark.parametrize("i", [2, 3, 6])
def test_window_equality_valid_but_next_cadence_invalid(minus_one, i):
    boundary = 14400 / (i - int(minus_one))
    cadences = np.array([boundary, np.nextafter(boundary, math.inf), 86400, 1.5 * 86400, 2 * 86400])
    actual, valid = core.optical_survey_tcad_seconds(
        cadences, i_det=i, t_night_s=14400, win_i_minus_one=minus_one)
    np.testing.assert_array_equal(actual, cadences)
    np.testing.assert_array_equal(valid, [True, False, True, False, True])


def test_zero_length_detection_window_has_no_subday_gap():
    _, valid = core.optical_survey_tcad_seconds(
        np.array([3600, 23 * 3600, 86400, 1.5 * 86400]),
        i_det=1, t_night_s=14400, win_i_minus_one=True)
    np.testing.assert_array_equal(valid, [True, True, True, False])


@pytest.mark.parametrize("maximum", [1, 53.19, 877.72, 10000])
def test_field_grid_is_unique_integer_and_covers_typical_counts(maximum):
    fields = core.build_N_exp_grid(1, maximum, 160)
    assert np.all(np.diff(fields) > 0)
    np.testing.assert_array_equal(fields, np.rint(fields))
    assert fields[0] == 1 and fields[-1] == math.floor(maximum)
    for anchor in (30, 50, 100, 150, 200):
        if anchor <= maximum:
            assert anchor in fields


@pytest.mark.parametrize("minus_one", [False, True])
@pytest.mark.parametrize("lf_on,exact", [(False, False), (False, True), (True, False), (True, True)])
def test_surface_slice_and_point_share_boundary_and_rates(minus_one, lf_on, exact):
    params = {**BASE, "win_iminus1": minus_one, "lf_on": lf_on,
              "full_integral": exact, "win_tp": exact}
    state = sb._build_models(params)
    boundary = 14400 / (2 - int(minus_one))
    X, T, _, Z, regimes, exposure, q, D = core.compute_surface(
        state["model_day"], state["model_night"], 2, optical_survey=True,
        color_regimes=True, t_night_s=14400, nx=20, ny=30, full_integral=exact)
    np.testing.assert_array_equal(X, np.rint(X))
    for t in (3600, 7200, boundary):
        assert t in T
    row, col = np.argwhere((T == boundary) & (X == 50))[0]
    assert np.isfinite(Z[row, col])
    point = sb._eval_point(50, boundary, 2, state["model_day"], state["model_night"],
        state["f_live"], state["f_live_night"], state["f_night"], True, False, 15,
        full_integral=exact)
    np.testing.assert_allclose(point, [10**Z[row, col], exposure[row, col], q[row, col], D[row, col]], rtol=1e-9)
    gap = (T > boundary) & (T < 86400)
    for values in (Z, regimes, exposure, q, D):
        assert np.all(np.isnan(values[gap]))

    valid = sb.compute_nslice(params, boundary)
    k = valid["N_sweep_flat"].index(50)
    assert valid["N_sweep_R_flat"][k] == pytest.approx(point[0], rel=1e-9)
    invalid = sb.compute_nslice(params, boundary + 1)
    for key in ("N_sweep_R_flat", "N_sweep_t_exp_flat", "N_sweep_q_med_flat", "N_sweep_D_med_Gpc_flat", "N_sweep_regime_flat"):
        assert all(v is None for v in invalid[key])
    distribution = sb.compute_qdview(params, 50, boundary + 1)
    assert distribution["error"] is None
    assert distribution["qdview_total_rate_q"] is None
    assert distribution["qdview_total_rate_D"] is None
    tslice = sb.compute_tslice(params, 50.01)
    assert tslice["error"] is None
    assert tslice["N_fix"] == 50
    assert tslice["t_cont_h_flat"][-1] == boundary / 3600
    assert tslice["t_cont_R_flat"][-1] == pytest.approx(point[0], rel=1e-9)


@pytest.mark.parametrize("max_fields", [1.5, 2.5, 5.5, 877.7])
def test_nonoptical_adaptive_grid_uses_whole_fields(max_fields):
    state = sb._build_models({**BASE, "optical_survey": False,
                             "omega_srv_deg2": 47 * max_fields})
    X, T, *_ = core.compute_surface(state["model_day"], None, 2,
        optical_survey=False, color_regimes=False, t_night_s=14400, nx=30, ny=40)
    np.testing.assert_array_equal(X, np.rint(X))
    assert X.min() >= 1 and X.max() <= math.floor(state["N_exp_max"])
    assert 3600 in T and 7200 in T


@pytest.mark.parametrize("minus_one", [False, True])
def test_optimizer_can_reach_inclusive_boundary_with_integer_fields(monkeypatch, minus_one):
    boundary = 14400 / (3 - int(minus_one))
    model = SimpleNamespace(win_i_minus_one=minus_one)
    # A known maximum exactly on the admissibility boundary and at one whole field count.
    def rate(model, i, N, t, full, **kwargs):
        return 10 - (N - 53)**2 - ((t - boundary) / 1000)**2
    monkeypatch.setattr(core, "_rate", rate)
    N, t, _ = core.maximize_log_surface_iterative(model, model, 3,
        0, math.log10(80), 0, math.log10(boundary * 1.5),
        optical_survey=True, t_night_s=14400, n0x=30, n0y=30, nfx=50, nfy=50)
    assert N == 53
    assert t == boundary
