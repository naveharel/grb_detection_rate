"""Bounded numerical checks for the LF/ZTF diagnostic study.

These exercise real engine distributions and independent single-luminosity
references. They do not run the parameter search or write result artifacts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import lf_rate_medians as study
from analysis import ztf_validation as ztf


SELECTED_LF = (-1.75, 42.25, 46.25)
CONTEXTS = list((mode, window) for mode in study.MODE_KEYS for window in study.WINDOWS)

# Obtained independently from the original ZTF settings and direct engine
# rate/q/D integrals at N_q = N_D = 1600, before the study script was written.
# Values are coverage-adjusted rate [/yr], median q, median D [Gpc].
DEFAULT_REFERENCE = {
    ("public", "conservative"): (0.74308659199, 1.24452132877, 1.73212536223),
    ("public", "optimistic"): (2.41576450759, 1.21551623034, 2.16483879299),
    ("high_cadence", "conservative"): (0.91507398151, 1.18621453251, 2.43299450085),
    ("high_cadence", "optimistic"): (1.03149063968, 1.16475222995, 2.45736512373),
}


@pytest.fixture(scope="module")
def verified():
    # Also exercises the real resolution-doubling acceptance gate and coverage
    # summaries. Only two LFs are evaluated; the search is never invoked.
    return study.verify_highlights({"default": study.DEFAULT_LF, "selected": SELECTED_LF})


@pytest.mark.parametrize("mode,window", CONTEXTS)
def test_default_lf_matches_independent_baseline(verified, mode, window):
    got = verified["default"]["result"]["windows"][window]["modes"][mode]
    rate, q_med, distance = DEFAULT_REFERENCE[mode, window]
    assert got["rate"] == pytest.approx(rate, rel=0.005)
    assert got["q_med"] == pytest.approx(q_med, abs=0.003)
    assert got["D_med_Gpc"] == pytest.approx(distance, rel=0.01)


@pytest.mark.parametrize("mode_name,window", CONTEXTS)
@pytest.mark.parametrize("alpha", [-3.5, 0.0])
def test_collapsed_lf_matches_independent_scaled_single_luminosity(mode_name, window, alpha):
    mode = ztf.MODES[study.MODE_KEYS[mode_name]]
    params = dict(ztf.BASE_PARAMS, **ztf.PHYS_NEW)
    params.update(i_det=mode["i_det"], f_live=mode["f_live"],
                  omega_srv_deg2=mode["omega_srv_deg2"], lf_on=False,
                  win_tp=True, win_iminus1=window == "optimistic")
    state = ztf.bridge._build_models(params)
    is_night = mode["t_cad_s"] < ztf.DAY_S
    base = state["model_night"] if is_night else state["model_day"]
    scaled = base._lf_scaled_copy(10.0 ** 43.5 / base.L0_erg_s())
    log_rate = scaled.rate_log10_full_integral(
        mode["i_det"], np.array([mode["N_exp"]]), np.array([mode["t_cad_s"]]),
        N_q=800, s_fade=0.3, s_rise=0.5, s_mode="discrete",
    )
    expected = float(10.0 ** log_rate.item())
    if is_night:
        expected *= ztf.T_NIGHT_S / ztf.DAY_S
    actual = study.rate_at((alpha, 43.5, 43.5), mode_name, window, nq=800)
    assert actual["raw_rate"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("mode_name,window", CONTEXTS)
@pytest.mark.parametrize("label,lf", [("default", study.DEFAULT_LF), ("selected", SELECTED_LF)])
def test_real_marginals_obey_geometry_and_distance_ceiling(verified, label, lf, mode_name, window):
    mode, model, night_factor = study.build_state(lf, mode_name, window)
    args = (mode["i_det"], mode["N_exp"], mode["t_cad_s"])
    q, dq = model.dR_dq_full_integral(*args, N_q=800, **study.SELECTION)
    d, dd = model.dR_dD_full_integral(*args, N_q=800, N_D=800, **study.SELECTION)
    cq, q_mass = study.cumulative(q, dq)
    cd, d_mass = study.cumulative(d, dd)
    for cdf in (cq, cd):
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0
        assert np.all(np.diff(cdf) >= -1e-14)
    assert d_mass == pytest.approx(q_mass, rel=0.01)

    # At q < 1 the light curve and selection are angle-independent. The
    # orientation measure q dq therefore gives F(0.5)/F(1) = 1/4. The finite
    # engine grid omits the tiny [0, q_first] interval, hence this tolerance.
    assert np.interp(0.5, q, cq) / np.interp(1.0, q, cq) == pytest.approx(0.25, abs=0.001)

    # Selection decreases with distance. The detected radial CDF must dominate
    # that of a complete uniform sphere, and its median cannot exceed the
    # complete-volume median D_Euc * 2^(-1/3).
    assert np.all(cd >= (d / model.phys.D_euc_cm) ** 3 - 1e-5)
    stats = verified[label]["result"]["windows"][window]["modes"][mode_name]
    ceiling_gpc = model.phys.D_euc_cm / ztf.GPC_TO_CM * 2.0 ** (-1.0 / 3.0)
    assert 0 < stats["D_med_Gpc"] <= ceiling_gpc
    assert stats["D_med_Gpc"] <= stats["D_90_Gpc"] <= 4.55
    assert q_mass * night_factor == pytest.approx(stats["raw_rate"], rel=0.01)
    assert 0 <= stats["frac_q_lt_1"] <= stats["frac_q_lt_1p5"] <= 1
    assert 0 <= stats["frac_q_gt_2"] <= 1 - stats["frac_q_lt_1p5"]
    assert stats["theta_med_deg"] == pytest.approx(math.degrees(0.1 * stats["q_med"]))


@pytest.mark.parametrize("label", ["default", "selected"])
def test_verified_coverage_summaries_and_same_endpoint_acceptance(verified, label):
    summary = verified[label]
    assert all(check["passes"] for check in summary["checks"])
    for window, result in summary["result"]["windows"].items():
        for efficiency in (0.2, 0.35, 0.5):
            coverage = summary["coverage"][str(efficiency)][window]
            accepted = []
            for mode, stats in result["modes"].items():
                rate = coverage["rates"][mode]
                assert rate == pytest.approx(stats["rate"] * efficiency / 0.35, rel=1e-12)
                target = study.TARGETS[mode]
                accepted.append(target["rate"] / 3 <= rate <= target["rate"] * 3
                                and target["distance"] * 0.65 <= stats["D_med_Gpc"]
                                <= target["distance"] * 1.35)
            # Both surveys must pass under this very same window prescription.
            assert coverage["passes"] == all(accepted)
    if label == "default":
        assert summary["result"]["classification"] == "fails"
    else:
        assert summary["result"]["classification"] == "passes_both_endpoints"
