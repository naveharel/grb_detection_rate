"""Independent scientific checks for the app's unbounded luminosity intensity.

The reference integrates the thesis piecewise light curve directly. It does not
call scale_free_lf kernels or reuse their quadrature/normalization functions.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from functools import lru_cache
import math

import numpy as np
import pytest

import grb_detect.scale_free_lf as powerlaw
from grb_detect.core import make_rate_model, make_finite_cutoff_rate_model
from grb_detect.detection_rate import PowerLawLuminosityFunction
from grb_detect.params import GPC_TO_CM

BASE = dict(A_log=-4.68, f_live=0.1, t_overhead_s=0.0, omega_exp_deg2=47.0,
            p=2.5, theta_j_rad=0.1, gamma0_log10=2.5, D_euc_gpc=4.55)
LF = dict(lf_on=True, lf_alpha=-2.0, lf_log10_A=0.0)
N, CADENCE = 100.0, 86400.0


def model(**changes):
    return make_rate_model(**{**BASE, **LF, **changes})


def rate(m, full=True, cadence=CADENCE, **selection):
    fn = m.rate_log10_full_integral if full else m.rate_log10
    return float(10**np.asarray(fn(2, N, cadence, **selection)))


@lru_cache
def _gauss(n=320):
    return np.polynomial.legendre.leggauss(n)


def _lightcurve(m, time):
    time = np.maximum(time, m.derived.t_dec_s)
    first_slope = 3*(m.phys.p-1)/4
    jet_flux = (m.derived.t_j_s/m.derived.t_dec_s)**(-first_slope)
    return np.where(time <= m.derived.t_j_s,
        (time/m.derived.t_dec_s)**(-first_slope),
        jet_flux*(time/m.derived.t_j_s)**(-m.phys.p))


def _peak_time(m, q):
    off_axis = np.maximum(np.asarray(q)-1, 0)
    return np.maximum(m.derived.t_dec_s, m.derived.t_j_s *
                      np.where(np.asarray(q) < 2, off_axis**(8/3), off_axis**2))


def _inverse_peak(m, time):
    if time <= m.derived.t_dec_s:
        return m.derived.q_dec
    return 1+(time/m.derived.t_j_s)**(3/8 if time < m.derived.t_j_s else 0.5)


def _angular_integral(m, exponent, cadence=CADENCE, lower=0, top=None):
    top = m.derived.q_nr if top is None else top
    required_time = (1 if m.win_i_minus_one else 2)*cadence
    boundaries = [lower, top, m.derived.q_dec, 2]
    if m.win_from_peak:
        boundaries.append(_inverse_peak(m, m.derived.t_j_s-required_time))
    else:
        boundaries.append(_inverse_peak(m, required_time))
    boundaries = sorted(set([lower, top]+[x for x in boundaries if lower < x < top]))
    nodes, weights = _gauss()
    result = 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        q = (left+right)/2+(right-left)/2*nodes
        peak = _peak_time(m, q)
        when = peak+required_time if m.win_from_peak else np.maximum(peak, required_time)
        result += (right-left)/2*np.dot(weights, q*_lightcurve(m, when)**exponent)
    return float(result)


def _reference_rate(m, cadence=CADENCE):
    alpha = m.lf.alpha
    limiting_flux = float(m.F_lim_Jy(np.array([m.t_exp_s(N, cadence)]))[0])*1e-23
    corner_luminosity = 4*np.pi*m.phys.D_euc_cm**2*limiting_flux
    return (float(m.f_Omega(N))*4*np.pi*(m.phys.D_euc_cm/GPC_TO_CM)**3
            * m.phys.theta_j_rad**2*m.lf.A
            * (corner_luminosity/m.lf.L_ref)**(alpha+1)
            / ((-alpha-1)*(2*alpha+5))
            * _angular_integral(m, -alpha-1, cadence))


@pytest.mark.parametrize("alpha", [-2.49, -2.4, -2, -1.75, -1.25, -1.01])
@pytest.mark.parametrize("cadence", [300.0, 86400.0, 5*86400.0])
def test_full_rate_matches_independent_piecewise_quadrature(alpha, cadence):
    m = model(lf_alpha=alpha)
    assert rate(m, cadence=cadence) == pytest.approx(_reference_rate(m, cadence), rel=1e-7)


@pytest.mark.parametrize("flags", [
    dict(win_i_minus_one=True), dict(win_from_peak=True),
    dict(win_i_minus_one=True, win_from_peak=True)])
def test_full_peak_window_matches_independent_time_selection(flags):
    m = model(**flags)
    assert rate(m) == pytest.approx(_reference_rate(m), rel=1e-7)


@pytest.mark.parametrize("alpha", [-2.5, -1, -3, 0, np.nan, np.inf])
def test_nonconvergent_or_nonfinite_slopes_are_rejected(alpha):
    with pytest.raises(ValueError):
        PowerLawLuminosityFunction(alpha=alpha)
    with pytest.raises(ValueError):
        model(lf_alpha=alpha)


def test_factory_distinguishes_scale_free_and_bounded_reference():
    m = model()
    assert isinstance(m.lf, PowerLawLuminosityFunction)
    assert m.lf.alpha == -2 and m.lf.A == 1 and m.lf.L_ref == 1e32
    assert make_rate_model(**BASE).lf is None
    assert make_rate_model(**BASE) is make_rate_model(**BASE, lf_on=False)
    with pytest.raises(TypeError):
        make_rate_model(**BASE, lf_on=True, lf_log10_L_min=42.5)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("alpha", [-2.4, -2, -1.25])
def test_normalization_flux_and_radius_scalings(full, alpha):
    m = model(lf_alpha=alpha)
    base = rate(m, full)
    assert rate(model(lf_alpha=alpha, lf_log10_A=0.7), full)/base == pytest.approx(10**0.7, rel=1e-8)
    assert rate(model(lf_alpha=alpha, A_log=BASE["A_log"]+0.4), full)/base == pytest.approx(
        10**(0.4*(alpha+1)), rel=1e-7)
    assert rate(model(lf_alpha=alpha, D_euc_gpc=2*BASE["D_euc_gpc"]), full)/base == pytest.approx(
        2**(2*alpha+5), rel=1e-7)


@pytest.mark.parametrize("full", [False, True])
def test_reference_luminosity_and_inactive_normalizations_are_invariant(full):
    m = model()
    reference_changed = copy.copy(m)
    multiplier = 7.0
    reference_changed.lf = replace(m.lf, L_ref=m.lf.L_ref*multiplier,
        log10_A=m.lf.log10_A+(m.lf.alpha+1)*math.log10(multiplier))
    assert rate(reference_changed, full) == pytest.approx(rate(m, full), rel=1e-9)
    inactive_changed = model(rho_grb_log10=-1, nu_log10=15.1,
                             epsilon_e_log10=-2, epsilon_B_log10=-1)
    assert rate(inactive_changed, full) == pytest.approx(rate(m, full), rel=1e-9)
    override = m._lf_scaled_copy(31.0)
    override.lf = m.lf
    assert rate(override, full) == pytest.approx(rate(m, full), rel=1e-9)


@pytest.mark.parametrize("alpha", [-2.49, -2.4, -2, -1.25, -1.01])
@pytest.mark.parametrize("floor", [0, 0.07])
def test_exact_distance_cdf_and_logarithmic_median(alpha, floor):
    m = model(lf_alpha=alpha)
    exponent = 2*alpha+5
    ratios = np.r_[0, np.geomspace(1e-18, 1, 45)]
    distribution = m.lf_distributions(2, N, CADENCE, full_integral=True,
        q_values=np.array([0, 1, m.derived.q_nr]),
        D_values_cm=ratios*m.phys.D_euc_cm,
        q_min=0.25, D_min_cm=floor*m.phys.D_euc_cm)
    expected_survival = (1-np.maximum(ratios, floor)**exponent)/(1-floor**exponent)
    assert np.allclose(distribution["D_survival"]/distribution["total_rate"],
                       expected_survival, rtol=1e-8, atol=1e-11)
    expected_median = ((1+floor**exponent)/2)**(1/exponent)
    assert distribution["D_med_cm"]/m.phys.D_euc_cm == pytest.approx(expected_median, rel=2e-7)
    assert distribution["total_rate"] == pytest.approx(
        rate(m, q_min=0.25, D_min_cm=floor*m.phys.D_euc_cm), rel=1e-8)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("selection", [dict(), dict(q_min=0.5, D_min_cm=0.03*GPC_TO_CM),
    dict(s_fade=0.3, s_rise=0.3),
    dict(s_fade=0.4, s_rise=0.4, rise_random_start=False, fade_random_start=False)])
def test_selected_mode_distributions_and_medians(full, selection):
    m = model()
    q = np.unique(np.r_[np.linspace(0, m.derived.q_nr, 5001),
                        m.derived.q_dec, 2, _inverse_peak(m, 2*CADENCE),
                        selection.get("q_min", 0)])
    # Refine the upper endpoint: the selected density is zero at the boundary,
    # whereas its left limit need not vanish (especially for rectangles).
    D = np.unique(np.r_[0, np.geomspace(1e-12, 1, 5001),
                       np.linspace(0.8, 1, 1001)])*m.phys.D_euc_cm
    result = m.lf_distributions(2, N, CADENCE, full_integral=full,
        q_values=q, D_values_cm=D, **selection)
    total = result["total_rate"]
    assert total == pytest.approx(rate(m, full, **selection), rel=1e-8)
    assert total > 0
    for key in ("q_survival", "D_survival"):
        assert result[key][0] == pytest.approx(total, rel=1e-8)
        assert result[key][-1] == pytest.approx(0, abs=total*1e-10)
        assert np.all(np.diff(result[key]) <= total*1e-10)
    assert np.trapezoid(result["dR_dq"], q) == pytest.approx(total, rel=0.008)
    assert np.trapezoid(result["dR_dD_per_cm"], D) == pytest.approx(total, rel=0.003)
    med = m.lf_distributions(2, N, CADENCE, full_integral=full,
        q_values=np.array([result["q_med"]]),
        D_values_cm=np.array([result["D_med_cm"]]), **selection)
    assert med["q_survival"][0] == pytest.approx(total/2, rel=2e-5)
    assert med["D_survival"][0] == pytest.approx(total/2, rel=2e-5)
    q_med, D_med = m.compute_medians(2, np.array([N]), np.array([CADENCE]),
                                    full_integral=full, **selection)
    assert q_med[0] == pytest.approx(result["q_med"], rel=2e-5)
    assert D_med[0] == pytest.approx(result["D_med_cm"], rel=2e-5)


@pytest.mark.parametrize("random_start", [False, True], ids=["hard", "random-phase"])
def test_narrow_surviving_interval_below_fading_cap_is_not_missed(random_start):
    m = model()
    s_fade = 0.2
    # Phase III has magnitude decline 2.5*p*log10(t2/t1); solve its
    # two-visit fading constraint directly, independent of the engine caps.
    fading_time = CADENCE/np.expm1(np.log(10)*s_fade/(2.5*m.phys.p))
    cap = 1+np.sqrt(fading_time/m.derived.t_j_s)
    assert 2 < cap < m.derived.q_nr
    lower = cap-1e-7
    nodes, weights = _gauss()
    q = (lower+cap)/2+(cap-lower)/2*nodes
    peak = _peak_time(m, q)
    survival = np.clip((fading_time-peak)/CADENCE, 0, 1) if random_start else 1.0
    moment = (cap-lower)/2*np.dot(weights,
        q*_lightcurve(m, np.maximum(peak, 2*CADENCE))**(-m.lf.alpha-1)*survival)
    expected = _reference_rate(m)*moment/_angular_integral(m, -m.lf.alpha-1)
    actual = rate(m, q_min=lower, s_fade=s_fade, fade_random_start=random_start)
    assert expected > 0 and actual > 0
    assert actual == pytest.approx(expected, rel=2e-6)


def test_exact_angular_survival_matches_independent_integral():
    m = model(lf_alpha=-1.25)
    q_floor = 0.4
    q = np.array([0, 0.4, 0.9, m.derived.q_dec, 1.5, 2, 4, m.derived.q_nr])
    result = m.lf_distributions(2, N, CADENCE, full_integral=True,
        q_values=q, D_values_cm=np.array([]), q_min=q_floor)
    denominator = _angular_integral(m, -m.lf.alpha-1, lower=q_floor)
    expected = [_angular_integral(m, -m.lf.alpha-1, lower=max(x, q_floor))
                / denominator for x in q]
    assert np.allclose(result["q_survival"]/result["total_rate"], expected,
                       rtol=1e-7, atol=1e-12)


def test_adaptive_quadrature_precision_preserves_rate_and_medians(monkeypatch):
    m = model(lf_alpha=-1.25, win_from_peak=True)
    arguments = dict(full_integral=True, q_values=np.array([0, 1, 2, 4]),
        D_values_cm=np.array([0.1, 0.5, 1])*m.phys.D_euc_cm,
        q_min=0.6, D_min_cm=0.03*GPC_TO_CM, s_rise=0.3, s_fade=0.3)
    monkeypatch.setattr(powerlaw, "ANGULAR_RTOL", 1e-5)
    coarse = m.lf_distributions(2, N, CADENCE, **arguments)
    monkeypatch.setattr(powerlaw, "ANGULAR_RTOL", 1e-10)
    refined = m.lf_distributions(2, N, CADENCE, **arguments)
    assert coarse["total_rate"] == pytest.approx(refined["total_rate"], rel=0.01)
    assert coarse["q_med"] == pytest.approx(refined["q_med"], abs=0.01)
    assert coarse["D_med_cm"] == pytest.approx(refined["D_med_cm"], rel=0.01)


def test_approximate_population_is_distinct_from_full_population():
    m = model()
    arguments = dict(q_values=np.array([0, 1, 2, 4]), D_values_cm=np.array([]))
    approximate = m.lf_distributions(2, N, CADENCE, full_integral=False, **arguments)
    full = m.lf_distributions(2, N, CADENCE, full_integral=True, **arguments)
    assert 0 < approximate["total_rate"] < full["total_rate"]
    assert abs(approximate["q_med"]-full["q_med"]) > 0.01


@pytest.mark.parametrize("full", [False, True])
def test_filters_reduce_rates_and_empty_selection_is_zero(full):
    m = model()
    base = rate(m, full)
    for selection in [dict(q_min=1.4), dict(D_min_cm=GPC_TO_CM),
                      dict(s_fade=0.4), dict(s_rise=0.4),
                      dict(s_fade=0.4, s_rise=0.4)]:
        assert 0 <= rate(m, full, **selection) <= base*(1+1e-10)
    for selection in [dict(q_min=m.derived.q_nr+1),
                      dict(D_min_cm=m.phys.D_euc_cm)]:
        assert rate(m, full, **selection) == 0


@pytest.mark.parametrize("full", [False, True])
def test_bright_tail_remains_finite_under_extreme_rise_selection(full):
    # The corresponding linear luminosity threshold exceeds float range;
    # its small negative LF exponent nevertheless gives a finite rate.
    result = rate(model(lf_alpha=-1.01), full, cadence=1e8, s_rise=2)
    assert np.isfinite(result) and result > 0


@pytest.mark.parametrize("full,flags,selection", [
    pytest.param(False, {}, {}, id="plain-approximate"),
    pytest.param(True, {}, {}, id="plain-full"),
    pytest.param(False, {}, dict(s_rise=0.3, s_fade=0.3), id="joint-approximate"),
    pytest.param(True, {}, dict(s_rise=0.3, s_fade=0.3), id="joint-random"),
    pytest.param(True, {}, dict(s_rise=0.3, s_fade=0.3,
                 rise_random_start=False, fade_random_start=False), id="joint-hard"),
    pytest.param(True, dict(win_from_peak=True), dict(s_rise=0.3, s_fade=0.3),
                 id="peak-window-random"),
    pytest.param(True, dict(win_from_peak=True, win_i_minus_one=True),
                 dict(s_rise=0.3, s_fade=0.3, rise_random_start=False),
                 id="peak-window-optimistic-hard-rise"),
    pytest.param(True, dict(win_from_peak=True, win_i_minus_one=True),
                 dict(s_rise=0.3, s_fade=0.3, fade_random_start=False),
                 id="peak-window-optimistic-hard-fade"),
])
def test_finite_cutoff_reference_converges_at_fixed_differential_normalization(
        full, flags, selection):
    m = model(**flags)
    F_lim = float(m.F_lim_Jy(np.array([m.t_exp_s(N, CADENCE)]))[0])*1e-23
    L_dec = 4*np.pi*m.phys.D_euc_cm**2*F_lim
    low, high = L_dec*1e-12, L_dec*1e18
    exponent = m.lf.alpha+1
    finite_density = m.lf.A*((high/m.lf.L_ref)**exponent
                          -(low/m.lf.L_ref)**exponent)/exponent
    fiducial_L_dec = 4*np.pi*m.phys.D_euc_cm**2*m.derived.F_dec_Jy*1e-23
    one_day_conversion = m.L0_erg_s()/fiducial_L_dec
    bounded = make_finite_cutoff_rate_model(**BASE, **flags, lf_on=True,
        lf_alpha=m.lf.alpha, lf_log10_L_min=np.log10(low*one_day_conversion),
        lf_log10_L_max=np.log10(high*one_day_conversion),
        rho_grb_log10=np.log10(finite_density))
    fn = bounded.rate_log10_full_integral if full else bounded.rate_log10
    kwargs = dict(N_q=6000) if full else {}
    reference = float(10**np.asarray(fn(2, N, CADENCE, **kwargs, **selection)))
    assert reference/rate(m, full, **selection) == pytest.approx(1, rel=0.01)
    if full:
        refined = float(10**np.asarray(fn(2, N, CADENCE, N_q=12000, **selection)))
        assert refined == pytest.approx(reference, rel=0.002)
    # The exact missing tails are quantified, not hidden by renormalizing phi.
    if full and not selection:
        b = -m.lf.alpha-1
        integral = _angular_integral(m, b)
        faint = 2*b/3*(low/L_dec)**(1.5-b)*_angular_integral(m, 1.5)/integral
        bright = 2*(1.5-b)/3*(high/L_dec)**(-b)*(m.derived.q_nr**2/2)/integral
        assert faint+bright < 1e-4
        assert reference/rate(m, full) == pytest.approx(1-faint-bright, rel=0.01)
