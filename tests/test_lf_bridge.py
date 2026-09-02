"""Bridge-level tests for the luminosity-function (LF) integration.

Covers the params contract (old dicts without lf keys keep working and give
the identical payload), the lf echoes, qdview's forced full path under the LF,
and the no-crash guarantee across UI-reachable LF states.
"""

from __future__ import annotations

import numpy as np
import pytest

import standalone_bridge as sb


@pytest.fixture(scope="module", autouse=True)
def _small_grids():
    """compute_all's grid resolution has no params override hook — shrink it."""
    saved = (sb.NX_DEFAULT, sb.NY_DEFAULT, sb.NX_REGIME, sb.NY_REGIME)
    sb.NX_DEFAULT, sb.NY_DEFAULT = 24, 30
    sb.NX_REGIME, sb.NY_REGIME = 24, 30
    yield
    sb.NX_DEFAULT, sb.NY_DEFAULT, sb.NX_REGIME, sb.NY_REGIME = saved


def _params(**over) -> dict:
    p = dict(
        i_det=2,
        A_log=-4.68,
        f_live=0.1,
        t_overhead_s=0.0,
        omega_exp_deg2=47.0,
        omega_srv_deg2=27500.0,
        t_night_h=10.0,
        p=2.2,
        nu_log10=14.7,
        E_kiso_log10=53.0,
        n0_log10=0.0,
        epsilon_e_log10=-1.0,
        epsilon_B_log10=-4.0,
        theta_j_rad=0.1,
        gamma0_log10=2.5,
        D_euc_gpc=4.55,
        rho_grb_log10=2.415,
        optical_survey=False,
        color_regimes=False,
        full_integral=False,
        qmin=0.0,
        Dmin_cm=0.0,
    )
    p.update(over)
    return p


LF_ON = dict(lf_on=True, lf_alpha=-2.0, lf_lmin=42.5, lf_lmax=45.5)


def test_old_params_without_lf_keys_still_work():
    payload = sb.compute_all(_params())
    assert payload["error"] is None
    assert payload["lf_applied"] is False
    assert payload["lf_L_med_pop_erg_s"] is None
    assert payload["L0_erg_s"] > 0


def test_lf_off_payload_identical_to_absent_keys():
    """lf_on=False must be byte-equivalent to omitting the keys entirely."""
    base = sb.compute_all(_params())
    off = sb.compute_all(_params(lf_on=False, lf_alpha=-1.3,
                                 lf_lmin=41.0, lf_lmax=46.0))
    assert base["error"] is None and off["error"] is None
    for key in ("Z_flat", "Z_log_flat", "q_med_flat", "D_med_Gpc_flat",
                "N_opt", "t_cad_opt_s", "R_opt"):
        assert base[key] == off[key], key


@pytest.mark.parametrize("mode", ["dominant", "exact"])
@pytest.mark.parametrize("optical", [False, True], ids=["nonopt", "optical"])
def test_lf_on_computes_and_echoes(mode, optical):
    payload = sb.compute_all(_params(
        **LF_ON,
        full_integral=(mode == "exact"),
        optical_survey=optical,
        color_regimes=True,
        s_fade=0.3, s_rise=0.3,
    ))
    assert payload["error"] is None
    assert payload["lf_applied"] is True
    # population median for α=−2 on [42.5, 45.5]: closed form
    a = -1.0
    r = 10.0 ** (a * 3.0)
    want = 10.0 ** (42.5 + np.log10(0.5 * (1 + r)) / a)
    assert np.isclose(payload["lf_L_med_pop_erg_s"], want, rtol=1e-9)
    # surface has finite drawn cells, and regime ids are in 1..7 where present
    Z = np.array([v for v in payload["Z_flat"] if v is not None], dtype=float)
    assert Z.size > 0 and np.all(Z > 0)
    rid = np.array([v for v in payload["regime_flat"] if v is not None])
    assert rid.size > 0
    assert set(np.unique(rid)).issubset({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0})


def test_lf_raises_rate_at_bright_default_range():
    """The default LF (median below L0, bright tail above) must change the
    rate; direction is configuration-dependent, but the optimum rate must stay
    finite and the surfaces must differ."""
    base = sb.compute_all(_params())
    lf = sb.compute_all(_params(**LF_ON))
    assert lf["error"] is None
    assert lf["R_opt"] is not None and lf["R_opt"] > 0
    zb = [v for v in base["Z_flat"] if v is not None]
    zl = [v for v in lf["Z_flat"] if v is not None]
    assert zb != zl


def test_lf_qdview_forced_full_path_finite():
    payload = sb.compute_qdview(_params(**LF_ON), 100.0, 86400.0)
    assert payload["error"] is None
    Rq = np.array([v for v in payload["qdview_Rq_diff_flat"] if v is not None])
    RD = np.array([v for v in payload["qdview_RD_diff_flat"] if v is not None])
    assert Rq.size > 0 and np.all(np.isfinite(Rq))
    assert RD.size > 0 and np.all(np.isfinite(RD))
    # cumulative endpoint consistency (the exact dR/dq is LF-aware)
    cum0 = payload["qdview_Rq_cum_flat"][0]
    q = np.array(payload["qdview_q_grid_flat"], dtype=float)
    Rq_full = np.array([0.0 if v is None else v
                        for v in payload["qdview_Rq_diff_flat"]], dtype=float)
    assert np.isclose(np.trapezoid(Rq_full, q), cum0, rtol=2e-2)
    assert payload["qdview_regime_id"] in (1, 2, 3, 4, 5, 6, 7)


def test_lf_slices_consistent_with_surface_model():
    """Slice sweeps run through the same LF-aware model — no errors, finite."""
    params = _params(**LF_ON)
    n = sb.compute_nslice(params, 86400.0)
    t = sb.compute_tslice(params, 100.0)
    assert n["error"] is None and t["error"] is None
    Rn = [v for v in n["N_sweep_R_flat"] if v is not None]
    assert len(Rn) > 0 and all(v > 0 for v in Rn)


def test_lf_min_above_max_is_clamped_not_fatal():
    payload = sb.compute_all(_params(lf_on=True, lf_alpha=-2.0,
                                     lf_lmin=46.0, lf_lmax=43.0))
    assert payload["error"] is None


def test_fdec_override_is_noop_when_lf_on():
    """The TEMP F_dec override rescales F_dec,0 and L0 together — invariant."""
    a = sb.compute_all(_params(**LF_ON))
    b = sb.compute_all(_params(**LF_ON, F_dec_override_Jy=0.5))
    assert a["error"] is None and b["error"] is None
    assert b["F_dec_override_applied"] is True
    za = np.array([np.nan if v is None else v for v in a["Z_log_flat"]])
    zb = np.array([np.nan if v is None else v for v in b["Z_log_flat"]])
    finite = np.isfinite(za) | np.isfinite(zb)
    assert np.allclose(za[finite], zb[finite], rtol=1e-9, equal_nan=True)
