"""Faithfulness tests for the figure subsystem's F_dec override.

`figures.figlib.overrides.set_F_dec` lets figure code vary the flux normalization
F_nu,dec without touching the read-only engine, by rescaling the cached derived flux
scales (F_dec, F_j, F_nr) on a *copy* of the model.

These tests pin the physics invariant the override relies on: because the detection
rate depends on flux only through the ratio F_lim / F_dec (and the comparison
F_lim < F_j), scaling F_dec by k is exactly equivalent to scaling the limiting-flux
anchor A_log (= log10 F_lim_ref) by -log10(k). If the engine ever changes so that a
geometric/temporal scale depends on F_dec, these tests break — which is the point.

Run with::

    .venv/Scripts/python -m pytest tests/test_fdec_override.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grb_detect.core import make_rate_model
from grb_detect.params import SurveyDesignParams
from grb_detect.constants import DEG2_TO_SR

from figures.figlib import models, overrides, presets
from figures.figlib.compute import q_median_at, rate_at

# A representative ZTF-like strategy point. N_exp MUST come from the model
# (n_exp_max), not a hardcoded 27500/47: the ZTF point sits exactly on the
# N_exp_max boundary, and a raw ratio can exceed it by a float epsilon and become
# unphysical (NaN). ztf_strategy_point derives N from the model, matching the app.
_DESIGN = SurveyDesignParams(omega_survey_max_sr=27500.0 * DEG2_TO_SR)


def _ztf_model():
    return make_rate_model(
        A_log=-4.68, f_live=0.2, t_overhead_s=15.0,
        omega_exp_deg2=47.0, design=_DESIGN,
    )


def _rebuilt_with_A(a_log):
    return make_rate_model(
        A_log=a_log, f_live=0.2, t_overhead_s=15.0,
        omega_exp_deg2=47.0, design=_DESIGN,
    )


_N_EXP, _T_CAD_S, _I_DET = models.ztf_strategy_point(_ztf_model(), i_det=presets.ZTF.i_det)


# k spans cadence-limited, flux-limited and the q_nr-capped regimes.
_K_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e5]


@pytest.mark.parametrize("k", _K_VALUES)
def test_override_matches_native_A_log_rebuild_qmed(k):
    """set_F_dec(model, k*F) reproduces a native A_log = A0 - log10(k) rebuild (q_med)."""
    model = _ztf_model()
    F_fid = model.derived.F_dec_Jy

    q_override = q_median_at(overrides.set_F_dec(model, k * F_fid), _N_EXP, _T_CAD_S, _I_DET)
    q_native = q_median_at(_rebuilt_with_A(-4.68 - math.log10(k)), _N_EXP, _T_CAD_S, _I_DET)

    assert q_override == pytest.approx(q_native, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("k", _K_VALUES)
def test_override_matches_native_A_log_rebuild_rate(k):
    """Same equivalence for the detection rate R_det."""
    model = _ztf_model()
    F_fid = model.derived.F_dec_Jy

    r_override = rate_at(overrides.set_F_dec(model, k * F_fid), _N_EXP, _T_CAD_S, _I_DET)
    r_native = rate_at(_rebuilt_with_A(-4.68 - math.log10(k)), _N_EXP, _T_CAD_S, _I_DET)

    assert r_override == pytest.approx(r_native, rel=1e-10)


def test_set_F_dec_identity_is_noop():
    """Overriding to the current F_dec leaves q_med bit-for-bit identical."""
    model = _ztf_model()
    same = overrides.set_F_dec(model, model.derived.F_dec_Jy)
    assert q_median_at(same, _N_EXP, _T_CAD_S, _I_DET) == q_median_at(model, _N_EXP, _T_CAD_S, _I_DET)


def test_override_does_not_mutate_original():
    """The lru-cached original model is untouched by an override (copy-on-write)."""
    model = _ztf_model()
    F_before = model.derived.F_dec_Jy
    _ = overrides.scale_F_dec(model, 100.0)
    assert model.derived.F_dec_Jy == F_before
    # Same cached object returned by a fresh build is also unchanged.
    assert _ztf_model().derived.F_dec_Jy == F_before


@pytest.mark.parametrize("k", [1e-2, 1.0, 50.0])
def test_flux_scale_coupling_preserved(k):
    """F_j/F_dec and F_nr/F_dec are invariant under the override (only the anchor moves)."""
    model = _ztf_model()
    d0 = model.derived
    d1 = overrides.scale_F_dec(model, k).derived
    assert d1.F_dec_Jy == pytest.approx(k * d0.F_dec_Jy, rel=1e-12)
    assert d1.F_j_Jy / d1.F_dec_Jy == pytest.approx(d0.F_j_Jy / d0.F_dec_Jy, rel=1e-12)
    assert d1.F_nr_Jy / d1.F_dec_Jy == pytest.approx(d0.F_nr_Jy / d0.F_dec_Jy, rel=1e-12)
    # Geometric/temporal scales must NOT change.
    assert d1.q_dec == d0.q_dec and d1.q_j == d0.q_j and d1.q_nr == d0.q_nr
    assert d1.t_dec_s == d0.t_dec_s and d1.t_j_s == d0.t_j_s
