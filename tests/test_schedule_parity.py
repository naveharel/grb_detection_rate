"""Snapshot parity for the schedule-cadence refactor.

Replays the params stored by tests/make_parity_snapshots.py (captured on the
pre-change code at the schedule-cadence branch point) through the current
standalone_bridge.compute_all and asserts the payloads match.

Non-optical configs: FULL-payload parity — non-optical mode is untouched by
the schedule refactor, so every value must match up to floating-point noise.

Optical configs: the sub-day cadence region was removed by design (optical
mode is now integer-day only; intra-night sampling lives in the N_v/Δt_v
schedule), so the surface grid, optimizer domain, gap keys, sub-day slice
content and the relocated ZTF-HC marker intentionally changed.  Parity is
asserted on the discrete-region whitelist instead: the dense day-line overlay,
the discrete t-slice, the N-slice and qd-view at their 1-day default
positions, the ZTF-public marker, and the domain-independent scalars — all of
which must be bit-preserved at the default N_v = 1.

Tolerances: values to rtol 1e-9 (ulp-level reassociation); optimizer argmax
location (N_opt, t_cad_opt_*) to rtol 1e-3 where compared (non-optical only).

Run with::

    .venv/Scripts/python -m pytest tests/test_schedule_parity.py -v
"""
from __future__ import annotations

import gzip
import json
import math
import pathlib

import pytest

import standalone_bridge as sb
from make_parity_snapshots import SNAP_GRIDS

DATA_DIR = pathlib.Path(__file__).parent / "data"
SNAPSHOTS = sorted(DATA_DIR.glob("parity_*.json.gz"))

# Keys located by an argmax grid search: position may shift within the zoom
# cell under ulp-level surface changes.
LOOSE_KEYS = {"N_opt", "t_cad_opt_h", "t_cad_opt_s", "t_exp_opt_s",
              "q_med_opt", "D_med_Gpc_opt"}
RTOL_TIGHT = 1e-9
RTOL_LOOSE = 1e-3

# Intended semantic change (all configs): the ZTF high-cadence marker moved
# from the pre-schedule sub-night hack (t_cad = 0.98·t_night/6) to its true
# schedule position (t_cad = 1 day, 6 visits/night), so its coordinates and
# derived values differ by design.  N_ztf_hc (the footprint) is unchanged and
# stays compared.
HC_MARKER_KEYS = {"t_cad_ztf_hc_h", "t_cad_ztf_hc_s", "R_ztf_hc",
                  "t_exp_ztf_hc_s", "q_med_ztf_hc", "D_med_Gpc_ztf_hc"}

# Optical configs: keys whose pre-change values live entirely on the
# integer-day cadence domain and must therefore be preserved by the
# sub-day-region removal (see module docstring).  qdview_* keys are matched
# by prefix (their default position is the 1-day cadence).
OPTICAL_COMPARE_KEYS = {
    # discrete-day overlay — the dense day-domain parity set
    "day_line_shape", "day_line_t_cad_days", "day_line_N_flat",
    "day_line_R_flat", "day_line_regime_flat", "day_line_t_exp_flat",
    "day_line_q_med_flat", "day_line_D_med_Gpc_flat",
    # t-slice, discrete region
    "t_disc_h_flat", "t_disc_R_flat", "t_disc_t_exp_flat",
    "t_disc_q_med_flat", "t_disc_D_med_Gpc_flat", "t_disc_regime_flat",
    "N_fix",
    # N-slice at its 1-day default position
    "N_sweep_flat", "N_sweep_R_flat", "N_sweep_t_exp_flat",
    "N_sweep_q_med_flat", "N_sweep_D_med_Gpc_flat", "N_sweep_regime_flat",
    "t_cad_fix_s", "t_cad_fix_h",
    # ZTF public marker (2-day cadence)
    "N_ztf", "t_cad_ztf_h", "t_cad_ztf_s", "R_ztf", "t_exp_ztf_s",
    "q_med_ztf", "D_med_Gpc_ztf",
    # domain-independent scalars
    "R_int_yr", "R_toward_day", "t_dec_s", "F_nu_tdec_Jy", "N_exp_max",
    "F_dec_override_applied", "error",
}


def _close(a: float, b: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def _assert_match(key: str, old, new, rtol: float, path: str = "") -> None:
    where = f"{key}{path}"
    if old is None or new is None:
        assert old is None and new is None, (
            f"{where}: None mismatch (old={old!r}, new={new!r})")
        return
    if isinstance(old, bool) or isinstance(new, bool):
        assert bool(old) == bool(new), f"{where}: {old!r} != {new!r}"
        return
    if isinstance(old, str) or isinstance(new, str):
        assert old == new, f"{where}: {old!r} != {new!r}"
        return
    if isinstance(old, (list, tuple)) or isinstance(new, (list, tuple)):
        assert isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)), (
            f"{where}: type mismatch ({type(old).__name__} vs {type(new).__name__})")
        assert len(old) == len(new), (
            f"{where}: length {len(old)} != {len(new)}")
        n_bad = 0
        first_bad = None
        for i, (a, b) in enumerate(zip(old, new)):
            try:
                _assert_match(key, a, b, rtol, path=f"{path}[{i}]")
            except AssertionError as exc:
                n_bad += 1
                if first_bad is None:
                    first_bad = str(exc)
        assert n_bad == 0, f"{where}: {n_bad}/{len(old)} entries differ; first: {first_bad}"
        return
    assert _close(float(old), float(new), rtol), (
        f"{where}: {old!r} != {new!r} (rtol={rtol})")


@pytest.fixture(scope="module", autouse=True)
def _snapshot_grids():
    """Run compute_all on the same shrunken grids the snapshots used."""
    saved = {k: getattr(sb, k) for k in SNAP_GRIDS}
    for k, v in SNAP_GRIDS.items():
        setattr(sb, k, v)
    yield
    for k, v in saved.items():
        setattr(sb, k, v)


@pytest.mark.parametrize(
    "snap_path", SNAPSHOTS, ids=[p.stem.replace("parity_", "") for p in SNAPSHOTS])
def test_payload_parity(snap_path):
    with gzip.open(snap_path, "rt", encoding="utf-8") as f:
        snap = json.load(f)
    payload = sb.compute_all(snap["params"])
    assert payload.get("error") is None, payload.get("error")

    old_payload = snap["payload"]
    assert set(old_payload) <= set(payload), (
        f"missing payload keys: {sorted(set(old_payload) - set(payload))}")
    optical = bool(snap["params"].get("optical_survey", False))
    n_compared = 0
    for key, old_val in old_payload.items():
        if key in HC_MARKER_KEYS:
            continue
        if optical and key not in OPTICAL_COMPARE_KEYS and not key.startswith("qdview_"):
            continue
        rtol = RTOL_LOOSE if key in LOOSE_KEYS else RTOL_TIGHT
        _assert_match(key, old_val, payload[key], rtol)
        n_compared += 1
    assert n_compared >= (20 if optical else 50), (
        f"suspiciously few keys compared ({n_compared}) — whitelist drift?")


def test_snapshots_exist():
    assert len(SNAPSHOTS) >= 10, (
        "parity snapshots missing — run tests/make_parity_snapshots.py "
        "on the pre-change code")
