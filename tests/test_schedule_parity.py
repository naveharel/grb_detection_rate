"""Snapshot parity for the schedule-cadence refactor.

Replays the params stored by tests/make_parity_snapshots.py (captured on the
pre-change code at the schedule-cadence branch point) through the current
standalone_bridge.compute_all and asserts the payloads match.

The value-preserving refactor phases (unified budget, single-model collapse,
f_night internalization) must keep every payload value unchanged up to
floating-point reassociation noise; the schedule physics itself only activates
at N_v >= 2, which no stored config uses.

Tolerances: rate/median/geometry values to rtol 1e-9 (ulp-level reassociation
from f_live = f_eff*f_night round-trips is ~1e-16); optimizer argmax location
(N_opt, t_cad_opt_*) to rtol 1e-3, since an ulp-level surface change may move
the winning grid node within the refined zoom cell; the optimum's height
(log10R_opt) stays at 1e-9.

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
    for key, old_val in old_payload.items():
        rtol = RTOL_LOOSE if key in LOOSE_KEYS else RTOL_TIGHT
        _assert_match(key, old_val, payload[key], rtol)


def test_snapshots_exist():
    assert len(SNAPSHOTS) >= 10, (
        "parity snapshots missing — run tests/make_parity_snapshots.py "
        "on the pre-change code")
