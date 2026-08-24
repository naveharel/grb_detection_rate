"""Snapshot regression parity for standalone_bridge.compute_all.

Replays the params stored by tests/make_parity_snapshots.py through the
current compute_all and asserts FULL-payload parity for every config — the
snapshots are a same-code regression baseline (re-baselined 2026-08-24 after
the reviewed schedule-era physics changes; see make_parity_snapshots.py).
Any unexplained failure here means a behavior change leaked in; regenerate
the snapshots only after deliberate, reviewed changes.

Tolerances: values to rtol 1e-9 (ulp-level reassociation); optimizer argmax
location (N_opt, t_cad_opt_*) to rtol 1e-3.

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
    assert set(old_payload) == set(payload), (
        f"payload key-set changed: missing "
        f"{sorted(set(old_payload) - set(payload))}, new "
        f"{sorted(set(payload) - set(old_payload))} — regenerate the "
        "snapshots if this change is deliberate")
    for key, old_val in old_payload.items():
        rtol = RTOL_LOOSE if key in LOOSE_KEYS else RTOL_TIGHT
        _assert_match(key, old_val, payload[key], rtol)


def test_snapshots_exist():
    assert len(SNAPSHOTS) >= 10, (
        "parity snapshots missing — run tests/make_parity_snapshots.py")


def test_budget_keys_are_mode_specific():
    """Each survey mode reads its own budget parameter — f_eff (usable
    night-window fraction) in optical mode, f_live (wall-clock live fraction)
    in non-optical mode.  They are different physical quantities; the mode
    toggle must never silently reinterpret one number as the other (the
    boot-parity bug of 2026-08-23)."""
    from make_parity_snapshots import BASE
    both = {**BASE, "f_eff": 0.48, "f_live": 0.2}

    st_opt = sb._build_models({**both, "optical_survey": True})
    f_night = float(st_opt["f_night"])
    assert st_opt["model"].instrument.f_live == pytest.approx(
        0.48 * f_night, rel=1e-9), "optical mode must use the f_eff key"

    st_non = sb._build_models({**both, "optical_survey": False})
    assert st_non["model"].instrument.f_live == pytest.approx(0.2, rel=1e-12), (
        "non-optical mode must use the wall-clock f_live key")

    # Legacy single-key callers stay bit-exact: f_live-only (pre-schedule)...
    flive_only = {k: v for k, v in BASE.items() if k != "f_eff"}
    st_leg = sb._build_models({**flive_only, "optical_survey": True})
    assert st_leg["model"].instrument.f_live == pytest.approx(
        BASE["f_live"], rel=1e-12)
    # ...and f_eff-only (early-schedule era) still resolves in both modes.
    feff_only = {k: v for k, v in BASE.items() if k != "f_live"}
    feff_only["f_eff"] = 0.37
    st_fo = sb._build_models({**feff_only, "optical_survey": False})
    assert st_fo["model"].instrument.f_live == pytest.approx(0.37, rel=1e-12)
