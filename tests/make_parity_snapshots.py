"""Generate pre-schedule-model parity snapshots of standalone_bridge.compute_all.

Run from the repo root on the pre-change code (the schedule-cadence branch
point, main @ e5dd9ff) to (re)create tests/data/parity_*.json.gz:

    .venv/Scripts/python tests/make_parity_snapshots.py

tests/test_schedule_parity.py replays the stored params through compute_all
after the value-preserving refactor phases and asserts the payloads match.
Regenerate only if the pre-change baseline itself is being redefined.

Grid resolutions are shrunk (module-level monkeypatch, mirrored by the parity
test) — identical code paths, a fraction of the runtime and file size.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import standalone_bridge as sb  # noqa: E402

GPC_TO_CM = 3.085677581491367e27

# Shrunken surface grids (same code paths; parity test applies the same patch).
SNAP_GRIDS = dict(NX_REGIME=48, NY_REGIME=60, NX_DEFAULT=40, NY_DEFAULT=50)

# App boot defaults (web slider defaults; optical switch boots unchecked).
BASE = {
    "i_det": 2, "A_log": -4.68, "f_live": 0.2, "t_overhead_s": 0.0,
    "omega_exp_deg2": 47.0, "omega_srv_deg2": 41253.0, "t_night_h": 10.0,
    "p": 2.2, "nu_log10": 14.7, "E_kiso_log10": 53.0, "n0_log10": 0.0,
    "epsilon_e_log10": -1.0, "epsilon_B_log10": -4.0, "theta_j_rad": 0.1,
    "gamma0_log10": 2.5, "D_euc_gpc": 4.55, "rho_grb_log10": 2.415,
    "optical_survey": False, "color_regimes": False, "full_integral": False,
    "qmin": 0.0, "Dmin_cm": 0.0, "s_fade": 0.0, "s_rise": 0.0,
    "rise_random_start": True, "fade_random_start": True,
    "toh_approx": False, "win_iminus1": False, "win_tp": False,
    # Slice-slider positions (app defaults), pinned explicitly.
    "nslice_tfix_log": 4.936514, "tslice_nfix_log": 2.0,
    "qdview_nfix_log": 2.0, "qdview_tfix_log": 4.936514,
}

CONFIGS: dict[str, dict] = {
    # Boot state and the two master modes of the surface.
    "boot_nonoptical":  {},
    "optical_default":  {"optical_survey": True},
    "optical_regimes":  {"optical_survey": True, "color_regimes": True},
    # Overhead handling, both treatments.
    "optical_toh_approx": {"optical_survey": True, "t_overhead_s": 15.0,
                           "toh_approx": True},
    "optical_toh_exact":  {"optical_survey": True, "t_overhead_s": 15.0},
    # Exact rate mode, plain and with every cut/window branch engaged.
    "optical_exact": {"optical_survey": True, "full_integral": True},
    "optical_exact_cuts": {
        "optical_survey": True, "full_integral": True,
        "s_fade": 0.3, "s_rise": 0.5, "qmin": 1.03, "Dmin_cm": 0.5 * GPC_TO_CM,
        "win_iminus1": True, "win_tp": True, "fade_random_start": False,
    },
    # The three UI presets (preset fields only; the UI's old dynamic t_night
    # floor clamps rubin's t_night to 17 h, mirrored here).
    "preset_ztf_public": {"optical_survey": True, "f_live": 0.08,
                          "t_overhead_s": 15.0},
    "preset_ztf_hc":     {"optical_survey": True, "i_det": 6, "f_live": 0.17,
                          "t_overhead_s": 15.0},
    "preset_rubin":      {"optical_survey": True, "f_live": 0.7, "A_log": -7.0,
                          "omega_exp_deg2": 9.6, "t_overhead_s": 30.0,
                          "t_night_h": 17.0},
}


def apply_snapshot_grids(module) -> None:
    for key, val in SNAP_GRIDS.items():
        setattr(module, key, val)


def main() -> None:
    apply_snapshot_grids(sb)
    outdir = pathlib.Path(__file__).parent / "data"
    outdir.mkdir(exist_ok=True)
    for name, overrides in CONFIGS.items():
        params = {**BASE, **overrides}
        t0 = time.perf_counter()
        payload = sb.compute_all(params)
        dt = time.perf_counter() - t0
        if payload.get("error") is not None:
            raise RuntimeError(f"{name}: compute_all failed:\n{payload['error']}")
        path = outdir / f"parity_{name}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"params": params, "payload": payload}, f)
        print(f"{name:22s} {dt:7.1f}s  -> {path.name}  "
              f"({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
