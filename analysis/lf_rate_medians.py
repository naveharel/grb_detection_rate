"""LF-only search against the project's approximate ZTF rate/distance benchmarks.

Run from the repository root:
    .venv/Scripts/python.exe -B analysis/lf_rate_medians.py --workers 4

The engine is imported without modification. Results are diagnostic comparisons,
not a likelihood fit to a selection-matched observed sample. See the accompanying
plan and observations note. JSON retains all evaluated candidates; CSV is flat.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.ztf_validation import BASE_PARAMS, MODES, PHYS_NEW  # noqa: E402
from grb_detect.constants import DAY_S  # noqa: E402
from grb_detect.params import GPC_TO_CM  # noqa: E402
from grb_detect.detection_rate import LuminosityFunction  # noqa: E402
import standalone_bridge as bridge  # noqa: E402

COVERAGE = 0.35
RATE_FACTOR = 3.0
DISTANCE_TOLERANCE = 0.35
MODE_KEYS = {"public": "A (public 2-night)", "high_cadence": "B (high-cad 6/night)"}
TARGETS = {"public": {"rate": 2.0, "distance": 3.67},
           "high_cadence": {"rate": 2.5, "distance": 3.88}}
WINDOWS = {"conservative": False, "optimistic": True}
SELECTION = dict(q_min=0.0, D_min_cm=0.0, s_fade=0.3, s_rise=0.5,
                 s_mode="discrete", rise_random_start=True, fade_random_start=True)
DEFAULT_LF = (-2.0, 42.5, 45.5)


def build_state(lf, mode_name, window):
    mode = MODES[MODE_KEYS[mode_name]]
    params = dict(BASE_PARAMS, **PHYS_NEW)
    params.update({k: mode[k] for k in ("i_det", "f_live", "omega_srv_deg2")})
    params.update(win_tp=True, win_iminus1=WINDOWS[window], s_fade=0.3,
                  s_rise=0.5, lf_on=lf is not None)
    reference = LuminosityFunction(*lf) if lf is not None else None
    state = bridge._build_models(params, finite_cutoff_lf=reference)
    subday = state["optical_on"] and mode["t_cad_s"] < DAY_S
    model = state["model_night"] if subday else state["model_day"]
    night_factor = state["f_night"] if subday else 1.0
    return mode, model, night_factor


def rate_at(lf, mode_name, window, nq=500):
    mode, model, night_factor = build_state(lf, mode_name, window)
    log_rate = model.rate_log10_full_integral(
        mode["i_det"], np.array([mode["N_exp"]]), np.array([mode["t_cad_s"]]),
        N_q=nq, **SELECTION)
    raw_rate = float(10.0 ** log_rate.item()) * night_factor
    if not math.isfinite(raw_rate) or raw_rate < 0:
        raise ValueError(f"Invalid rate: {lf}, {mode_name}, {window}")
    return dict(raw_rate=raw_rate, rate=raw_rate * COVERAGE, N_q=nq,
                t_exp_s=float(model.t_exp_s(mode["N_exp"], mode["t_cad_s"])))


def cumulative(x, density):
    """CDF on the engine's native integration grid, with zero initial mass."""
    if not np.all(np.isfinite(density)) or np.min(density) < -1e-12:
        raise ValueError("Invalid marginal density")
    mass = np.r_[0.0, np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(x))]
    if mass[-1] <= 0:
        raise ValueError("Cannot normalize an empty marginal")
    return mass / mass[-1], float(mass[-1])


def distribution_at(lf, mode_name, window, nq=500, nd=400):
    stats = rate_at(lf, mode_name, window, nq)
    mode, model, night_factor = build_state(lf, mode_name, window)
    args = (mode["i_det"], mode["N_exp"], mode["t_cad_s"])
    q, dq = model.dR_dq_full_integral(*args, N_q=nq, **SELECTION)
    d, dd = model.dR_dD_full_integral(*args, N_q=nq, N_D=nd, **SELECTION)
    cq, q_mass = cumulative(q, dq)
    cd, d_mass = cumulative(d / GPC_TO_CM, dd * GPC_TO_CM)
    q_med = float(np.interp(0.5, cq, q))
    stats.update(
        N_D=nd, q_med=q_med, theta_med_rad=q_med * model.phys.theta_j_rad,
        theta_med_deg=math.degrees(q_med * model.phys.theta_j_rad),
        D_med_Gpc=float(np.interp(0.5, cd, d / GPC_TO_CM)),
        D_90_Gpc=float(np.interp(0.9, cd, d / GPC_TO_CM)),
        frac_q_lt_1=float(np.interp(1.0, q, cq)),
        frac_q_lt_1p5=float(np.interp(1.5, q, cq)),
        frac_q_gt_2=float(1 - np.interp(2.0, q, cq)),
        q_integral_raw=q_mass * night_factor,
        D_integral_raw=d_mass * night_factor,
        marginal_rel_error=abs(d_mass / q_mass - 1),
        rate_integral_rel_error=abs(q_mass * night_factor / stats["raw_rate"] - 1),
    )
    return stats


def score(modes, coverage=COVERAGE):
    values = []
    for name, stats in modes.items():
        target = TARGETS[name]
        values.append(abs(math.log(stats["raw_rate"] * coverage / target["rate"]))
                      / math.log(RATE_FACTOR))
        if "D_med_Gpc" not in stats:
            return None
        values.append(abs(stats["D_med_Gpc"] / target["distance"] - 1)
                      / DISTANCE_TOLERANCE)
    return max(values)


def rate_screen(modes):
    # A 1% margin prevents grid-level numerical differences excluding an edge fit.
    return all(t["rate"] / RATE_FACTOR / 1.01 <= modes[n]["rate"]
               <= t["rate"] * RATE_FACTOR * 1.01 for n, t in TARGETS.items())


def evaluate(lf, nq=500, nd=400, force_distributions=False):
    boundary = lf is not None and (lf[0] in (-3.5, 0) or lf[1] in (41, 46)
                                    or lf[2] in (42, 47.5))
    result = dict(lf=lf, on_search_boundary=boundary, windows={})
    for window in WINDOWS:
        modes = {name: rate_at(lf, name, window, nq) for name in MODE_KEYS}
        screened = rate_screen(modes)
        if screened or force_distributions:
            modes = {name: distribution_at(lf, name, window, nq, nd) for name in MODE_KEYS}
        s = score(modes)
        result["windows"][window] = dict(modes=modes, rate_screen=screened,
                                         score=s, passes=s is not None and s <= 1.0)
    result["classification"] = classify(result)
    return result


def classify(result):
    count = sum(w["passes"] for w in result["windows"].values())
    return ("fails", "passes_one_endpoint", "passes_both_endpoints")[count]


def coarse_grid():
    return [(float(a), float(lo), float(hi))
            for a, lo, hi in product(np.arange(-3.5, 0.01, 0.25),
                                    np.arange(41, 46.01, 0.5),
                                    np.arange(42, 47.51, 0.5)) if lo <= hi]


def refine_grid(results):
    seeds = set()
    shared = sorted((r for r in results if math.isfinite(joint_score(r))),
                    key=lambda r: (joint_score(r), r["lf"]))
    seeds.update(tuple(r["lf"]) for r in shared[:10])
    for window in WINDOWS:
        ranked = sorted((r for r in results if r["windows"][window]["score"] is not None),
                        key=lambda r: (r["windows"][window]["score"], r["lf"]))
        seeds.update(tuple(r["lf"]) for r in ranked[:10])
    refined = set()
    for a, lo, hi in seeds:
        for da, dl, dh in product(np.arange(-0.25, 0.251, 0.125),
                                  np.arange(-0.5, 0.501, 0.25),
                                  np.arange(-0.5, 0.501, 0.25)):
            point = (float(np.clip(a + da, -3.5, 0)),
                     float(np.clip(lo + dl, 41, 46)),
                     float(np.clip(hi + dh, 42, 47.5)))
            if point[1] <= point[2]:
                refined.add(point)
    return sorted(refined - {tuple(r["lf"]) for r in results})


def run_grid(points, workers, stage):
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, result in enumerate(pool.map(evaluate, points, chunksize=4), 1):
            result["stage"] = stage
            results.append(result)
            if n % 100 == 0 or n == len(points):
                print(f"{stage}: {n}/{len(points)} ({time.perf_counter()-start:.1f}s)", flush=True)
    return results


def joint_score(result):
    scores = [w["score"] for w in result["windows"].values()]
    return max(scores) if all(s is not None for s in scores) else math.inf


def choose_highlights(results):
    chosen = {"single_default": None, "lf_default": DEFAULT_LF}
    complete = [r for r in results if math.isfinite(joint_score(r))]
    if complete:
        chosen["best_shared"] = min(complete, key=joint_score)["lf"]
    for window in WINDOWS:
        eligible = [r for r in results if r["windows"][window]["score"] is not None]
        if eligible:
            chosen[f"best_{window}"] = min(eligible, key=lambda r: r["windows"][window]["score"])["lf"]
        misses = [r for r in eligible if r["windows"][window]["score"] > 1]
        if misses:
            chosen[f"near_miss_{window}"] = min(misses, key=lambda r: r["windows"][window]["score"])["lf"]
    passed = [r for r in complete if r["classification"] == "passes_both_endpoints"]
    if passed:
        chosen["most_onaxis_shared_fit"] = max(
            passed, key=lambda r: min(m["frac_q_lt_1"] for w in r["windows"].values()
                                     for m in w["modes"].values()))["lf"]
    narrow = [r for r in complete if r["lf"][1] == r["lf"][2]]
    if narrow:
        chosen["best_collapsed_lf"] = min(narrow, key=joint_score)["lf"]
    return chosen


def convergence(low, high):
    checks = []
    for window in WINDOWS:
        for name in MODE_KEYS:
            a = low["windows"][window]["modes"][name]
            b = high["windows"][window]["modes"][name]
            check = dict(window=window, mode=name,
                         rate_change=abs(a["rate"] / b["rate"] - 1),
                         distance_change=abs(a["D_med_Gpc"] / b["D_med_Gpc"] - 1),
                         q_change=abs(a["q_med"] - b["q_med"]),
                         marginal_rel_error=b["marginal_rel_error"],
                         rate_integral_rel_error=b["rate_integral_rel_error"])
            check["passes"] = (check["rate_change"] < 0.01 and check["distance_change"] < 0.01
                               and check["q_change"] < 0.01
                               and check["marginal_rel_error"] < 0.01
                               and check["rate_integral_rel_error"] < 0.01)
            checks.append(check)
    return checks


def verify_highlights(chosen):
    cache = {}
    output = {}
    for label, lf in chosen.items():
        key = tuple(lf) if lf is not None else None
        if key not in cache:
            nq, nd = 1500, 1000
            low = evaluate(key, nq, nd, True)
            for attempt in range(4):
                high = evaluate(key, 2*nq, 2*nd, True)
                checks = convergence(low, high)
                if all(c["passes"] for c in checks):
                    break
                low, nq, nd = high, 2*nq, 2*nd
            else:
                raise RuntimeError(f"Convergence failed for {label}: {checks}")
            coverage = {}
            for eff in (0.2, 0.35, 0.5):
                coverage[str(eff)] = {
                    window: dict(score=score(w["modes"], eff),
                                 passes=score(w["modes"], eff) <= 1,
                                 rates={name: m["raw_rate"]*eff for name, m in w["modes"].items()})
                    for window, w in high["windows"].items()}
            cache[key] = dict(result=high, checks=checks, coverage=coverage)
        output[label] = cache[key]
        print(f"verified {label}: {lf}, {output[label]['result']['classification']}", flush=True)
    return output


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, results):
    rows = []
    for r in results:
        a, lo, hi = r["lf"] if r["lf"] is not None else (None, None, None)
        for window, w in r["windows"].items():
            for name, stats in w["modes"].items():
                rows.append(dict(stage=r.get("stage", "verified"), lf_on=r["lf"] is not None,
                                 label=r.get("label", ""),
                                 alpha=a, log10_L_min=lo, log10_L_max=hi,
                                 on_search_boundary=r["on_search_boundary"],
                                 classification=r["classification"], window=window, mode=name,
                                 score=w["score"], passes=w["passes"], **stats))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def provenance():
    files = sorted((ROOT / "grb_detect").glob("*.py"))
    files += [ROOT / "standalone_bridge.py", Path(__file__), ROOT / "analysis/ztf_validation.py"]
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        python=sys.version, numpy=np.__version__, physics=PHYS_NEW,
        survey_base=BASE_PARAMS, survey_modes=MODES, selection=SELECTION,
        windows=WINDOWS, coverage=COVERAGE, targets=TARGETS,
        rate_factor=RATE_FACTOR, distance_tolerance=DISTANCE_TOLERANCE,
        search_bounds=dict(alpha=[-3.5, 0], log10_L_min=[41, 46], log10_L_max=[42, 47.5]),
        distance_yardstick="Inherited comoving proxy medians, not a cosmological prediction",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/lf_rate_medians_results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    write_json(out / "metadata.json", meta)
    coarse = run_grid(coarse_grid(), args.workers, "coarse")
    write_json(out / "coarse.json", coarse)
    refined = run_grid(refine_grid(coarse), args.workers, "refined")
    write_json(out / "refined.json", refined)
    results = coarse + refined
    write_csv(out / "scan.csv", results)
    chosen = choose_highlights(results)
    verified = verify_highlights(chosen)
    write_json(out / "highlights.json", verified)
    write_csv(out / "highlights.csv", [dict(v["result"], label=k) for k, v in verified.items()])
    counts = {stage: {c: sum(r["classification"] == c for r in rs)
                     for c in ("passes_both_endpoints", "passes_one_endpoint", "fails")}
              for stage, rs in (("coarse", coarse), ("refined", refined))}
    summary = dict(screening_counts=counts, highlighted_parameters=chosen,
                   all_highlights_converged=all(c["passes"] for v in verified.values() for c in v["checks"]))
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
