"""Rebuild the report figures using the project publication style."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figures.figlib import axes, io, style
from analysis.simplified_luminosity_function import theory as th
from analysis.simplified_luminosity_function import ztf

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT = RESULTS / "figures"
ALPHAS = [-2.4, -2., -1.75, -1.25, -1.05]
LINES = ["-", "--", "-.", ":", (0, (6, 2, 1, 2, 1, 2))]


def label(ax, text):
    ax.set_title(text, loc="left", fontsize=9, pad=7)


def contribution_distributions():
    fig, aa = plt.subplots(2, 2, figsize=(style.COL_DOUBLE, 5.8), layout="constrained")
    x = np.logspace(-6, 6, 900)
    q = np.unique(np.r_[np.geomspace(.035, th.DEFAULT_SHAPE.q_nr, 1000),
                        th.DEFAULT_SHAPE.q_dec, 2])
    d = np.linspace(0, 1, 800)
    qc = np.linspace(0, 4, 300)
    for i, a in enumerate(ALPHAS):
        kw = dict(color=style.CATEGORICAL[i], linestyle=LINES[i], label=rf"$\alpha={a:g}$")
        aa[0, 0].plot(x, th.conditional_log_luminosity_density(x, a), **kw)
        aa[0, 1].plot(q, q*th.moment_density(q, a)/th.angular_moment(a), **kw)
        aa[1, 0].plot(d, th.distance_cdf(d, a), **kw)
        aa[1, 1].plot(qc, [th.angular_cdf(t, a) for t in qc], **kw)
    aa[0, 0].set(xscale="log", xlabel=r"$L/L_{\mathrm{wall}}(q)$",
                 ylabel=r"$p(\ln L\mid q)$", ylim=(0, .40))
    aa[0, 1].set(xscale="log", xlabel=r"$q$", ylabel=r"$p(\ln q)$")
    aa[1, 0].set(xlabel=r"$D/D_{\mathrm{Euc}}$", ylabel=r"$P(D'<D)$", ylim=(0, 1.03))
    aa[1, 1].set(xlabel=r"$q$", ylabel=r"$P(q'<q)$", ylim=(0, 1.03))
    for ax, title in zip(aa.flat, ["(a) Local luminosity threshold", "(b) Continuous angular contribution",
                                  "(c) Distance distribution", "(d) Continuous angular CDF"]):
        label(ax, title)
    axes.place_legend(aa[1, 0], loc="lower right", frameon=True, framealpha=.9, edgecolor="none")
    axes.faded_guide_line(aa[0, 0], 1, axis="x")
    axes.faded_guide_line(aa[0, 1], th.DEFAULT_SHAPE.q_dec, axis="x")
    axes.faded_guide_line(aa[1, 1], 1, axis="x")
    return fig


def cadence_comparison():
    fig, aa = plt.subplots(2, 2, figsize=(style.COL_DOUBLE, 5.8), layout="constrained")
    q = np.unique(np.r_[np.linspace(.01, 6, 1000), th.DEFAULT_SHAPE.q_dec, 2,
                        th.DEFAULT_SHAPE.inverse_peak_time(4*th.DAY_S)])
    selections = [("continuous", "Continuous"), ("legacy", "Thesis step"),
                  ("peak_step", "Peak-window step"), ("random_phase", "Random phase")]
    for i, (prescription, name) in enumerate(selections):
        kw = dict(cadence_s=2*th.DAY_S, i_det=2, prescription=prescription)
        j = th.angular_moment(-2, **kw)
        aa[0, 0].plot(q, q*th.moment_density(q, -2, **kw)/j,
                      color=style.CATEGORICAL[i], linestyle=LINES[i], label=name)
        if prescription != "random_phase":
            aa[0, 1].plot(q, 1/th.selection_shape(q, **kw),
                          color=style.CATEGORICAL[i], linestyle=LINES[i], label=name)
    aa[0, 0].set(xlabel=r"$q$", ylabel=r"$p(\ln q)$", xlim=(0, 6))
    aa[0, 1].set(yscale="log", xlabel=r"$q$", ylabel=r"$L_{\mathrm{wall}}(q)/L_{\mathrm{dec}}$", xlim=(0, 6))
    axes.place_legend(aa[0, 0], loc="upper right", frameon=True, framealpha=.9, edgecolor="none")
    axes.place_legend(aa[0, 1], loc="lower right", frameon=True, framealpha=.9, edgecolor="none")
    days = np.geomspace(.01, 30, 100)
    for i, a in enumerate([-2., -1.25]):
        base = np.array([th.angular_moment(a, cadence_s=d*th.DAY_S, i_det=2,
                                          prescription="legacy") for d in days])
        for prescription, ls, name in [("random_phase", "-", "phase"), ("peak_step", "--", "step")]:
            vals = np.array([th.angular_moment(a, cadence_s=d*th.DAY_S, i_det=2,
                       prescription=prescription, n=64) for d in days])
            aa[1, 0].plot(days, vals/base, color=style.CATEGORICAL[i], linestyle=ls,
                          label=rf"$\alpha={a:g}$, {name}")
    aa[1, 0].set(xscale="log", xlabel=r"$t_{\mathrm{cad}}\ [\mathrm{day}]$",
                 ylabel=r"$R/R_{\mathrm{thesis\ step}}$")
    axes.place_legend(aa[1, 0], loc="lower left", frameon=True, framealpha=.9, edgecolor="none")
    axes.faded_guide_line(aa[1, 0], 1)
    slopes = np.linspace(-2.49, -1.01, 220)
    for i, (prescription, title) in enumerate([("continuous", "Continuous"), ("legacy", "2 day, i = 2")]):
        ratios = [th.corner_ratios(a, cadence_s=2*th.DAY_S, i_det=2, prescription=prescription) for a in slopes]
        for key, ls, name in [("angular_box_over_exact", "--", "core"),
                              ("integrated_thesis_corner_over_exact", "-", "moving")]:
            aa[1, 1].plot(slopes, [r[key] for r in ratios], color=style.CATEGORICAL[i],
                          linestyle=ls, label=f"{title}: {name}")
    aa[1, 1].set(xlabel=r"$\alpha$", ylabel=r"$R_{\mathrm{approx}}/R_{\mathrm{full}}$", ylim=(0, 1.04))
    axes.place_legend(aa[1, 1], loc="lower left", frameon=True, framealpha=.9, edgecolor="none")
    for ax, title in zip(aa.flat, [r"(a) Angular weight: $\alpha=-2$, $i=2$, 2 day",
                                  "(b) Required wall luminosity", "(c) Timing prescription errors, i = 2",
                                  "(d) Two different corner approximations"]):
        label(ax, title)
    return fig


def ztf_comparison():
    rows = list(csv.DictReader((RESULTS / "ztf_scan.csv").open()))
    a = np.array([float(r["alpha"]) for r in rows])
    score = np.array([float(r["score"]) for r in rows])
    distance = np.array([float(r["D_med_Gpc"]) for r in rows])
    selected = json.loads((RESULTS / "ztf_highlights.json").read_text())["best_shared"]["result"]
    fig, aa = plt.subplots(2, 2, figsize=(style.COL_DOUBLE, 5.8), layout="constrained")
    rate_score = np.max(np.array([np.abs(np.log([float(r[f"{w}_{m}_rate"])/target for r in rows]))
                                 for w in ["conservative", "optimistic"]
                                 for m, target in [("public", 2), ("high_cadence", 2.5)]]), axis=0)/np.log(3)
    aa[0, 0].plot(a, score, color=style.PRIMARY, label="Joint score")
    aa[0, 0].plot(a, rate_score, color=style.CATEGORICAL[0], linestyle="--", label="Rate term")
    aa[0, 0].plot(a, np.maximum(np.abs(distance/3.67-1), np.abs(distance/3.88-1))/.35,
                  color=style.CATEGORICAL[1], linestyle="-.", label="Distance term")
    aa[0, 0].plot(selected["alpha"], selected["score"], "o", color=style.ACCENT,
                  markersize=4, label="Selected shared model")
    aa[0, 0].set(xlabel=r"$\alpha$", ylabel=r"$\mathrm{normalized\ discrepancy}$",
                 xlim=(-2.1, -1.05), ylim=(.35, 1.65))
    axes.faded_guide_line(aa[0, 0], 1)
    axes.place_legend(aa[0, 0], loc="upper right", frameon=True, framealpha=.9, edgecolor="none")
    aa[1, 0].plot(a, distance, color=style.PRIMARY, label="Both surveys")
    for target, color, name in [(3.67, style.CATEGORICAL[0], "Public tolerance"),
                                 (3.88, style.CATEGORICAL[1], "High-cadence tolerance")]:
        aa[1, 0].axhspan(.65*target, 1.35*target, color=color, alpha=.09, label=name)
    aa[1, 0].plot(selected["alpha"], 4.55*th.distance_median(selected["alpha"]), "o",
                  color=style.ACCENT, markersize=4)
    aa[1, 0].set(xlabel=r"$\alpha$", ylabel=r"$D_{\mathrm{med}}\ [\mathrm{Gpc}]$",
                 xlim=(-2.1, -1.05), ylim=(1.5, 5.4))
    axes.place_legend(aa[1, 0], loc="upper left", frameon=True, framealpha=.9, edgecolor="none")
    q = np.unique(np.r_[np.linspace(0, 4, 1300), th.DEFAULT_SHAPE.q_dec, 2])
    order = [("public", "conservative", "Public C"), ("public", "optimistic", "Public O"),
             ("high_cadence", "conservative", "High C"), ("high_cadence", "optimistic", "High O")]
    for i, (mode, window, short) in enumerate(order):
        stats = selected["windows"][window]["modes"][mode]
        target = 2 if mode == "public" else 2.5
        aa[0, 1].plot(i, stats["rate"]/target, "o", color=style.CATEGORICAL[i], markersize=6)
        case = ztf.build_case(mode, window)
        density = q*ztf.filtered_moment(case, q, selected["alpha"])/stats["angular_moment"]
        aa[1, 1].plot(q, density, color=style.CATEGORICAL[i], linestyle=LINES[i], label=short)
    aa[0, 1].set(xticks=range(4), xticklabels=[t[2] for t in order],
                 ylabel=r"$R/R_{\mathrm{target}}$", ylim=(0, 3.3), xlim=(-.5, 3.5))
    for y in [1/3, 1, 3]:
        axes.faded_guide_line(aa[0, 1], y)
    aa[1, 1].set(xlabel=r"$q$", ylabel=r"$p(q)$", xlim=(0, 4))
    axes.place_legend(aa[1, 1], loc="upper right", frameon=True, framealpha=.9, edgecolor="none")
    axes.faded_guide_line(aa[1, 1], 1, axis="x")
    for ax, title in zip(aa.flat, ["(a) One normalization for all four cases", "(b) Selected effective rates",
                                  "(c) Common distance prediction", "(d) Selected angular distributions"]):
        label(ax, title)
    return fig


def main():
    style.use_style()
    for fn in [contribution_distributions, cadence_comparison, ztf_comparison]:
        paths = io.savefig_pub(fn(), fn.__name__, output_dir=OUTPUT, formats=("pdf", "png"))
        print("Saved", ", ".join(str(p.relative_to(ROOT)) for p in paths))
    (RESULTS / "figure_rendering.json").write_text(json.dumps({
        "usetex": bool(plt.rcParams["text.usetex"]), "matplotlib": matplotlib.__version__,
        "inspection": "See final validation record; generation alone is not visual approval."}, indent=2)+"\n")


if __name__ == "__main__":
    main()
