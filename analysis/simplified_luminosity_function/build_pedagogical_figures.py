"""Build two explanatory LF figures, using the unchanged analysis theory.

Run from the repository root with::

    .venv/Scripts/python.exe analysis/simplified_luminosity_function/build_pedagogical_figures.py

The project publication style discovers LaTeX when available and otherwise
intentionally uses its STIX mathtext fallback. Only the two named PDF/PNG
outputs are written. Luminosity always means L_nu(t_dec).
"""
from __future__ import annotations

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

OUTPUT = Path(__file__).resolve().parent / "results" / "figures"
ALPHAS = (-2.4, -2.0, -1.25)
LINESTYLES = ("-", "--", "-.")


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=9, pad=7)


def pedagogical_distributions():
    """Show the normalized luminosity, distance, and angular distributions."""
    fig, panels = plt.subplots(
        2, 2, figsize=(style.COL_DOUBLE, 5.7), layout="constrained")
    luminosity_ratio = np.geomspace(1e-6, 1e6, 1601)
    distance_ratio = np.unique(np.r_[0, np.geomspace(1e-7, 1, 1200)])
    angle = np.unique(np.r_[np.geomspace(0.05, th.DEFAULT_SHAPE.q_nr, 2000),
                            th.DEFAULT_SHAPE.q_dec, 2])
    for index, alpha in enumerate(ALPHAS):
        appearance = dict(color=style.CATEGORICAL[index],
                          linestyle=LINESTYLES[index],
                          label=rf"$\alpha={alpha:g}$")
        panels[0, 0].plot(luminosity_ratio,
            th.conditional_log_luminosity_density(luminosity_ratio, alpha),
            **appearance)
        panels[0, 1].plot(distance_ratio, th.distance_cdf(distance_ratio, alpha),
                         **appearance)
        panels[1, 0].plot(angle,
            angle * th.moment_density(angle, alpha) / th.angular_moment(alpha),
            **appearance)
        panels[1, 1].plot(angle,
            [th.angular_cdf(value, alpha) for value in angle], **appearance)

    panels[0, 0].set(
        xscale="log", xlim=(1e-6, 1e6), ylim=(0, 0.37),
        xlabel=r"$L/L_{\mathrm{Euc}}(q)$",
        ylabel=r"Normalized rate per $\mathrm{d}\ln L$ (fixed $q$)")
    panels[0, 1].set(
        xlim=(0, 1), ylim=(0, 1.03), xlabel=r"$D/D_{\mathrm{Euc}}$",
        ylabel=r"$R_{\mathrm{det}}(<D)/R_{\mathrm{det}}$")
    panels[1, 0].set(
        xscale="log", xlim=(0.05, th.DEFAULT_SHAPE.q_nr), ylim=(0, 2.12),
        xlabel=r"$q=\theta_{\mathrm{obs}}/\theta_j$",
        ylabel=r"$\dfrac{1}{R_{\mathrm{det}}}\dfrac{\mathrm{d}R_{\mathrm{det}}}{\mathrm{d}\ln q}$")
    panels[1, 1].set(
        xscale="log", xlim=(0.05, th.DEFAULT_SHAPE.q_nr), ylim=(0, 1.03),
        xlabel=r"$q=\theta_{\mathrm{obs}}/\theta_j$",
        ylabel=r"$R_{\mathrm{det}}(<q)/R_{\mathrm{det}}$")

    for panel, title in zip(panels.flat, (
            "(a) Luminosity contributions at fixed angle",
            "(b) Cumulative distance distribution",
            "(c) Angular rate contributions",
            "(d) Cumulative angular distribution")):
        _title(panel, title)

    axes.place_legend(panels[0, 0], loc="upper right")
    axes.place_legend(panels[0, 1], loc="lower right")
    axes.faded_guide_line(panels[0, 0], 1, axis="x")
    for panel in (panels[1, 0], panels[1, 1]):
        axes.faded_guide_line(panel, 1, axis="x")
    return fig


def pedagogical_cadence():
    """Show how the thesis repeated-detection condition moves the threshold."""
    fig, panels = plt.subplots(
        1, 2, figsize=(style.COL_DOUBLE, 3.05), layout="constrained")
    selection = dict(cadence_s=2*th.DAY_S, i_det=2, prescription="legacy")
    cadence_angle = th.DEFAULT_SHAPE.inverse_peak_time(4*th.DAY_S)
    angle = np.unique(np.r_[np.linspace(0, 6, 2200),
        th.DEFAULT_SHAPE.q_dec, 2, cadence_angle])
    continuous_threshold = 1/th.selection_shape(angle)
    repeated_threshold = 1/th.selection_shape(angle, **selection)
    cadence_threshold = float(1/th.DEFAULT_SHAPE.lightcurve(4*th.DAY_S))

    # Draw the required threshold first, then its dashed continuous component:
    # beyond q_i the superposed curves deliberately show their equality.
    panels[0].plot(angle, repeated_threshold, color=style.PRIMARY,
        label=r"$\max[L_{\mathrm{Euc}}(q),L_i]/L_{\mathrm{dec}}$")
    panels[0].plot(angle, continuous_threshold, color=style.CATEGORICAL[0],
        linestyle="--", label=r"$L_{\mathrm{Euc}}(q)/L_{\mathrm{dec}}$")
    cadence_line = axes.faded_guide_line(panels[0], cadence_threshold)
    cadence_line.set_label(r"$L_i/L_{\mathrm{dec}}$")
    panels[0].set(xlim=(0, 6), yscale="log", ylim=(0.5, 2e8),
        xlabel=r"$q=\theta_{\mathrm{obs}}/\theta_j$",
        ylabel=r"Threshold luminosity divided by $L_{\mathrm{dec}}$")

    for selection_parameters, color, line, label in (
            ({}, style.CATEGORICAL[0], "--", "Continuous detection"),
            (selection, style.PRIMARY, "-", "Thesis repeated detections")):
        panels[1].plot(angle,
            angle*th.moment_density(angle, -2, **selection_parameters)
            / th.angular_moment(-2, **selection_parameters),
            color=color, linestyle=line, label=label)
    panels[1].set(xlim=(0, 6), ylim=(0, 2.12),
        xlabel=r"$q=\theta_{\mathrm{obs}}/\theta_j$",
        ylabel=r"$\dfrac{1}{R_{\mathrm{det}}}\dfrac{\mathrm{d}R_{\mathrm{det}}}{\mathrm{d}\ln q}$")
    _title(panels[0], "(a) Higher luminosity required by cadence")
    _title(panels[1], r"(b) Angular contributions, $\alpha=-2$")
    axes.place_legend(panels[0], loc="lower right", frameon=True,
                      framealpha=0.9, edgecolor="none")
    axes.place_legend(panels[1], loc="upper right", frameon=True,
                      framealpha=0.9, edgecolor="none")
    for panel in panels:
        axes.faded_guide_line(panel, cadence_angle, axis="x")
    return fig


def main():
    using_tex = style.use_style()
    print("Text rendering:", "LaTeX" if using_tex else "intended STIX mathtext fallback")
    for build in (pedagogical_distributions, pedagogical_cadence):
        paths = io.savefig_pub(build(), build.__name__, output_dir=OUTPUT,
                               formats=("pdf", "png"))
        print("Saved", ", ".join(str(path.relative_to(ROOT)) for path in paths))


if __name__ == "__main__":
    main()
