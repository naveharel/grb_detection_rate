"""GRB detection rate at the ZTF operating point vs the afterglow flux normalization.

R_det (events per year) at the ZTF strategy point, as the on-axis flux normalization
F_nu,dec is overridden across a range while ALL other physical parameters are held
fixed. The override is the physically-faithful flux-only rescaling implemented in
``figlib.overrides.set_F_dec``.

Two curves, identical ZTF configuration (optical survey mode on; at the 2-day operating
point the sub-day f_night factor is inactive, so the day model is app-faithful):
  * solid  — ZTF standard parameters only.
  * dashed — same, plus the fading-rate filter s_min = 0.3 mag/day (discrete mode).
The filter can only lower the rate (monotone in s_min), so the dashed curve sits at or
below the solid one; the gap at the fiducial F_dec is the rate ZTF gives up to the cut.

Physics shown: at ZTF's fiducial parameters the detection is *cadence-limited*
(rate independent of brightness), so R_det is flat. Only for afterglows much brighter
does the point become *flux-limited* and the rate rise with brightness.

Run:
    .venv/Scripts/python figures/fig_rate_vs_Fdec.py

Output: figures/output/fig_rate_vs_Fdec.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import numpy as np

from figures.figlib import axes, compute, io, models, overrides, presets, style

FIG_NAME = "fig_rate_vs_Fdec"

# Sweep span around the fiducial F_dec (decades below / above) and sampling density.
# Matches fig_qmedian_vs_Fdec.py so the x-axis is identical: wide enough to show the
# cadence-limited floor and the flux-limited rise.
DEX_BELOW, DEX_ABOVE, N_PTS = 2.5, 5.5, 700

# Fading-rate filter for the dashed curve. Discrete mode is the engine/app default and
# the correct model for survey alert-stream fading cuts (well-defined at ZTF i_det=10).
S_MIN, S_MODE = 0.3, "discrete"


def make_figure():
    import matplotlib.pyplot as plt

    style.use_style()

    model = models.build_model_from_preset(presets.ZTF)
    N_ztf, t_cad_ztf_s, i_det = models.ztf_strategy_point(model)

    F_fid_Jy = float(model.derived.F_dec_Jy)

    # Rate at the fiducial F_dec, with and without the fading filter (the ZTF point).
    R0_fid = compute.rate_at(model, N_ztf, t_cad_ztf_s, i_det)
    R1_fid = compute.rate_at(model, N_ztf, t_cad_ztf_s, i_det, s_min=S_MIN, s_mode=S_MODE)

    # Sweep F_dec, overriding ONLY the normalization at each step.
    F_Jy = np.logspace(np.log10(F_fid_Jy) - DEX_BELOW,
                       np.log10(F_fid_Jy) + DEX_ABOVE, N_PTS)
    overridden = [overrides.set_F_dec(model, F) for F in F_Jy]
    R0 = np.array([compute.rate_at(m, N_ztf, t_cad_ztf_s, i_det) for m in overridden])
    R1 = np.array([compute.rate_at(m, N_ztf, t_cad_ztf_s, i_det, s_min=S_MIN, s_mode=S_MODE)
                   for m in overridden])
    # Regime family from the baseline (geometric) regime — independent of s_min, so a
    # single set of bands applies to both curves.
    families = [compute.regime_family(compute.regime_id_at(m, N_ztf, t_cad_ztf_s, i_det))
                for m in overridden]

    F_mJy = F_Jy * 1e3

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Color roles & legend/overlap rules follow docs/figure_standards.md.
    fig, ax = plt.subplots(figsize=style.figsize("single", aspect=0.82))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Semantic regime shading (cool = cadence-limited, warm = flux-limited).
    axes.shade_family_bands(
        ax, F_mJy, families,
        colors={"cadence_limited": style.REGIME_FAMILY["cadence_limited"],
                "flux_limited": style.REGIME_FAMILY["flux_limited"]},
        alpha=style.SHADE_ALPHA,
        labels={"cadence_limited": "cadence-limited", "flux_limited": "flux-limited"},
        label_y=0.86,
    )

    # Two curves: same quantity (ZTF rate), distinguished by linestyle (not color) per
    # docs/figure_standards.md §2. Solid = ZTF only; dashed = ZTF + fading filter.
    ax.plot(F_mJy, R0, color=style.PRIMARY, lw=1.5, ls="-", zorder=3, label="ZTF")
    ax.plot(F_mJy, R1, color=style.PRIMARY, lw=1.5, ls="--", zorder=3,
            label=r"$s_{\min}=0.3\ \mathrm{mag\,day}^{-1}$")

    ax.set_xlabel(r"$F_{\nu,\mathrm{dec}}\ \mathrm{[mJy]}$")
    ax.set_ylabel(r"$R_{\mathrm{det}}\,[\mathrm{yr}^{-1}]$")

    # Fiducial F_dec: a full-height guide line at normal strength that fades out across the
    # "cadence-limited" band label it crosses (layering rule — the text stays fully readable).
    # The ZTF operating point is identified via the legend, not a leader line.
    axes.draw_F_dec(ax, model, axis="x", in_mJy=True)

    # Operating-point markers on each curve at the fiducial F_dec; the gap between them is
    # the rate the fading cut removes at ZTF's actual operating point.
    F_fid_mJy = F_fid_Jy * 1e3
    ax.scatter([F_fid_mJy], [R0_fid], s=24, color=style.ACCENT, zorder=5,
               label="ZTF operating point")
    if np.isfinite(R1_fid):
        ax.scatter([F_fid_mJy], [R1_fid], s=24, color=style.ACCENT, zorder=5,
                   label="_nolegend_")

    # Legend over the shaded bands → sanctioned translucent background (reads as a legend,
    # not floating data points). Forced to the lower-right empty corner so it clears the
    # "cadence-limited" region label that "best" would otherwise overlap (§3 priority).
    axes.place_legend(ax, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")

    return fig, dict(F_fid_Jy=F_fid_Jy, R0_fid=R0_fid, R1_fid=R1_fid,
                     N_ztf=float(N_ztf), t_cad_ztf_s=float(t_cad_ztf_s), i_det=int(i_det))


def main():
    fig, info = make_figure()
    paths = io.savefig_pub(fig, FIG_NAME)
    R0, R1 = info["R0_fid"], info["R1_fid"]
    print(f"F_dec (fiducial)            = {info['F_fid_Jy']*1e3:.4g} mJy")
    print(f"R_det (ZTF, no filter)      = {R0:.4g} yr^-1")
    print(f"R_det (ZTF, s_min=0.3)      = {R1:.4g} yr^-1")
    if np.isfinite(R0) and R0 > 0 and np.isfinite(R1):
        print(f"ratio  (filtered / baseline) = {R1 / R0:.4f}")
    print("wrote:", *[str(p) for p in paths])


if __name__ == "__main__":
    main()
