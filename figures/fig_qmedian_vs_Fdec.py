"""Median viewing angle of ZTF-detected GRBs vs the afterglow flux normalization.

q_median (the median q = theta_obs / theta_j of detected bursts) at the ZTF strategy
point, as the on-axis flux normalization F_nu,dec is overridden across a range
while ALL other physical parameters are held fixed. The override is the
physically-faithful flux-only rescaling implemented in ``figlib.overrides.set_F_dec``.

Physics shown: at ZTF's fiducial parameters the detection is *cadence-limited*
(q_max = q_i, set by the cadence, independent of brightness), so q_median is flat.
Only for afterglows ~100x brighter does q_Euc exceed q_i and the point become
*flux-limited* (q_median rises with brightness), saturating at the non-relativistic
geometric ceiling q_median = q_nr / sqrt(2).

Run:
    .venv/Scripts/python figures/fig_qmedian_vs_Fdec.py

Output: figures/output/fig_qmedian_vs_Fdec.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import numpy as np

from figures.figlib import axes, compute, io, models, overrides, presets, style

FIG_NAME = "fig_qmedian_vs_Fdec"

# Sweep span around the fiducial F_dec (decades below / above) and sampling density.
# Wide enough to show the cadence-limited floor, the flux-limited rise, and the
# q_nr/sqrt(2) ceiling.
DEX_BELOW, DEX_ABOVE, N_PTS = 2.5, 5.5, 700


def make_figure():
    import matplotlib.pyplot as plt

    style.use_style()

    model = models.build_model_from_preset(presets.ZTF)
    N_ztf, t_cad_ztf_s, i_det = models.ztf_strategy_point(model)

    F_fid_Jy = float(model.derived.F_dec_Jy)
    q_nr = float(model.derived.q_nr)
    q_med_fid = compute.q_median_at(model, N_ztf, t_cad_ztf_s, i_det)

    # Sweep F_dec, overriding ONLY the normalization at each step.
    F_Jy = np.logspace(np.log10(F_fid_Jy) - DEX_BELOW,
                       np.log10(F_fid_Jy) + DEX_ABOVE, N_PTS)
    overridden = [overrides.set_F_dec(model, F) for F in F_Jy]
    q_med = np.array([compute.q_median_at(m, N_ztf, t_cad_ztf_s, i_det) for m in overridden])
    families = [compute.regime_family(compute.regime_id_at(m, N_ztf, t_cad_ztf_s, i_det))
                for m in overridden]

    F_mJy = F_Jy * 1e3

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Color roles & legend/overlap rules follow docs/figure_standards.md.
    fig, ax = plt.subplots(figsize=style.figsize("single", aspect=0.82))
    ax.set_xscale("log")

    # Semantic regime shading (cool = cadence-limited, warm = flux-limited).
    axes.shade_family_bands(
        ax, F_mJy, families,
        colors={"cadence_limited": style.REGIME_FAMILY["cadence_limited"],
                "flux_limited": style.REGIME_FAMILY["flux_limited"]},
        alpha=style.SHADE_ALPHA,
        labels={"cadence_limited": "cadence-limited", "flux_limited": "flux-limited"},
        label_y=0.86,  # below the q_nr/sqrt(2) ceiling line so each guide line has one clean fade
    )

    ax.plot(F_mJy, q_med, color=style.PRIMARY, lw=1.5, zorder=3)

    # Finalize the axes limits before drawing the guide lines, so the fade auto-detection
    # (faded_guide_line) sees the final text positions.
    q_sat = q_nr / np.sqrt(2.0)  # geometric saturation ceiling for the median
    ax.set_xlabel(r"$F_{\nu,\mathrm{dec}}\ \mathrm{[mJy]}$")
    ax.set_ylabel(r"$q_{\mathrm{median}}$")
    ax.set_ylim(bottom=min(q_med.min(), q_med_fid) - 0.4, top=q_sat + 0.8)

    # Geometric saturation ceiling q_nr/sqrt(2): a full-width faint guide line; its label sits
    # above the line (nothing crosses it), so it stays uniform.
    ax.axhline(q_sat, ls=":", lw=0.8, color=style.REFERENCE,
               alpha=style.REFERENCE_ALPHA, zorder=1)
    ax.annotate(r"$q_{\mathrm{nr}}/\sqrt{2}$", xy=(0.02, q_sat),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=7, color=style.REFERENCE,
                va="bottom", ha="left")

    # Fiducial F_dec: a full-height guide line at normal strength that fades out across the
    # "cadence-limited" band label it crosses (layering rule — the text stays fully readable).
    # The ZTF operating point is identified via the legend, not a leader line.
    axes.draw_F_dec(ax, model, axis="x", in_mJy=True)
    ax.scatter([F_fid_Jy * 1e3], [q_med_fid], s=24, color=style.ACCENT, zorder=5,
               label="ZTF operating point")

    # One marked point → a legend with the sanctioned translucent background (reads as a
    # legend over the shaded band, not a floating data point).
    axes.place_legend(ax, frameon=True, framealpha=0.9, edgecolor="none")

    return fig, dict(F_fid_Jy=F_fid_Jy, q_med_fid=q_med_fid, q_sat=q_sat, q_nr=q_nr,
                     q_dec=float(model.derived.q_dec), q_j=float(model.derived.q_j))


def main():
    fig, info = make_figure()
    paths = io.savefig_pub(fig, FIG_NAME)
    print(f"F_dec (fiducial)         = {info['F_fid_Jy']*1e3:.4g} mJy")
    print(f"q_median (ZTF, fiducial) = {info['q_med_fid']:.4f}")
    print(f"q_nr/sqrt(2) (ceiling)   = {info['q_sat']:.4f}")
    print(f"q_dec={info['q_dec']:.3f}  q_j={info['q_j']:.3f}  q_nr={info['q_nr']:.3f}")
    print("wrote:", *[str(p) for p in paths])


if __name__ == "__main__":
    main()
