"""Axis-formatting and reference-line helpers.

These read derived scales straight off a model (``model.derived.q_dec`` etc.) so they
stay consistent with whatever physics or F_dec override the model was built with.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from . import style

_Q_TEX = {
    "q_dec": r"$q_{\mathrm{dec}}$",
    "q_j":   r"$q_{j}$",
    "q_nr":  r"$q_{\mathrm{nr}}$",
}


def set_loglog(ax, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Log-log axes with decade major ticks and 2..9 minor subticks."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _text_fade_zones(ax, value, axis):
    """Axes-fraction spans, along the line's free axis, of in-plot text the line crosses.

    A vertical line (``axis='x'``) is tested against each text's x-extent and, when it falls
    inside, that text's *y* span (axes fraction) becomes a fade zone; a horizontal line
    (``axis='y'``) uses each crossed text's *x* span. Reads window extents after a layout pass;
    returns ``[]`` if no renderer is available so the caller falls back to a plain line.
    """
    fig = ax.figure
    try:
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return []
    inv = ax.transAxes.inverted()
    if axis == "x":
        pos = ax.get_xaxis_transform().transform((value, 0.0))[0]
    else:
        pos = ax.get_yaxis_transform().transform((0.0, value))[1]
    zones = []
    for txt in ax.texts:
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        try:
            bb = txt.get_window_extent(renderer)
        except Exception:
            continue
        if axis == "x":
            if not (bb.x0 <= pos <= bb.x1):
                continue
            f0 = inv.transform((bb.x0, bb.y0))[1]
            f1 = inv.transform((bb.x0, bb.y1))[1]
        else:
            if not (bb.y0 <= pos <= bb.y1):
                continue
            f0 = inv.transform((bb.x0, bb.y0))[0]
            f1 = inv.transform((bb.x1, bb.y0))[0]
        zones.append((min(f0, f1), max(f0, f1)))
    return zones


def faded_guide_line(ax, value, *, axis="y", color=style.REFERENCE, lw=0.8,
                     dash=(0.016, 0.012), alpha=style.REFERENCE_ALPHA, zorder=1,
                     fade=True, fade_pad=0.08):
    """Full-extent dashed guide line that fades to transparent over in-plot text it crosses.

    ``axis='y'`` draws a horizontal line at data ``value`` across the full x-axis; ``axis='x'``
    a vertical line at data ``value`` up the full y-axis. The line is built from per-dash
    segments so its alpha can ramp from ``alpha`` down to 0 across each text label it overlaps
    — a *local gradient* (the layering rule, docs/figure_standards.md §4), not a uniform dim and
    not a hard cut. Crossed labels are auto-detected from ``ax.texts`` (call this after the text
    is placed). With ``fade=False``, no crossing, or no renderer it is a plain uniform dashed
    line. ``dash=(on, off)`` and ``fade_pad`` are in axes fraction along the span. Returns the
    LineCollection.
    """
    on, off = dash
    period = on + off
    zones = _text_fade_zones(ax, value, axis) if fade else []

    def _factor(tm):
        f = 1.0
        for z0, z1 in zones:
            if tm < z0 - fade_pad or tm > z1 + fade_pad:
                continue
            if z0 <= tm <= z1:
                return 0.0
            d = (z0 - tm) if tm < z0 else (tm - z1)
            f = min(f, max(0.0, d / fade_pad))
        return f

    base = to_rgb(color)
    segs, rgba = [], []
    t = 0.0
    while t < 1.0:
        s, e = t, min(t + on, 1.0)
        a = alpha * _factor(0.5 * (s + e))
        if a > 0.0:
            segs.append([(value, s), (value, e)] if axis == "x" else [(s, value), (e, value)])
            rgba.append((*base, a))
        t += period

    trans = ax.get_xaxis_transform() if axis == "x" else ax.get_yaxis_transform()
    lc = LineCollection(segs, colors=rgba, linewidths=lw, transform=trans, zorder=zorder)
    ax.add_collection(lc, autolim=False)
    return lc


def draw_q_scales(
    ax,
    model,
    *,
    which=("q_dec", "q_j", "q_nr"),
    axis: str = "y",
    color: str = style.REFERENCE,
    lw: float = 0.8,
    alpha: float = style.REFERENCE_ALPHA,
    zorder: float = 1,
    label: bool = True,
    text_kw: dict | None = None,
):
    """Draw reference lines at the derived q-scales (q_dec, q_j, q_nr).

    ``axis='y'`` draws horizontal lines (q on the y-axis); ``axis='x'`` draws vertical
    lines. Returns the dict of plotted values. Each line is drawn via ``faded_guide_line``
    (full extent, faint, fading to transparent over any text it crosses — the layering rule);
    the labels are placed just inside the axes, on top.
    """
    d = model.derived
    vals = {"q_dec": float(d.q_dec), "q_j": float(d.q_j), "q_nr": float(d.q_nr)}
    tkw = {"color": color, "fontsize": 7, "va": "bottom", "ha": "right"}
    if text_kw:
        tkw.update(text_kw)
    for key in which:
        v = vals[key]
        faded_guide_line(ax, v, axis=axis, color=color, lw=lw, alpha=alpha,
                         zorder=zorder, dash=(0.006, 0.007))
        if label:
            if axis == "y":
                ax.annotate(_Q_TEX[key], xy=(0.99, v), xycoords=("axes fraction", "data"),
                            xytext=(0, 1), textcoords="offset points", **tkw)
            else:
                ax.annotate(_Q_TEX[key], xy=(v, 0.99), xycoords=("data", "axes fraction"),
                            xytext=(2, 0), textcoords="offset points",
                            **{**tkw, "va": "top", "ha": "left"})
    return vals


def shade_family_bands(ax, x, families, *, colors, alpha: float = 0.10,
                       labels: dict | None = None, label_y: float = 0.94,
                       label_fontsize: int = 7):
    """Shade contiguous runs of a categorical per-point label along the x-axis.

    ``x`` is the (monotonic) x-coordinate array; ``families`` is a same-length
    sequence of category strings; ``colors`` maps category -> color. Optional
    ``labels`` maps category -> display text placed once per band (at axes-fraction
    height ``label_y``). Returns the list of (category, x_lo, x_hi) bands drawn.
    """
    x = np.asarray(x, dtype=float)
    fam = list(families)
    bands = []
    i = 0
    n = len(fam)
    while i < n:
        j = i
        while j + 1 < n and fam[j + 1] == fam[i]:
            j += 1
        cat = fam[i]
        if cat in colors:
            lo, hi = x[i], x[j]
            ax.axvspan(lo, hi, color=colors[cat], alpha=alpha, lw=0, zorder=0)
            bands.append((cat, lo, hi))
            if labels and cat in labels:
                xc = np.sqrt(lo * hi) if ax.get_xscale() == "log" else 0.5 * (lo + hi)
                ax.annotate(labels[cat], xy=(xc, label_y), xycoords=("data", "axes fraction"),
                            ha="center", va="top", fontsize=label_fontsize,
                            color=colors[cat])
        i = j + 1
    return bands


def draw_F_dec(ax, model, *, axis: str = "x", in_mJy: bool = True,
               color: str = style.REFERENCE, lw: float = 1.0,
               alpha: float = style.REFERENCE_ALPHA, zorder: float = 1, **kw):
    """Reference line at the model's fiducial F_dec. Returns the value plotted.

    With ``in_mJy`` the value is converted from the engine's Jy to mJy (matching an mJy flux
    axis). ``axis='x'`` draws a vertical line, ``axis='y'`` horizontal. Drawn via
    ``faded_guide_line``: full extent, faint (``alpha``), fading to transparent over any text
    it crosses (the layering rule — labels stay fully readable). Extra kwargs (``dash``,
    ``fade``, ``fade_pad``) pass through.
    """
    val = float(model.derived.F_dec_Jy) * (1.0e3 if in_mJy else 1.0)
    faded_guide_line(ax, val, axis=axis, color=color, lw=lw, alpha=alpha,
                     zorder=zorder, **kw)
    return val


def place_legend(ax, *, loc: str = "best", **kw):
    """Frameless legend following the project legend-placement priority.

    Default ``loc='best'`` lets matplotlib pick the inside location least overlapping the
    data. If that still collides (caught in visual inspection), force an empty corner in
    priority order upper right -> upper left -> lower right -> lower left; or move it
    outside-right with ``loc='center left', bbox_to_anchor=(1.02, 0.5)`` when the data fills
    the axes. Frameless is the house style; pass ``frameon=True, framealpha=0.9,
    edgecolor='none'`` for the sanctioned translucent background when a legend must sit over
    shading (so it reads as a legend, not a floating data point). Use a legend for discrete
    series/marked points; reserve in-plot text for labelling regions. Full spec:
    docs/figure_standards.md. Returns the Legend.
    """
    return ax.legend(loc=loc, **kw)
