"""AAS / ApJ matplotlib style for publication figures.

`use_style()` applies the committed ``styles/grbpaper_aas.mplstyle`` and then layers
the LaTeX rendering choice on top:

* ``usetex=True``  — render all text through a real LaTeX install (manuscript-exact
  fonts). Requires a working TeX toolchain (``latex`` + ``dvipng``); if it is not on
  PATH, ``use_style`` auto-discovers a standard MiKTeX/TeX Live install and adds it.
* ``usetex=False`` — matplotlib mathtext with STIX (Times-compatible) fonts. No TeX
  needed. Math/labels written as ``$...$`` render in both modes from the same string.
* ``usetex=None`` (default) — use LaTeX. This is the project default for the paper.
  As a safety net, if no TeX toolchain is found ``use_style`` warns and falls back to
  mathtext so figures still build on a machine without TeX.

Resolution precedence for the effective mode: an explicit ``usetex=`` argument beats
the ``GRBFIG_USETEX`` env var (1/0), which beats the default (LaTeX).
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

_STYLE_PATH = Path(__file__).resolve().parent.parent / "styles" / "grbpaper_aas.mplstyle"

# ── Column widths (inches) — AAS / ApJ ───────────────────────────────────────
COL_SINGLE = 3.5
COL_DOUBLE = 7.1

# LaTeX preamble used when usetex is active. mathptmx gives Times text+math and is
# present in essentially every TeX distribution (newtxmath is a modern alternative
# but is not always installed). Edit here to switch fonts.
LATEX_PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathptmx}"

# ── Colorblind-safe categorical palette (Wong 2011) ──────────────────────────
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#7F7F7F",
}
CATEGORICAL = [PALETTE[k] for k in ("blue", "orange", "green", "red", "purple", "sky")]

# ── Regime semantics (opt-in; for regime-coloured figures only) ──────────────
# warm = flux-limited (A1-A3), cool = cadence-limited (A4-A6), neutral = A7.
REGIME_COLORS = {
    1: "#B2182B", 2: "#EF8A62", 3: "#FDDBC7",   # warm
    4: "#2166AC", 5: "#67A9CF", 6: "#D1E5F0",   # cool
    7: "#7F7F7F",                                # neutral
}
REGIME_FAMILY = {
    "flux_limited":    PALETTE["red"],
    "cadence_limited": PALETTE["blue"],
    "doubly_limited":  PALETTE["grey"],
}

# ── Semantic color roles (which color for what) ──────────────────────────────
# Reference these by role, not by raw hex, so every figure reads the same way.
# Full spec: docs/figure_standards.md.
PRIMARY = PALETTE["black"]   # the single main data series
ACCENT = PALETTE["red"]      # the one operating point / marker that must stand out
REFERENCE = "0.45"           # reference lines (axhline/axvline) AND their labels (grey)
SHADE_ALPHA = 0.10           # alpha for regime/region shading (keeps overlaid text legible)
REFERENCE_ALPHA = 0.6        # guide line's normal/open-space alpha; it fades to 0 over text


def tex_available() -> bool:
    """True if a usetex toolchain (latex + dvipng) is on PATH."""
    return bool(shutil.which("latex")) and bool(shutil.which("dvipng"))


def _candidate_tex_bindirs() -> list[str]:
    """Well-known TeX ``bin`` directories to probe when latex/dvipng aren't on PATH.

    Covers per-user and system MiKTeX and year-stamped TeX Live trees. Returns only
    paths that exist (newest TeX Live year first).
    """
    import glob
    import sys

    cands: list[str] = []
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        if local:  # per-user MiKTeX (the default "just me" install)
            cands.append(os.path.join(local, "Programs", "MiKTeX", "miktex", "bin", "x64"))
        cands += [  # system-wide MiKTeX
            os.path.join(pf, "MiKTeX", "miktex", "bin", "x64"),
            os.path.join(pfx86, "MiKTeX", "miktex", "bin", "x64"),
        ]
        for sub in ("windows", "win32"):  # TeX Live: C:\texlive\<year>\bin\<sub>
            cands += sorted(glob.glob(rf"C:\texlive\*\bin\{sub}"), reverse=True)
    else:
        cands += ["/Library/TeX/texbin", "/usr/local/bin", "/opt/homebrew/bin"]
        cands += sorted(glob.glob("/usr/local/texlive/*/bin/*"), reverse=True)
    return [d for d in cands if os.path.isdir(d)]


def _ensure_tex_on_path() -> bool:
    """Make latex/dvipng discoverable for usetex; return whether they now are.

    If a toolchain is already on PATH, do nothing. Otherwise probe the standard TeX
    install directories (per-user MiKTeX, system MiKTeX, TeX Live) and, if one supplies
    both ``latex`` and ``dvipng``, prepend it to this process's PATH. This lets the
    LaTeX default work when TeX is installed but never exported to PATH — common for a
    per-user MiKTeX install on Windows.
    """
    if tex_available():
        return True
    for d in _candidate_tex_bindirs():
        if shutil.which("latex", path=d) and shutil.which("dvipng", path=d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return True
    return False


def _resolve_usetex(usetex: bool | None) -> bool:
    if usetex is None:
        env = os.environ.get("GRBFIG_USETEX")
        if env is not None:
            return env.strip() not in ("", "0", "false", "False", "no")
        return True  # usetex is the project default; use_style() falls back safely
    return bool(usetex)


def use_style(usetex: bool | None = None) -> bool:
    """Apply the publication style. Returns the effective ``usetex`` flag.

    ``usetex`` defaults to LaTeX (``None`` -> True). If LaTeX is requested but no TeX
    toolchain (``latex`` + ``dvipng``) is on PATH, a warning is emitted and the style
    falls back to mathtext so figures still build.
    """
    plt.style.use(str(_STYLE_PATH))

    effective = _resolve_usetex(usetex)
    if effective and not _ensure_tex_on_path():
        warnings.warn(
            "usetex requested but no LaTeX toolchain (latex/dvipng) found on PATH or in "
            "the standard MiKTeX/TeX Live install locations; rendering would fail. "
            "Falling back to mathtext. Install TeX (MiKTeX or TeX Live) to enable usetex.",
            RuntimeWarning,
            stacklevel=2,
        )
        effective = False

    mpl.rcParams["text.usetex"] = effective
    if effective:
        mpl.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
    return effective


def figsize(width: str | float = "single", aspect: float = 0.72) -> tuple[float, float]:
    """Return (w, h) in inches. ``width`` is 'single', 'double', or an explicit float.

    Height = width * aspect.
    """
    w = {"single": COL_SINGLE, "double": COL_DOUBLE}.get(width, width)
    return (float(w), float(w) * float(aspect))


def regime_cmap_and_norm():
    """(ListedColormap, BoundaryNorm) for imshow/pcolormesh of regime ids 1..7."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap([REGIME_COLORS[k] for k in range(1, 8)])
    norm = BoundaryNorm(np.arange(0.5, 8.5, 1.0), cmap.N)
    return cmap, norm
