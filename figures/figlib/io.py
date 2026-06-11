"""Figure output helper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def savefig_pub(
    fig,
    name: str,
    *,
    output_dir: Path | None = None,
    formats=("png",),
    dpi_png: int = 600,
    close: bool = True,
):
    """Save ``fig`` as ``<name>.<ext>`` for each format into the output dir.

    Deterministic filenames (no timestamp) so re-running overwrites cleanly. Defaults
    to PNG only (raster at ``dpi_png``). Vector PDF for the paper is opt-in — pass
    ``formats=("pdf", "png")`` (or ``("pdf",)``) when you specifically want it. Returns
    the list of written paths.
    """
    out = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out / f"{name}.{ext}"
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = dpi_png
        fig.savefig(path, **kw)
        written.append(path)
    if close:
        plt.close(fig)
    return written
