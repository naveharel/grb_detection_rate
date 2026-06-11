# Figures subsystem — standards & usage

Reference for `figures/`, the publication-quality plotting subsystem. It generates
static PNG figures (vector PDF opt-in) for the paper and internal testing by **importing** the physics
engine in `grb_detect/`. The engine stays read-only; figures never modify it.

`figures/README.md` is the one-screen quick start. This document is the full standard.

---

## 1. Setup

```
.venv\Scripts\python -m pip install -r requirements-figures.txt   # matplotlib (numpy already present)
```

Dependencies live only here — the in-browser app build (`build_standalone.py`) stays
stdlib + numpy and never imports matplotlib.

**LaTeX (default renderer).** Text is rendered through a real LaTeX install
(`text.usetex`) so fonts match the manuscript. This needs a system TeX toolchain
(`latex` + `dvipng`) — MiKTeX or TeX Live; it is **not** a pip package.

- **usetex is the default.** `style.use_style()` with no argument uses LaTeX.
- **Auto-discovery.** If `latex`/`dvipng` aren't on `PATH`, `use_style` probes the
  standard install locations (per-user & system MiKTeX, TeX Live) and prepends the
  toolchain's bin dir to the process `PATH` — so a normally-installed TeX works even
  when the shell's `PATH` doesn't list it (see `style._candidate_tex_bindirs`).
- **Safety fallback.** If no TeX toolchain is found (not on `PATH`, not in the standard
  locations), `use_style` emits a `RuntimeWarning` and falls back to matplotlib
  mathtext (STIX, Times-compatible) so figures still build on a machine without TeX.
- **Override** per call with `style.use_style(usetex=False)` (or `True`), or globally
  with the `GRBFIG_USETEX=0/1` environment variable. Precedence: explicit arg >
  env var > default (LaTeX).

> Windows note: a TeX installed in a non-standard location (outside the MiKTeX/TeX Live
> defaults auto-discovery probes) still needs its bin dir on `PATH`, e.g.
> `…\MiKTeX\miktex\bin\x64`. Installing TeX while a shell is open also won't update that
> shell's `PATH` — open a new terminal.

---

## 2. Visual standards & the inspection gate

The full visual spec — house style, **color roles**, **legend-placement priority**,
**text-visibility / anti-overlap** rules, math-label conventions, output formats — and the
**mandatory visual-inspection gate** every figure must pass before it counts as done live
in their own focused doc:

> **[`docs/figure_standards.md`](figure_standards.md)** — read it before making or
> reviewing a figure.

Those standards are encoded in `figures/styles/grbpaper_aas.mplstyle` (static rcParams) and
`figures/figlib/style.py` (palettes, color roles, usetex toggle), and applied through the
`figlib` helpers below — reference them by name rather than re-hardcoding values.

---

## 3. Architecture (`figures/figlib/`)

| Module | Responsibility |
|--------|----------------|
| `style` | `use_style(usetex=None)`, `figsize()`, `PALETTE`/`CATEGORICAL`, color roles (`PRIMARY`/`ACCENT`/`REFERENCE`/`SHADE_ALPHA`), regime colors, `tex_available()` |
| `presets` | `SurveyPreset` dataclasses mirroring `web/app.js` `PRESETS` (`ZTF`, `RUBIN`) — single source of truth |
| `models` | `build_model_from_preset()`, `ztf_strategy_point()`, `n_exp_max()` |
| `compute` | `q_median_at()`, `medians_at()`, `rate_at()`, `regime_id_at()`, `regime_family()` |
| `overrides` | copy-on-write "model surgery" to vary quantities the public API hides |
| `axes` | `set_loglog()`, `faded_guide_line()`, `draw_q_scales()`, `draw_F_dec()`, `shade_family_bands()`, `place_legend()` |
| `io` | `savefig_pub()` |

Two layers, by intent:
- **Reproduce the app/engine** — `models` + `compute` build models exactly as
  `standalone_bridge.py` does, so figures match the app. For whole surfaces/slices you
  may also call `standalone_bridge.compute_all(params)` directly.
- **Go past the public API** — `overrides` (§5).

---

## 4. Writing a new figure

Each `figures/fig_*.py` follows the same five steps and exposes `make_figure()`
returning the `Figure` (so a future "build all" can import them). The script puts the
repo root on `sys.path`, so run it directly:

```
.venv\Scripts\python figures\fig_<name>.py        # writes figures/output/<name>.png
```

Skeleton:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from figures.figlib import axes, compute, io, models, overrides, presets, style

def make_figure():
    import matplotlib.pyplot as plt
    style.use_style()                                   # 1. style (usetex default)
    model = models.build_model_from_preset(presets.ZTF) # 2. model(s)
    N, t, i = models.ztf_strategy_point(model)
    y = compute.q_median_at(model, N, t, i)             # 3. compute (engine API)
    fig, ax = plt.subplots(figsize=style.figsize("single"))
    ax.plot(..., color=style.PRIMARY)                   # 4. plot (color roles — §2 spec)
    ax.set_xlabel(r"$...$"); ax.set_ylabel(r"$...$")
    axes.place_legend(ax)                               #    legend per placement priority
    return fig

if __name__ == "__main__":
    io.savefig_pub(make_figure(), "fig_<name>")         # 5. save (PNG; PDF opt-in via formats=)
```

**Before it counts as done**, run the mandatory visual-inspection gate: render the PNG,
open it, and check it against the checklist in
[`docs/figure_standards.md`](figure_standards.md#7-visual-inspection-mandatory).

Multi-panel/complex figures follow the same steps — build several models (e.g. ZTF +
Rubin, or a sweep of `overrides`-modified models) and use
`plt.subplots(..., figsize=style.figsize("double", aspect=...))`.

---

## 5. Going past the public API — `overrides`

`make_rate_model` exposes survey + physics parameters but **not** the engine's derived
internal scales (`t_dec`, `q_dec/q_j/q_nr`, and the flux normalizations
`F_dec/F_j/F_nr`). To vary those in a figure, use `figlib.overrides`.

Mechanism: `make_rate_model` is `lru_cache`d (identical builds return the *same*
object), and `DerivedAfterglowScales` is frozen. So overrides do **copy-on-write**:
`copy.copy(model)` → bind a `dataclasses.replace`-d `_derived` on the copy. The cached
original is never mutated.

```python
from figures.figlib import models, overrides, presets
m  = models.build_model_from_preset(presets.ZTF)
m2 = overrides.set_F_dec(m, 0.5e-3)     # flux normalization → 0.5 mJy (all else fixed)
m3 = overrides.scale_F_dec(m, 100.0)    # ×100 brighter
```

`set_F_dec`/`scale_F_dec` are physically faithful: they scale `F_dec_Jy`, `F_j_Jy`,
`F_nr_Jy` by the **same** factor. This is exact because `F_j`/`F_nr` are `F_dec` times
pure functions of `q_dec`/`q_nr` (independent of `F_dec`), and flux enters the
q-computation only through the ratio `F_lim/F_dec` and the test `F_lim < F_j`. It is
equivalent to a native `A_log` rebuild (verified — see §6).

**Contract for new override helpers:**
1. Operate on a copy (`overrides.copy_model` / `override_derived`); never mutate a
   cached instance in place.
2. Scale/replace physically-coupled fields together (as `set_F_dec` does).
3. Document the physics invariant the override relies on.
4. Prefer rebuilding via `make_rate_model` for anything the public API already exposes
   (physics params); reserve surgery for the hidden derived scales.

**Do not** add override hooks to the engine (`grb_detect/`, `standalone_bridge.py`) —
that constraint is the whole reason this layer exists.

---

## 6. Testing & gotchas

- **Override faithfulness** is pinned by `tests/test_fdec_override.py`:
  `set_F_dec(m, k·F)` must reproduce a natively-built model with `A_log` shifted by
  `-log10(k)`, for k across the cadence-/flux-/q_nr-limited regimes. Add an analogous
  faithfulness test for any new override helper.
- Run everything: `.venv\Scripts\python -m pytest tests/ -q`.
- **N_exp_max boundary gotcha.** The ZTF reference point sits exactly on the
  `N_exp_max` boundary. Always derive `N_exp` from the model
  (`models.ztf_strategy_point` → `n_exp_max(model)`); a hardcoded `27500/47` can exceed
  `N_exp_max(model)` by a float epsilon, making the strategy unphysical → `NaN`.
- **Preset sync.** `figlib/presets.py` mirrors `web/app.js` `PRESETS`. If you change a
  preset in either place, update the other.
- **App-parity check.** At the fiducial F_dec, `q_median_at(model, *ztf_strategy_point(model))`
  equals `standalone_bridge.compute_all(default ZTF params)["q_med_ztf"]` (currently
  2.81801305).

---

## 7. Example

`figures/fig_qmedian_vs_Fdec.py` — median viewing angle `q_median` of ZTF-detected GRBs
as the flux normalization `F_nu,dec^(G)` is overridden (all other physics fixed). Shows
the cadence-limited floor (`q_median = q_i/√2`), the flux-limited rise, and the
geometric ceiling `q_median = q_nr/√2`.
