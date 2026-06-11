# Figure visual standards & the inspection gate

What every `figures/fig_*.py` must look like, and the mandatory check it must pass before
it counts as done. This is the spec consulted when **making or reviewing** a figure;
`docs/figures.md` covers how the subsystem is built and run. The standards are AAS/ApJ
conventions, encoded in `figures/styles/grbpaper_aas.mplstyle` (static rcParams) and
`figures/figlib/style.py` (palettes, color roles, the usetex toggle) and applied through
`figlib` helpers — reference them by name, don't re-hardcode values.

---

## 1. House style (AAS/ApJ)

| Aspect | Standard |
|--------|----------|
| Family | Times-like serif. usetex preamble `\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathptmx}`; fallback `mathtext.fontset = stix` |
| Sizes | base 9 pt; axis labels 9; tick labels 8; legend 8; annotations ≥ 7 (never below 7) |
| Column widths | single 3.5 in, double 7.1 in (`style.COL_SINGLE` / `COL_DOUBLE`, via `style.figsize`) |
| Ticks | inward, on all four sides, with minor ticks |
| Layout | `constrained_layout` |
| Lines | axes 0.8 pt, data 1.3 pt, reference lines ≤ 1.0 pt |
| Output | PNG by default at `savefig.dpi = 600`, `bbox='tight'`; opt-in PDF uses `pdf.fonttype = 42` (editable vector text) |
| Legend | frameless (one sanctioned exception in §3) |

---

## 2. Color roles — which color for what

Apply color **by role**, via the constant, so every figure reads the same way. Never
encode meaning by color alone (the palette is colorblind-safe, but print/photocopy is not):
pair color with linestyle, marker, or a label.

| Role | Constant | Value | Use |
|------|----------|-------|-----|
| Primary | `style.PRIMARY` | black | the single main data series |
| Categorical | `style.CATEGORICAL` | blue, orange, green, red, purple, sky | multiple series, taken in order (also the default `prop_cycle`) |
| Accent | `style.ACCENT` | red | the one operating point / marker that must stand out |
| Reference | `style.REFERENCE` | grey `0.45` | reference/guide lines (`axes.faded_guide_line`) **and their labels**; thin, dashed/dotted, drawn at `style.REFERENCE_ALPHA = 0.6` and fading to transparent over text they cross (see §4 layering) |
| Regime | `style.REGIME_FAMILY` | warm / cool / grey | regime figures only — warm = flux-limited (A1–A3), cool = cadence-limited (A4–A6), grey = doubly-limited (A7) |

`style.REGIME_COLORS` (ids 1–7) and `style.regime_cmap_and_norm()` color shaded regime
surfaces; `axes.shade_family_bands()` shades contiguous regime bands along an axis. Region
shading uses alpha `style.SHADE_ALPHA` (= 0.10) so overlaid text stays legible. Do **not**
print "A1…A7" regime labels — use the descriptive words or the semantic color.

---

## 3. Identifying series & points — legends vs. labels

Use a **legend** to identify discrete series and marked points (a curve, an operating
point); use **in-plot text** to label *regions* (e.g. the shaded regime bands). Do **not**
tie a label to a point with a leader / annotation line — leaders clutter the plot, and a
lone labelled marker reads as stray data. (So `fig_qmedian_vs_Fdec.py` shows its ZTF
operating point in the legend and names its regime *bands* with direct text.)

Legend placement, in priority order:
1. **Default `axes.place_legend(ax)`** → `loc="best"`: matplotlib picks the inside spot
   least overlapping the data. Frameless (house rcParam).
2. **Over data or shading, add the sanctioned semi-opaque background** so it reads as a
   legend, not a floating data point, and stays legible:
   `axes.place_legend(ax, frameon=True, framealpha=0.9, edgecolor="none")`.
3. If "best" still collides (caught in inspection), force an empty corner in priority order
   **upper right → upper left → lower right → lower left** (largest empty region).
4. If the data fills the axes, move the legend **outside-right**:
   `axes.place_legend(ax, loc="center left", bbox_to_anchor=(1.02, 0.5))`.

---

## 4. Text visibility & anti-overlap

- **Sizes.** Labels/title 9 pt, ticks/legend 8 pt, annotations ≥ 7 pt — never below 7.
- **Contrast.** Put text on the plain background or over shading with alpha ≤ 0.12; text
  color at least as dark as `0.45`. Never place dark text over a saturated fill.
- **No overlap.** Text must not overlap other text, the data curve, or markers. A guide line
  **fades out** where it crosses text (below), so it never sits on a label. Still, place a
  reference-line's label at the line's *quiet* end (a corner with no curve) and offset an
  annotation ≥ 3 pt from its anchor.
- **Layering — guide lines fade over text.** Guide/reference lines span the full axis at their
  normal strength (`style.REFERENCE_ALPHA`) and **fade out — alpha gradient to transparent —
  over the span where they cross a text label** (like the dashed phase lines in a published
  light-curve schematic): a *local gradient*, **not** a uniform dim and **not** a hard cut. The
  line resumes full strength past the label, so it still reads as spanning the whole axis.
  Z-order keeps guide lines beneath the data (curve/markers on top); the **fade** is
  specifically how a line yields to *text*. Stack, back → front: shaded bands → guide lines →
  data curves → markers/points → text labels & legend (always on top). `axes.faded_guide_line`
  (and `draw_F_dec` / `draw_q_scales`, which delegate to it) auto-detect crossed labels and
  apply the fade; use it for any guide line. (A legend's own background already hides a line
  behind it — no fade needed there.)
- **No clipping.** Nothing clipped at the axes edges. `bbox='tight'` pads the saved figure,
  but keep labels a few pt inside the axes so they read at 100 %.
- **Rotated labels.** A 90° label for a vertical reference line sits *beside* the line in
  empty space, offset — never on top of it.

---

## 5. Math & labels

Always write labels as math strings `$...$` — they render in **both** usetex and mathtext
from the same source. Use `\mathrm{}` for upright multi-letter subscripts. Examples in use:
`r"$F_{\nu,\mathrm{dec}}^{(G)}\ \mathrm{[mJy]}$"`, `r"$q_{\mathrm{median}}$"`,
`r"$R_{\mathrm{det}}\,[\mathrm{yr}^{-1}]$"`. (The `<sub>…</sub>` rule is for the in-browser
app only; it does not apply to matplotlib figures.)

---

## 6. Output formats

`io.savefig_pub(fig, name)` writes `<name>.png` (600 dpi) to `figures/output/` by default —
**PNG only**. The vector `<name>.pdf` (the paper deliverable, `pdf.fonttype = 42` editable
text) is **opt-in**: pass `formats=("pdf", "png")` (or `("pdf",)`) only when a PDF is
specifically wanted. Names are deterministic (no timestamps) so re-runs overwrite and
`\includegraphics` references stay stable. `figures/output/` is gitignored — scripts are
the source of truth; regenerate on demand.

---

## 7. Visual inspection (mandatory)

Every figure must pass a visual inspection before it counts as done — including any time a
figure is regenerated. The mechanism is literally **looking at the rendered PNG**:

1. Render to PNG with `io.savefig_pub`.
2. Open the PNG and inspect it (zoom in) against this checklist:
   - [ ] Builds; usetex active — or the mathtext fallback is intended and noted.
   - [ ] Colors follow the role table in §2.
   - [ ] Discrete series/points use a legend (translucent background if over shading — reads
         as a legend, not a stray point); regions use in-plot text; no leader lines.
   - [ ] All text meets its §1 minimum size and is legible at 100 %.
   - [ ] No text overlaps other text, the data curve, or markers (a faint guide line behind
         text is fine).
   - [ ] Guide/reference lines render at normal strength and **fade out** over any text they
         cross (a smooth gradient — no hard cut, no uniform dim; §4 layering); the text is fully
         legible and the line still reads as spanning the axis.
   - [ ] No text or marker clipped at the axes edges.
   - [ ] Axis labels and units are correct; all math renders.
   - [ ] Physics annotations point at the right features.
3. If any item fails, fix the script and re-render; repeat until every item passes.

`fig_qmedian_vs_Fdec.py` is the worked example that first passed this gate.
