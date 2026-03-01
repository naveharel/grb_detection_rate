# UI & Design Reference

## Guiding principle

Modern, sleek, professional scientific tool. Dark-first. Every visual choice should reduce cognitive load, not add decoration.

## Layout structure

```
┌──────────────────────────────────────────────────┐
│ Navbar  (52 px, flex-shrink: 0)                  │
│  [☰ toggle] [title]  [preset▾] [↓CSV] [3D|N|t] [◑]│
├──────────────────────────────────────────────────┤
│ Metrics strip  (36 px, flex-shrink: 0)           │
│  R★ … | R_ZTF … | Gain … | t★ … | N★ … | t_exp★  │
├───────────┬──────────────────────────────────────┤
│ Sidebar   │  Main panel (view-panel active)       │
│ (0→300 px │                                       │
│  CSS      │   [3D surface | N-slice | t-slice]    │
│  transit) │                                       │
│           ├──────────────────────────────────────┤
│           │ Status bar (collapses when empty)     │
└───────────┴──────────────────────────────────────┘
```

- **Dark mode** is the default; toggle via `data-theme` attribute on `<html>` + CSS custom properties.
  - Dark: `[data-theme="dark"]` (default, no attribute needed if `:root` = dark)
  - Light: `[data-theme="light"]`
- **Collapsible sidebar:** open by default. CSS `width: 0 / 300px` transition on `.app-sidebar` / `.app-sidebar.open`.
- **Status bar:** uses `:not(:empty)` to collapse (zero padding, no border) when no children, avoiding dead space when not in debug mode.

## View-mode switcher

Three mutually exclusive views controlled by segmented button group in the navbar:
- **3D** (default) — full 3D surface plot
- **N-slice** — R vs N_exp at optimal t_cad
- **t-slice** — R vs t_cad at optimal N_exp

All three figures are pre-computed on every callback update. Switching is instant (clientside callback sets `display: none` / `display: flex` on the panel divs). No re-computation on view switch.

## Metrics strip

Always visible (even when debug info is hidden). Shows 6 badges separated by thin dividers:

| Badge | Content |
|-------|---------|
| R★ | Optimal detection rate [/yr] |
| R_ZTF | ZTF reference rate [/yr] |
| Gain | R★ / R_ZTF ratio (green if ≥1, coral if <1) |
| t★ | Optimal cadence (formatted: s / min / hr / day) |
| N★ | Optimal N_exp (integer) |
| t_exp★ | Optimal exposure time |

## Sidebar parameter controls

Each parameter has a **label + number input** row above a **slider**:
- Labels: concise, no parenthetical notes. E.g. `"i"` not `"i (min. detections)"`, `"log A [Jy]"` not `"log(A [Jy]) — flux anchor"`.
- Slider ↔ input sync:
  - Slider → input: clientside callback (instant, no server round-trip)
  - Input → slider: server callback on `n_blur`, value clamped to slider range
- No tooltip popups on sliders (`tooltip` prop omitted entirely).
- Parameters grouped in DBC Accordion sections: Strategy / Instrument / Constraints / Appearance.

## Regime colours

Vibrant Material Design accent palette — never muted or desaturated:

| Regimes | Colour family | Hex examples |
|---------|--------------|-------------|
| A1–A3 (distance-limited) | Warm: red → orange → amber | `#FF1744`, `#FF9100`, `#FFD740` |
| A4–A6 (cadence-limited) | Cool: blue → cyan → teal | `#2979FF`, `#00E5FF`, `#1DE9B6` |
| A7 (doubly limited) | Neutral gray | `#9E9E9E` |

**Never show "A1"…"A7" labels to users.** Use descriptive phrases ("Distance-limited · Range III") or rely on colour semantics alone.

## Markers

| Point | Symbol | Colour |
|-------|--------|--------|
| Optimal (★) | Diamond | Gold `#fbbf24` |
| ZTF reference | Circle | Coral `#f87171` |

No X markers anywhere.

## Legend & colorbar

- **Legend:** always top-left (`x=0.01, y=0.98, orientation="v"`) — avoids colorbar and Plotly toolbar overlap.
- **Colorbar:** positioned slightly inward (e.g. `x=0.85`) — never at the far right edge where Plotly's modebar icons live.

## CSS design tokens

All colours, radii, shadows, and transitions are CSS custom properties on `:root` (dark) overridden by `[data-theme="light"]`. Never hardcode colours in component styles.

Key tokens: `--bg`, `--surface-0..3`, `--border`, `--text-hi/mid/lo`, `--accent`, `--amber`, `--coral`, `--green`, `--font-ui`, `--font-mono`.
