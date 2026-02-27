# GRB Detection Rate Project — Claude Context

## Project Purpose
Interactive Dash web app modeling GRB (gamma-ray burst) afterglow detection rates as a function of survey strategy parameters. Physics-first; everything must faithfully represent the underlying analytic model from the Bachelor's thesis (Tier 1, see below).

## Source Hierarchy
1. **Tier 1 (ground truth):** Bachelor's thesis "Detection Rates of GRB Afterglows in Optical Surveys — An Analytic Approach" + the two Hebrew presentations. All code must agree with these.
2. **Tier 2 (literature):** External papers in `sources/`. Use to learn subject; flag contradictions with Tier 1.
3. **Tier 3 (unverified notes):** "Project notes" and "Tentative Paper structure" PDFs. Treat with skepticism; cross-reference against Tier 1.

See [physics-model.md](.claude/physics-model.md) for the full physics reference.

## File Map
| File | Role |
|------|------|
| `plot_3d.py` | Dash app layout + callbacks (main entry point) |
| `grb_detect/plot_3d_core.py` | Surface computation, optical cadence grid, optimizer |
| `grb_detect/detection_rate.py` | Core `DetectionRateModel` class — piecewise rate formulas |
| `grb_detect/params.py` | Frozen dataclasses for all parameters |
| `grb_detect/afterglow_ism.py` | Derived scales: t_dec, t_j, q_dec, q_j, q_nr |
| `grb_detect/pls.py` | Power-law segment (PLS) models: PLSG, PLSH |
| `grb_detect/survey.py` | Survey strategy helpers: t_exp, F_lim, sky coverage |
| `grb_detect/constants.py` | Unit conversions (DAY_S, DEG2RAD, etc.) |
| `assets/style.css` | CSS design system: dark/light theme, rc-slider overrides, layout |
| `sources/` | Physics reference documents (PDFs, presentations) |

## Key Physics Summary
- **4 afterglow phases:** I (t < t_dec, coasting), II (t_dec–t_j, rel. decel.), III (t_j–t_nr, jet), IV (t > t_nr, Newtonian)
- **PLS G** (optical band, ν_m < ν < ν_c): most relevant. a_II = p−1, a_III = p, default p = 2.5
- **7 detection regimes (A1–A7):** classified by whether q_Euc or q_i is the binding constraint, and which phase applies
  - A1–A3: flux-limited (q_Euc > q_i) — physically: going deeper helps
  - A4–A6: cadence-limited (q_i > q_Euc) — physically: observing more frequently helps
  - A7: doubly limited (both constraints tight)
- **Key scales:** t_dec ≈ 19 s, t_j ≈ 10⁴ × t_dec ≈ 0.66 d, q_dec ≈ 1.001, q_j = 2, q_nr = √2/θ_j ≈ 14
- **Euclidean approximation:** D_euc ≈ 5.28 Gpc (z ≈ 2.0 in flat ΛCDM, H0=70). Hardcoded consistently with z_Euc=2.0.
- **f_live two-model architecture:** Sub-day cadence uses plain `f_live`; multi-day cadence uses `f_live × f_night` where `f_night = t_night / 86400`. These are separate `model_night` / `model_day` instances in the surface computation.
- **Negative t_exp:** When `f_live × t_cad / N_exp < t_overhead`, the strategy is unphysical. `t_exp_s()` and `exposure_time_s()` return `NaN` (not a negative number) so the surface correctly shows no detection rate there.

## Deferred Issues
- **Gap region for i_det = 1:** The gap `t_night/i_det ≤ t_cad < 1 day` incorrectly invalidates strategies for i_det=1. Deferred until i_det=1 support is added.

## User Preferences

### General workflow
- Physics accuracy above all else — question assumptions, read the relevant code before proposing changes
- Prefer minimal, focused changes — no over-engineering, no unsolicited refactors
- Always ask before making large architectural decisions
- Debug/version info hidden unless `GRB_DEBUG=1` env var is set (`html.Details` element, closed by default)

### UI / Design
- **Modern, sleek, professional look** — rounded corners, clean spacing, dark-by-default
- **Dark mode is the default**; light/dark toggle is implemented via `data-theme` attribute on `<html>` + CSS custom properties (`html[data-theme="dark"]` / `html[data-theme="light"]`)
- **Collapsible sidebar** (open by default) for all parameter sliders; toggle button in header
- **Slider controls:** every slider is paired with an editable `dcc.Input(type="number")` showing the current value — no tooltip popups (`tooltip` param omitted). Sync pattern:
  - Slider → input: clientside callback (instant, no round-trip)
  - Input → slider: server callback on `n_blur`, clamped to slider range
- **Parameter labels:** concise, no explanatory text in parentheses or after a dash (e.g. `"i"` not `"i (min. detections)"`, `"log A [Jy]"` not `"log(A [Jy]) — flux anchor"`)
- **Regime colors:** vibrant Material Design accent palette — warm (red/orange/amber) = flux-limited A1–A3; cool (teal/cyan/blue) = cadence-limited A4–A6; neutral gray = A7. Never use muted/desaturated colors for regimes.
- **Regime labels:** never show "A1"…"A7" to users — use descriptive phrases ("Flux-limited · decel.", etc.) or rely on semantic color alone
- **Markers:** ZTF marker: coral circle. Optimum marker: gold diamond
- **Legend:** always positioned top-left (`x=0.01, y=0.98, orientation="v"`) to avoid overlapping the colorbar
- **Colorbar:** positioned slightly inward (not at the far right edge) to avoid overlapping Plotly's toolbar icons
