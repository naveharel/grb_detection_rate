# GRB Detection Rate Project — Claude Context

## Project Purpose
Interactive Dash web app modeling GRB (gamma-ray burst) afterglow detection rates as a function of survey strategy parameters. Physics-first; everything must faithfully represent the underlying analytic model from the Bachelor's thesis (Tier 1).

## Source Hierarchy
1. **Tier 1 (ground truth):** Bachelor's thesis "Detection Rates of GRB Afterglows in Optical Surveys — An Analytic Approach" + the two Hebrew presentations. All code must agree with these.
2. **Tier 2 (literature):** External papers in `sources/`. Use to learn subject; flag contradictions with Tier 1.
3. **Tier 3 (unverified notes):** "Project notes" and "Tentative Paper structure" PDFs. Treat with skepticism; cross-reference against Tier 1.

## Reference Files
- [Physics model](.claude/physics-model.md) — phases, regimes, key scales, f_live architecture
- [UI & design](.claude/ui-design.md) — layout, color palette, markers, legend rules, CSS tokens
- [Technical notes](.claude/tech-notes.md) — app architecture, Plotly/Dash compatibility, debugging checklist

## File Map

**Frontend (modular package — all UI code lives here):**
| File | Role |
|------|------|
| `plot_3d.py` | Thin entry point: Dash init, register callbacks, run server |
| `components/layout.py` | Full page layout: navbar, metrics strip, sidebar, view panels |
| `components/sidebar.py` | DBC Accordion with parameter sliders and switches |
| `components/figures.py` | All Plotly figure builders (3D surface, N-slice, t-slice, metrics bar) |
| `callbacks/surface.py` | Main update callback (surface + slices + metrics) + CSV export |
| `callbacks/sync.py` | Bidirectional slider ↔ input sync (clientside + server) |
| `callbacks/ui.py` | Theme, sidebar collapse, view-mode switching, presets |
| `assets/style.css` | CSS design system: tokens, dark/light, component overrides |

**Physics engine (never modify):**
| File | Role |
|------|------|
| `grb_detect/plot_3d_core.py` | Surface computation, optical cadence grid, optimizer |
| `grb_detect/detection_rate.py` | Core `DetectionRateModel` class — piecewise rate formulas |
| `grb_detect/params.py` | Frozen dataclasses for all parameters |
| `grb_detect/afterglow_ism.py` | Derived scales: t_dec, t_j, q_dec, q_j, q_nr |
| `grb_detect/pls.py` | Power-law segment models: PLSG, PLSH |
| `grb_detect/survey.py` | Survey strategy helpers: t_exp, F_lim, sky coverage |
| `grb_detect/constants.py` | Unit conversions (DAY_S, DEG2RAD, etc.) |
| `sources/` | Physics reference documents (PDFs, presentations) |

## General Workflow
- Physics accuracy above all else — read relevant code before proposing changes
- Minimal, focused changes — no over-engineering, no unsolicited refactors
- Ask before large architectural decisions
- Debug/version info hidden unless `GRB_DEBUG=1` env var is set
