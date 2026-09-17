# GRB Detection Rate Project — Codex Context

## Session Continuity
At the start of project work, read [docs/codex_context.md](docs/codex_context.md).
It records the scientific model, implementation contracts, research direction,
and review findings from the Codex onboarding. Check the current branch and working
tree before relying on its dated snapshot: some local docs describe a different
branch. This handover supplements, and does not replace, the source hierarchy below.

## Project Purpose
Standalone in-browser app (Pyodide + Plotly) modeling GRB (gamma-ray burst)
afterglow detection rates as a function of survey strategy parameters. Built as a
single self-contained HTML file from `build_standalone.py` (run
`python build_standalone.py` to regenerate `grb_detection_rate.html`). Physics-first:
everything must faithfully represent the analytic model from the Bachelor's thesis
(Tier 1). The previous Dash version is frozen on `dash-legacy` — don't port changes
back unless asked. (Details: [docs/architecture.md](docs/architecture.md).)

## Source Hierarchy
1. **Tier 1 (ground truth):** Bachelor's thesis "Detection Rates of GRB Afterglows in Optical Surveys — An Analytic Approach" + the two Hebrew presentations. All code must agree with these. When in doubt about notation, formulas, or regime names, defer to the bachelor's project.
2. **Tier 2 (literature):** External papers in `sources/`. Use to learn subject; flag contradictions with Tier 1.
3. **Tier 3 (unverified notes):** "Project notes" and "Tentative Paper structure" PDFs. Treat with skepticism; cross-reference against Tier 1.

## Workflow Rules
- **Plan mode**: When in plan mode, produce a plan file and get it approved before executing *any* code changes. Do not exit plan mode without an approved plan. After exiting plan mode and receiving user approval, then implement.
- **Physics accuracy above all else** — read relevant code before proposing changes.
- **Notation**: When in doubt, rely on the bachelor's project (Tier 1) for symbols, formulas, and regime names.
- **HTML build flow**: after editing `build_standalone.py` or anything under `grb_detect/`, run `python build_standalone.py` to regenerate `grb_detection_rate.html`. The built HTML is checked in.
- Minimal, focused changes — no over-engineering, no unsolicited refactors.
- Ask before large architectural decisions.
- The Dash app on `dash-legacy` is frozen — never propose changes there unless explicitly asked.

## Hard Constraints (easy to violate — always apply)
- **Subscript notation**: in all user-visible text (HTML + Plotly strings) use `<sub>…</sub>` tags, **never** plain underscores (`t_OH`, `N_exp`, `t_cad`, …). HTML `title=""` tooltips are the only exception. Full element list: [docs/ui_reference.md](docs/ui_reference.md#subscript-notation-rule).
- **Frozen-dataclass `R_int_yr`**: baked in at class-load time with default rho/D_euc. When constructing with custom `rho_grb_gpc3_yr` or `D_euc_cm`, compute `R_int_yr` manually and pass it explicitly. Details: [docs/physics_model.md](docs/physics_model.md#frozen-dataclass-warning--r_int_yr).
- **Physics engine is read-only** unless fixing a physics bug — `grb_detect/` and `standalone_bridge.py`.
- **Figure subsystem imports, never edits, the engine**: `figures/` generates static paper/test plots by importing `grb_detect/`. To vary quantities the public API hides (e.g. `F_dec`), use the copy-on-write helpers in `figures/figlib/overrides.py` — do **not** add override hooks to the engine. Figures render via **LaTeX (`text.usetex`) by default** (needs a system TeX toolchain; auto-falls back to mathtext if absent). Visual spec + the **mandatory visual-inspection gate** every figure must pass: [docs/figure_standards.md](docs/figure_standards.md); build/run & `figlib` API: [docs/figures.md](docs/figures.md).

## Documentation Map
Read the relevant detail doc on demand:

| Doc | Covers | Read when… |
|-----|--------|------------|
| [docs/architecture.md](docs/architecture.md) | File map (HTML app + physics engine), build pipeline, repo layout, legacy Dash note | Adding/moving files, changing the build, or orienting in the codebase |
| [docs/physics_model.md](docs/physics_model.md) | Physics notation glossary, `make_rate_model()` kwargs, night-schedule survey model (f_eff, N_v, Δt_v; integer-day optical cadences), q_min/D_min filters | Touching `grb_detect/` or `standalone_bridge.py`, or any rate/physics change |
| [docs/ui_reference.md](docs/ui_reference.md) | Sidebar accordion sections, slider design rules, subscript-notation detail | Editing `web/` (template/CSS/JS) or adding/changing sliders |
| [docs/implementation_reference.tex](docs/implementation_reference.tex) | Formal derivations and piecewise A1–A7 rate formulas | Verifying or deriving the underlying math |
| [README.md](README.md) | End-user quick start (download/build the HTML) | Helping a user run the app |
| [tests/filter_verification_notes.md](tests/filter_verification_notes.md) | q_min/D_min verification checklist | Validating the detection-rate filters |
| [docs/figures.md](docs/figures.md) | Publication-figure subsystem build & usage: setup (usetex default + mathtext fallback), `figlib` API, copy-on-write `overrides` contract (F_dec), preset sync, N_exp_max gotcha, testing. Quick start: [figures/README.md](figures/README.md) | Building or running static matplotlib figures for the paper/testing |
| [docs/figure_standards.md](docs/figure_standards.md) | Figure **visual spec** (color roles, legend-placement priority, text-visibility/anti-overlap, house style, math labels) + the **mandatory visual-inspection gate/checklist** | Making, restyling, or reviewing any matplotlib figure |
| [docs/luminosity_function_derivation.tex](docs/luminosity_function_derivation.tex) | Standalone paper-style derivation of the luminosity-distribution results in thesis notation (`L ≡ L_ν(t_dec)`, threshold luminosities `L_dec, L_j, L_nr, L_i`, no `s`); the built PDF sits alongside (untracked) | Writing up or re-deriving the luminosity-function results for the paper |
