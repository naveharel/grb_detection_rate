# GRB afterglow detection-rate explorer

Interactive, browser-based explorer for the analytic GRB afterglow
detection-rate surface
\( R_\mathrm{det}(N_\mathrm{exp}, t_\mathrm{cad}) \) — the model from
"Detection Rates of GRB Afterglows in Optical Surveys: An Analytic Approach"
(Bachelor's thesis).

The active version of this project is a **single self-contained HTML file**
that runs the full physics model in the browser via
[Pyodide](https://pyodide.org). No server, no install, just open the file.

## Use it

1. Download [`grb_detection_rate.html`](grb_detection_rate.html).
2. Open it in any modern browser.

On first load the page bootstraps Pyodide (~5–10 s) and then evaluates the
detection-rate surface entirely client-side. All sliders, presets (ZTF /
Rubin LSST), and CSV export work offline.

## Build from source

The HTML file is generated from `build_standalone.py`, which zips the
`grb_detect/` physics package together with `standalone_bridge.py`,
base64-encodes the zip, and injects it (plus all sliders) into the HTML
template:

```bash
python build_standalone.py
```

Produces / overwrites `grb_detection_rate.html` in the repo root. The build
needs only the Python standard library — no third-party packages.

## Layout

| Path | Role |
|---|---|
| `build_standalone.py` | Builds the standalone HTML (template + slider helpers + zip injection) |
| `grb_detection_rate.html` | Built artifact — the app |
| `standalone_bridge.py` | Bridge exposing `compute_all(params)` etc. to the in-browser Python |
| `grb_detect/` | Pure-Python physics engine (detection-rate model, regimes, surveys) |
| `tests/` | Physics verification tests (`numpy` + `pytest` required to run) |
| `docs/implementation_reference.tex` | Implementation notes |

The physics engine in `grb_detect/` follows the bachelor's-thesis notation
and is the source of truth for everything the UI displays.

## Legacy Dash version

Earlier development used a Dash + Plotly web app served via Gunicorn. That
version is preserved unchanged:

- **Branch**: [`dash-legacy`](https://github.com/naveharel/grb_detection_rate/tree/dash-legacy)
- **Tag**: `dash-final` (the last commit shared with `main`)
- **Hosted**: <https://grb-detections.onrender.com> (Render auto-deploys
  the `dash-legacy` branch)

The Dash branch is frozen — all new features land in the HTML app on
`main`.
