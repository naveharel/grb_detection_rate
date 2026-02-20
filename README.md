# GRB afterglow detection-rate surface (Dash)

This repository contains an interactive Dash app (`plot_3d.py`) for exploring the
analytic detection-rate surface \(R_\mathrm{det}(N_\mathrm{exp}, t_\mathrm{cad})\) for GRB afterglows.

## Local run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python plot_3d.py
```

Then open the local URL printed by Dash.

## Deploy (Render example)

This repo includes a `Procfile` so a platform like Render can run:

```bash
gunicorn plot_3d:server
```

Set the build and start commands in your hosting provider to:

- Build: `pip install -r requirements.txt`
- Start: `gunicorn plot_3d:server`
