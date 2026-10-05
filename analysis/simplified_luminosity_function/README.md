# Simplified luminosity function

Report-only research study approved on 2026-10-01. The model is an event-rate
intensity, not a normalized intrinsic luminosity PDF:

`d rho/dL = A/L_ref * (L/L_ref)**alpha`, `L_ref = 1e32 erg s^-1 Hz^-1`,
`L = L_nu(t_dec)`, `A` in Gpc^-3 yr^-1 per natural-log luminosity interval.
The all-luminosity detected rate exists for `-2.5 < alpha < -1` with zero
distance floor, finite angular cap and a luminosity-independent light-curve shape.

The final report has tracked [LaTeX source](../../docs/simplified_luminosity_function.tex).
The compiled PDF at `docs/simplified_luminosity_function.pdf` and the generated
`results/` directory are local, Git-ignored artifacts. Recreate the numerical
outputs with the commands below and compile the report using [the toolchain guide](toolchain.md).
The [approved plan](plan.md) records the original study scope and acceptance criteria.

The report was rewritten on 2026-10-04 at the user's request: explicit project
notation, step-by-step analytic derivation, general population behavior, and
the connection to the bounded LF now form the main text. It uses the existing
physical threshold `L_Euc(q)`, rather than `G`, `J_alpha`, or abbreviated exponents.
The detailed ZTF scans and numerical tables remain available here as supporting
results; the PDF now keeps one small table and two illustrative figures.

## Reproduce the numerical results

Run these commands from the repository root with the project venv:

```powershell
.venv/Scripts/python.exe -B -m analysis.simplified_luminosity_function.theory_results
.venv/Scripts/python.exe -B analysis/simplified_luminosity_function/ztf.py
.venv/Scripts/python.exe -B -m pytest analysis/simplified_luminosity_function/test_theory.py analysis/simplified_luminosity_function/test_ztf.py analysis/simplified_luminosity_function/test_pedagogical_equations.py -p no:cacheprovider -q
.venv/Scripts/python.exe -B analysis/simplified_luminosity_function/build_tables.py
.venv/Scripts/python.exe -B analysis/simplified_luminosity_function/build_figures.py
.venv/Scripts/python.exe -B analysis/simplified_luminosity_function/build_pedagogical_figures.py
```

The final command generates the two figures used in the current report. The
older table and figure builders reproduce the archived numerical study; their
TeX includes are no longer inputs to the revised manuscript. Explicit test
module paths avoid doctest collection of the original UTF-16 console log.

The theory module uses NumPy, with independent segmented Gauss-Legendre checks;
SciPy is not required. Plots use Matplotlib and `figures.figlib`. ZTF settings
are imported from the existing, unchanged `analysis/lf_rate_medians.py` study,
which imports `analysis/ztf_validation.py` and the unchanged bridge/engine.
Keep these companion analysis files when moving this study to another checkout.

`ztf.py` includes the full finite-cutoff engine comparison by default; use
`--skip-engine` only for exploratory reruns. Regenerating the original numerical study requires
the complete comparison results. Inspect `--help` for current CLI switches.
Running `build_tables.py` while that comparison is incomplete intentionally fails.

## Calculations and data

- `theory.py` contains the closed angular integrals, piecewise light curve,
  random-phase selection, detected CDFs, luminosity/angle medians and cutoff tails.
- `theory_results.py` generates 50 representative theory cases and the convergence
  and divergence examples. `results/theory.json` includes the theory test log and
  resolution checks; the accompanying CSVs expose the tables directly.
- `ztf.py` implements the all-L integral with the previous study's joint rise/fade
  kernel. It scans 1,499 slopes at 0.001 spacing and calibrates a shared amplitude.
  `results/ztf_summary.json` records the scan, source hashes and acceptance bounds;
  `ztf_highlights.json` records selected models and fixed-amplitude coverage tests;
  `ztf_verification.json` records quadrature and finite-cutoff engine comparisons.
- `build_tables.py` and `build_figures.py` reproduce the original detailed study.
- `build_pedagogical_figures.py` exports the two figures used in the revised report.
- `test_pedagogical_equations.py` independently verifies the expanded angular
  formulas, logarithmic limits, finite-cutoff flux response, conditional medians,
  and global luminosity regimes.
- `baseline_hashes.json` records protected files before the study.
  `results/validation.json` preserves the initial report's verification record.
  `results/validation_pedagogical.json` records the rewritten report's checks,
  source preservation, current tool versions, and full visual inspection.

All parameter sets are explicit in the data. The theory examples use the rounded
thesis `t_dec=19.4 s, p=2.5`. The observational benchmark uses the previous study's
`p=2.2` shape and `D_Euc=4.55 Gpc`, with both timing endpoints and coverage 0.35.

The ZTF kernel intentionally preserves the previous model's phase-at-peak
rise/fade cuts and fixed `t_p + i_eff*t_cad` window. It is distinct from the pure
cadence calculation's fully randomized detection-count threshold. The engine's
piecewise cutoff prescriptions can produce angular discontinuities at `q=2`.

## Numerical and scientific limits

The model has no finite total intrinsic rate or emissivity over `(0,infinity)`.
Medians can become extraordinarily large or small near convergence boundaries;
use logarithmic quantiles where necessary. The detected mean L diverges for
`alpha >= -2`, including the selected ZTF model.

The calibrated model passes the inherited factor-three rate and 35% distance
tolerances; these are approximate project benchmarks, not a population fit with
statistical confidence intervals. Distance distributions are necessarily identical
across survey selections homogeneous in `L/D^2`. Luminosity and angular predictions
remain separate diagnostics. The inherited observational limitations are detailed
in `analysis/lf_rate_medians_observations.md` and the saved original study results.

Finite-cutoff engine checks hold the differential amplitude fixed, transform the
spectral bounds to the engine's one-day luminosity, and rescale the finite total
rate. They do not change the engine or its immutable parameters. Rates and distance
errors are relative; angular-median differences are absolute q differences;
luminosity-median CDF and angular-fraction errors are absolute probabilities.

## Typesetting and visual verification

See `toolchain.md` for the discovered/restored compiler and rendering commands.
Compile from `docs/` so relative figure paths resolve. A conventional
installation uses:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error simplified_luminosity_function.tex
pdflatex -interaction=nonstopmode -halt-on-error simplified_luminosity_function.tex
```

Both plot builders request the repository's default LaTeX rendering. The
initial study used STIX mathtext fallback (`results/figure_rendering.json`).
For this revision, MiKTeX `latex` and `dvipng` were available and both new
figures rendered with LaTeX. The report was compiled with Tectonic. Regenerated
figures and all report pages must pass the visual gate in
`docs/figure_standards.md`; rerunning a script does not itself repeat that gate.

The app, bridge, physics engine and previous LF manuscript remain unchanged.
