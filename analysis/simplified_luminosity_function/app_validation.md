# Scale-free LF app integration: validation record

Validated 2026-10-04. The scale-free LF replaces the app's bounded LF; the bounded calculation remains available through the explicit Python reference factory.

## Executed test commands and results

Commands were run from the repository root with the existing virtual environment.

| Check | Command | Result | Log |
|---|---|---|---|
| Before engine changes | `.venv/Scripts/python.exe -B -m pytest tests --ignore=tests/test_browser_e2e.py -q` | 283 passed, 1 skipped, 1 failed; 294.64 s | `tmp/simplified_lf_app_baseline.txt` |
| Full nonbrowser regression | `.venv/Scripts/python.exe -B -m pytest tests --ignore=tests/test_browser_e2e.py -q` | 352 passed, 1 skipped, 1 failed; 287.96 s | `tmp/simplified_lf_app_tests.txt` |
| Latest expanded LF checks | `.venv/Scripts/python.exe -B -m pytest tests/test_powerlaw_luminosity_function.py tests/test_lf_bridge.py -q` | 89 passed; 39.17 s | `tmp/simplified_lf_focused_tests.txt` |
| Final bridge and optical batching | `.venv/Scripts/python.exe -B -m pytest tests/test_lf_bridge.py -q -p no:cacheprovider` | 20 passed; 16.17 s | Rechecked after batching day overlays; includes two scalar-strategy parity cases |
| Historical analyses | `.venv/Scripts/python.exe -B -m pytest analysis/simplified_luminosity_function/test_theory.py analysis/simplified_luminosity_function/test_ztf.py analysis/simplified_luminosity_function/test_pedagogical_equations.py analysis/test_lf_rate_medians.py -q` | 70 passed; 12.80 s | `tmp/simplified_lf_historical_tests.txt` |

Both full-suite runs have the same sole failure: `tests/test_qd_views.py::test_qdview_differential_integrates_to_cumulative[True]`, in the existing single-luminosity calculation. Both also report the same 28 overflow warnings in the existing fading-cap exponential expressions. No new numerical regressions were found.

The full regression run began before the last focused tests were added. The separate, final 89-test focused run covers those additions; the full-run count must not be presented as collecting every final added test.

## Scientific acceptance checks

- Independent, segmented Gauss–Legendre integration of the thesis piecewise light curve agrees with full LF rates to relative tolerance `1e-7`: slopes −2.49, −2.4, −2, −1.75, −1.25 and −1.01; cadences 300 s, 1 day and 5 days; conservative/optimistic and from-peak windows.
- Both calculation modes satisfy normalization, flux and Euclidean-radius scalings to `1e-7` or better. Reference-luminosity changes and inactive single-luminosity normalizations leave rates invariant to `1e-9`.
- Exact angular survival agrees with independent integration to `1e-7`. Exact distance survival agrees with its closed form to `1e-8`; distance medians agree to `2e-7`, including the very small median at slope −2.49 and nonzero distance floors.
- Selected-mode cumulative distributions are monotone, have the correct endpoints and reproduce their own mode's total. Numerically integrated angular and distance densities agree with the total within 0.8% and 0.3%, respectively. Median survival is one half within `2e-5`. Approximate and full selected populations are explicitly checked to differ.
- Narrow surviving intervals just below a fading cap remain positive and agree with independent hard/random-phase integration to `2e-6`. Nonconvergent slopes, empty selections and an extreme rise-filter threshold exceeding ordinary floating-point luminosity range are covered.
- Finite-cutoff references hold the differential rate normalization fixed: the finite intrinsic density is the integral of the same intensity over the retained luminosity interval, rather than a fixed normalized population. The comparison uses bounds `1e-12 L_dec` and `1e18 L_dec`, with the explicit deceleration-to-one-day luminosity conversion. Agreement is within 1% for plain and joint rise/fade selection, both rate modes where applicable, hard/random switches and both peak-window prescriptions. The plain exact case additionally evaluates both omitted tails and verifies their combined fractional contribution is below `1e-4`.
- Doubling the finite-reference angular resolution from 6,000 to 12,000 points changes rates by less than 0.2%. Tightening the adaptive full-LF angular tolerance from `1e-5` to `1e-10` preserves rates and distance medians within 1% and angular medians within 0.01.

## Interface and artifact checks

Bridge tests cover both optical/nonoptical and approximate/exact modes, normalization-only rate changes, undefined intrinsic readouts, ignored inactive controls and flux override, slices, selected-population totals, and disclosure of distance mass below the plotted range.

The LF browser regression was updated to exercise the new slope/normalization controls, actual disabling of inactive inputs, Exact switching, preset retention, and restoration of saved single-luminosity values. Final browser regression passed: `.venv/Scripts/python.exe -B -m pytest tests/test_browser_e2e.py::test_lf_toggle_end_to_end -q` (1 passed in 26.56 s). The real Chromium browser loaded Pyodide, NumPy, and Plotly successfully; there were no JavaScript or app-status errors. Manual checks covered both modes and all three presets. At alpha=-2.49 the exact distance panel shows 87.1% below its plotting range, the first survival hover shows 12.9%, and the distance median remains visible as 4.04e-15 Gpc. The final q/D guide labels were visually inspected inside the canvas. Screenshots and detailed browser records are under `tmp/browser/`, including `lf_browser_final_validation.json`.

The coordinating agent verified the updated 16-page implementation-reference PDF visually, with no overfull-box warnings, and verified byte parity for all 10 Python sources embedded in the standalone bundle. These artifact checks were performed by that agent and are recorded here from its verified results.

## Production-grid performance

With alpha=-2, A=1, joint rise/fade cuts 0.5/0.3 mag/day, and the full
200 by 160 surface, final optical calculations completed locally in 2.54 s
(approximate) and 4.19 s (Exact with peak window). Batching the LF day-overlay
medians reduced the approximate case from 19.42 s while retaining identical
optimum rate and selected-population results. These timings are local Python
measurements, not browser timing guarantees. The new batching is confined to
the scale-free LF path; both modes are checked against individual strategy
evaluations, including masked values.

Final annotation-only follow-up: q/D captions were moved into the existing top margin after visual review found the angular plateau crossing the title. JavaScript syntax passed; the app was rebuilt and both final panels were rendered and visually inspected without overlap, clipping, or browser errors. Final screenshots are `tmp/browser/lf-angle-final.png` and `tmp/browser/lf-distance-final.png`; DOM layout checks are in `tmp/browser/lf_browser_final_layout.json`.
