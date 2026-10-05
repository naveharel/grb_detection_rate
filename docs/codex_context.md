# GRB project handover for future Codex sessions

Reviewed 2026-09-07 on `main`, commit
`f06998fd23fe6906edf415ad2093492ba562dba9` (LF performance work).
This is a dated onboarding record, not a new scientific authority or a claim that
every equation has been independently validated. Recheck code and Git state when
using it. The user authorized saving important findings for future sessions.
Update confirmed facts as work changes; keep unresolved issues explicit.

## Purpose and source map

The product is an interactive analytic model of serendipitous GRB afterglow
detection rates and survey optimization. The scientific purpose is understandable
scalings and physical intuition, with order-of-magnitude predictive ambitions.
An orphan afterglow has no detected prompt GRB; this includes missed on-axis
prompt emission as well as off-axis events. These terms are not interchangeable.

The hierarchy in [AGENTS.md](../AGENTS.md) controls the work:

- **Tier 1:** the 29-page bachelor's thesis, *Detection Rates of Gamma-Ray Burst
  Afterglows in Optical Surveys - An Analytic Approach*, in `sources/`, and two
  Hebrew presentations. The 66-slide research presentation covers rate
  construction; the 41-slide advanced-project presentation develops synchrotron
  spectra and radiation transformations.
- **Tier 2:** local Sari/Piran/Narayan (1998), Sari/Piran/Halpern (1999),
  Granot/Sari (2002), Nakar/Piran/Granot (2002), and broader references.
  Their subject map is synchrotron spectra, spreading jets, spectral breaks,
  orphan detectability, and relativistic/Newtonian dynamics.
- **Tier 3:** `sources/Project notes 14.8.26.pdf` (54 pages) and
  `sources/Tentative Paper structure.pdf` (3 pages). Useful extensions and plans
  coexist with contradictory labels, units, and unfinished derivations.
  Implementation/manuscript documents are working references, not Tier 1.

Thesis roadmap: light curves/normalization through Eq. 24; peak times/fluxes
Eqs. 32-34; geometry/rate integral Eqs. 37-39; full-time survey Eqs. 45-53;
detection probability Eq. 60; cadence horizons Eqs. 63-64; repeated detections
Eqs. 69-77; historical ZTF comparison Eq. 78. Project notes extend toward
finite-exposure averaging (pp. 36, 40-48), optimization/overhead (pp. 49-51),
and an observational sample (p. 52).

[analysis/model_gap_report.md](../analysis/model_gap_report.md) explains the
luminosity-distribution motivation: lowering one representative burst's
brightness lowers both rates and detection distances, making it hard to match
both observations. A distribution separates the numerous faint population from
the bright, selection-biased detected sample. The latest commits implement this
extension and accelerate its full-integral and median calculations.

Historical validation is parameter- and sample-dependent. Do not present
`analysis/validation_output.txt` as a fresh run. `PHYS_OLD` in
`analysis/ztf_validation.py` copies the newer dictionary and changes only some
entries; it is not a complete restoration of thesis parameters. Its coverage
efficiency is an external multiplier, not engine physics.

## Checkout and branch traps

Onboarding began with modified `web/template.html` and `grb_detection_rate.html`:
the LF toggle moved to Settings, with dependent parameters/derived values under
the revealed “Luminosity function” Parameters group. Preserve IDs and follow
established toggle behavior. `AGENTS.md`, the LF TeX draft, and implementation/LF
PDFs were already untracked. Onboarding adds this note and the startup link,
without altering app, engine, tests, or manuscript.

**The ignored local `docs/physics_model.md` and `docs/ui_reference.md` describe
night-schedule features from `schedule-cadence` that do not exist on `main`.**
This checkout has no `grb_detect/schedule.py`, `NightSchedule`, `schedule_on`,
`N_v`, or `dt_v_s` factory API. `schedule-cadence` and `dtv-window` are separate
experimental branches. `dash-legacy` is frozen; do not port changes there.

`sources/`, `CLAUDE.md`, `.claude/`, and local `architecture.md`,
`physics_model.md`, `ui_reference.md` are Git-ignored. They may be missing from
another checkout or retain context across branch switches. Remote-tracking refs
were inspected locally, not refreshed over the network.

## Scientific model and code map

The rate calculation assumes a top-hat double jet, uniform ISM, a Euclidean
sphere with uniform source density/random orientation, and PLS G throughout the
modeled optical light curve. Off-axis emission turns on sharply at the peak;
late Newtonian emission and spectral-break crossings are excluded from rates.
A cosmology helper and `PLSH` class exist, but do not imply a validated
cosmological or general multi-PLS rate calculation.

| Layer | Source and responsibility |
|---|---|
| Parameters/units | `grb_detect/params.py`, `constants.py`; frozen containers |
| Dynamics/spectrum | `afterglow_ism.py`, `pls.py`; deceleration, jet break, flux |
| Rates/distributions | `detection_rate.py`; seven regions, q integral, cuts, LF, medians |
| Model operations | `core.py`, `survey.py`; cached factory, grids, optical admissibility, optimizer |
| Python/JS boundary | `standalone_bridge.py`; models, surfaces, points, slices, q/D views |
| Browser | `web/template.html`, `web/styles.css`, `web/app.js` |
| Packaging | `build_standalone.py`; sliders, source bundle, generated HTML |

\[
q=\theta_{\rm obs}/\theta_j,\quad \widetilde q=q-1,\quad
q_{\rm dec}=1+(\Gamma_0\theta_j)^{-1},\quad q_j=2,\quad
q_{\rm nr}=\sqrt{2}/\theta_j .
\]

The orientation measure is **\(q\,dq\)**, not
\(\widetilde q\,d\widetilde q\). The outer angle ensures
\(\theta_j^2q_{\rm nr}^2/2=1\) in the approximate all-sky normalization.
Distinguish four dynamical phases, four full-time survey ranges, and seven
implementation regions A1-A7.

For PLS G, spectral slope is \(-(p-1)/2\); on-axis temporal slopes are
\(-3(p-1)/4\) before the jet break and \(-p\) afterward.
\(t_j=t_{\rm dec}(\Gamma_0\theta_j)^{8/3}\). Peak time is \(t_{\rm dec}\)
on-axis, then \(t_j\widetilde q^{8/3}\), then \(t_j\widetilde q^2\).
`F_dec_Jy`, `F_j_Jy`, and `F_nr_Jy` must co-scale under brightness-only changes.

The flux horizon satisfies
\(D_{\max}/D_{\rm Euc}=\sqrt{F_p(q,D_{\rm Euc})/F_{\rm lim}}\);
the cadence horizon satisfies \(t_+(D_i)=T_{\rm req}\).
With hard cadence selection and no cuts,

\[
R_{\ge i}=f_\Omega\theta_j^2 R_{\rm int}\int_0^{q_{\rm nr}}q\,dq\,
\min\left(D_{\max}(q)/D_{\rm Euc},D_i/D_{\rm Euc},1\right)^3,\qquad
R_{\rm int}=\frac{4\pi}{3}\rho_{\rm GRB}D_{\rm Euc}^3 .
\]

The dominant approximation uses a corner rectangle:
\(R\simeq f_\Omega\theta_j^2R_{\rm int}q_\star^2
(D_\star/D_{\rm Euc})^3/2\).
A1 is saturation; A2/A3 are Euclidean-distance limited with flux setting the
angle; A4/A5/A6 are cadence-distance limited; A7 is the on-axis flux-limited
case. Ordering \(q_{\rm Euc},q_i\) and the special angles determines the region.
**`A0` is the valid-strategy mask**, despite some stale prose.

`q_min` removes smaller viewing angles; `D_min_cm` excludes actual nearby
sources. In a rectangle use the positive angular difference
\(q_\star^2-q_{\min}^2\) and positive radial-shell volume. Its conditional medians
follow square/cube averages of bounds. Full-integral/LF medians must come from
the detected distribution; averaging component medians is wrong.

### Defaults versus thesis

Current library defaults include \(p=2.2\), \(\epsilon_B=10^{-4}\),
\(D_{\rm Euc}=4.55\) Gpc, \(\rho=260\ {\rm Gpc}^{-3}{\rm yr}^{-1}\),
\(E_{\rm k,iso}=10^{53}\) erg, \(n=1\), \(\theta_j=0.1\),
\(\Gamma_0=10^{2.5}\), \(\epsilon_e=0.1\). They give
\(t_{\rm dec}\simeq19.4\) s, \(t_j\simeq2.24\) days, and
\(F_{\rm dec}\simeq0.00895\) Jy at \(5\times10^{14}\) Hz.
Thesis fiducials use \(p=2.5\), \(\epsilon_B=10^{-2}\), about 5.28 Gpc,
and \(F_{\rm dec}\simeq2.3\) Jy. State parameter sets in comparisons.

Library survey defaults include `f_live=0.1` and a 27,500 square-degree maximum;
UI defaults use `f_live=0.2` and approximately the full sky. Presets change
instrument/count settings, not every scientific parameter or user constraint.

**Frozen-dataclass trap:** direct construction with custom `rho_grb_gpc3_yr` or
`D_euc_cm` does not recompute `R_int_yr`. Calculate and pass it explicitly.
`core.make_rate_model()` does this correctly.

## Survey, window, and filter semantics on main

\(N_{\rm exp}\) counts pointings/fields per repeat cycle; \(i\) counts required
detections of one event. \(t_{\rm exp}=f_{\rm live}t_{\rm cad}/N_{\rm exp}
-t_{\rm OH}\), and \(F_{\rm lim}=A(t_{\rm exp}/30\,{\rm s})^{-1/2}\).
Overhead approximation evaluates zero model overhead but still enforces actual
overhead feasibility for the surface and optimizer.

Optical mode dispatches between day and night models. Subday cadence uses
\(f_{\rm live}/f_{\rm night}\) internally and multiplies rates by
\(f_{\rm night}=t_{\rm night}/{\rm day}\). It permits
\(t_{\rm cad}<t_{\rm night}/i\), excludes the gap up to one day, and permits
only integer-day multiples above it. Arbitrary cadences are not rounded to days.
The UI enforces \(f_{\rm night}\ge f_{\rm live}\). This is not the separate
branch's unified night-schedule model.

“Exact rate mode” is numerical full q integration within these assumptions,
not exact astrophysics or automatic use of the full thesis probability
\(P_{\ge i}(\tau)=\operatorname{clip}(\tau-(i-1),0,1)\).
The thesis treats single detections separately; `main` has an unresolved
optical single-detection gap TODO.

- Engine defaults: `win_i_minus_one=False`, `win_from_peak=False`.
  Bridge default: `win_iminus1=True`. The UI checkbox “Detection window spans
  i·t_cad” is inverted into `win_iminus1`.
- `win_from_peak` requires \(t_+\ge t_{p,\rm eff}+T_{\rm req}\), making the
  horizon q-dependent and routing rate evaluation through integration.
  The on-axis effective peak time must be at least \(t_{\rm dec}\).
- First enabling exact mode initializes its correction suboptions; later
  choices are independent. Inspect `readParams()` to reproduce a UI state;
  `full_integral=True` alone is insufficient.
- The fading cut is normally a discrete mag/day difference across
  \((i-1)t_{\rm cad}\), bypassed for \(i=1\). Random first-observation timing
  adds survival weighting beyond a best-case angular cap.
- The rise cut models a previous nondetection and contrast
  \(\eta=10^{s_{\rm rise}t_{\rm cad}/(2.5\,{\rm day})}\), reducing the horizon.
  Rise/fade share the **same** random start; use joint survival, not independent
  probabilities. Exactly zero explicitly bypasses a cut.
- Strong-rise dominant rates include the larger of angular and on-axis
  lower-bound contributions. Dominant q/D views retain a simpler rectangle;
  full-integral views are appropriate for this corner. There is no universal
  full-versus-dominant ordering under every option.

## Luminosity function and paper notation (updated 2026-10-04)

The app LF setting now uses `PowerLawLuminosityFunction`, with
`dR/dL = A/L_ref (L/L_ref)^alpha` for all positive luminosities,
**L = L_nu(t_dec)** in erg/s/Hz, fixed L_ref = 1e32, and defaults A = 1
Gpc^-3 yr^-1 and alpha = -2. The strict convergence interval is
-2.5 < alpha < -1. A is the rate density per natural-log luminosity at
L_ref; this intensity has no finite intrinsic total or population median.
The bridge parameters are `lf_on`, `lf_alpha`, and `lf_log10_A`.

The current derivation is [Simplified Luminosity Function](simplified_luminosity_function.tex).
Keep physical luminosities and explicit thesis notation in paper-facing text.
The earlier standalone bounded-LF manuscript remains a separate reference.
`make_finite_cutoff_rate_model` is the explicit Python reference constructor
for bounded comparisons; its luminosity remains nu L_nu(1 day), so convert
bounds and differential normalization before comparing it with the app model.
The app exposes no bounded-LF controls.

The new `scale_free_lf.py` kernels integrate luminosity analytically from
zero to infinity. Exact mode also integrates distance analytically and uses
adaptive, breakpoint-aware angle integration. Existing light curves and
peak-window/joint rise-fade selection semantics are retained. Approximate
mode integrates dominant rectangles and keeps each base interval's selected
angular/on-axis branch fixed for distribution queries. All app rates,
medians, slices and q/D views follow the selected mode; never average
component medians. A requested peak-time-dependent window keeps the full
integral fallback. Regime color means the dominant integrated contribution.

For full-mode detections with D_min = 0,
`P(D < d) = (d/D_Euc)^(2alpha+5)` and
`D_med = D_Euc * 2^(-1/(2alpha+5))`. For nonzero D_min use the corresponding
power difference. Slopes near -2.5 can yield extremely small medians; do not
use a coarse linear-distance histogram or normalize by a displayed grid's
first point. q/D plot totals cover the entire selected population; the UI
indicates appreciable probability below the plotting range.

The LF switch remains in Settings and alpha/log A controls in Parameters.
Rho, frequency, epsilon_e, epsilon_B and the deceleration-flux override are
inactive (and disabled) while LF is enabled. Luminosity is defined at the
current band, so the PLS-G frequency amplitude cancels. Intrinsic counters
and the single-burst brightness readout are hidden; their controls restore
on LF exit. Exact remains selectable. Pure-amplitude and sensitivity
invariance of normalized distributions permits reuse across exposure counts
at fixed cadence.

## Build, browser, and figure contracts

`build_standalone.py` embeds eight engine files plus the bridge as a base64 zip,
then combines template, CSS, JS, and slider markup. The checked-in HTML must
accompany relevant source changes. It loads Pyodide 0.27.0, Plotly 2.35.2, and
fonts from CDNs: a cold start requires network access. “Standalone” does not
mean fully offline with an empty browser cache.

`compute_all`, `compute_nslice`, `compute_tslice`, and `compute_qdview` return
flattened payloads, converting nonfinite values to `None`. Display floor is
0.01/year; raw surface values are retained for export. Markers, hover fields,
metrics, slices, and surfaces must share model/filters and optical dispatch.
Markers below the floor are hidden. ZTF reference points use sidebar physics,
not a hidden preset model: public 15,000/47 pointings at two days; HC 2,500/47
at 0.98 times the night length divided by six. Pointings are clipped to the
allowed maximum; current i can make HC infeasible.

Preserve UI IDs and use `<sub>` for visible HTML/Plotly subscripts. Follow
Settings-toggle/Parameters-dependent-block conventions. The UI debounces full
and slice updates, but Pyodide calculation is synchronous on the browser thread.

Figures import the read-only engine. Read [figures.md](figures.md) and
[figure_standards.md](figure_standards.md), use copy-on-write helpers in
`figures/figlib/overrides.py`, and never mutate cached models.
LaTeX rendering defaults on with mathtext fallback; PNG is default, PDF opt-in.
Every created figure needs visual inspection at its intended publication size.
**Parity caveat:** `build_model_from_preset()` returns a day model with library
window defaults; it does not automatically reproduce the bridge's i-minus-one
window or subday optical behavior.

## Review findings for follow-up

Onboarding made no physics or manuscript fixes.

1. **Reproduced test failure:** full-mode differential R(D) does not integrate
   back to its cumulative curve within tolerance.
   `tests/test_qd_views.py::test_qdview_differential_integrates_to_cumulative[True]`
   gives 138.069344/year versus 134.684903/year (2.5129%, tolerance 2%).
   `_compute_qdview_sweep` resamples 200 linear internal points onto 100 log
   display points. Using 400 display points on that same profile reduces the
   integral mismatch to 0.3508%; 1600 gives 0.0461%. There is internal-grid
   error too: its integral is 134.685555/year versus the q-rate 136.247199/year;
   1600 internal distance points give 136.321914/year. This supports a
   discretization diagnosis for this fixture, not global accuracy.
2. **Thesis/manuscript algebra needs author review.** The LF draft already flags
   Eq. 45's sign, Eq. 69's exponent, and reversed project-notes Table 17 headers.
   A further mismatch in thesis Eqs. 73-77 is that
   \(x^{-3p/2}(\sqrt{x}+1)^2\) tends to \(x^{1-3p/2}\), while the printed
   asymptote is \(x^{-3(p-2)/2}\), differing by two powers. The LF draft repeats
   it in its cadence comparison: at p=2.5 the algebraic single-burst slope is
   -2.75, not -0.75. Its claim of LF steepening to -1.5 must be revisited.
   Distinguish the unsimplified equations; do not silently rewrite Tier 1.
3. **LF dominance prose mixes densities per L and per log L.** Endpoint
   dominance follows \(L\phi(L)R(L)\). In unsaturated phase III this scales as
   \(L^{\alpha+1+1/p}\); clean bright-side convergence toward the knee requires
   \(\alpha<-1-1/p\), not all \(\alpha<-1\). Finite bounds and constant/cross
   terms matter. Use segment integrals over the broad three-category summary;
   recheck “rises/falls” wording and claimed parameter ranges before publication.
4. **Other stale prose:** thesis Eq. 24 prints a positive PLS-G frequency
   exponent, inconsistent with its spectrum and the engine's negative one.
   The rate-figure docstring says cadence-limited rate is brightness-independent;
   that describes its angular median, not the volume. At 0.1, 1, and 10 times
   fiducial F_dec, read-only figure helpers gave rates 0.161276, 5.099983,
   161.275630/year while q median stayed 1.65113.
   `tests/filter_verification_notes.md` has old Dash commands/checklists and a
   final D-min interpretation conflicting with actual radial-shell selection.
5. **Unimplemented directions:** finite-exposure flux averaging, general
   cosmological rate integration, and the separate schedule model are not
   features of this checkout. Their scientific value does not authorize an
   architectural expansion during unrelated work.

## Verification and environment

On 2026-09-07 this completed in 311 seconds:

```powershell
.venv/Scripts/python.exe -B -m pytest tests --ignore=tests/test_browser_e2e.py -p no:cacheprovider -q
```

**271 passed, 1 skipped, 1 failed**, with 28 `expm1` overflow warnings at
extreme fading parameters in exact-mode crash tests. The failure is detailed
above; do not claim an entirely green baseline.

Read-only build verification decoded the embedded zip and matched all nine
Python files byte-for-byte to sources. Reassembling template/CSS/JS/slider
substitutions in memory also matched the current HTML exactly, without
regeneration. A browser connection was unavailable; no live browser QA was
completed. Browser E2E tests were excluded (their fixture rebuilds HTML).
PDF text and selected rendered thesis pages were inspected; literature was
mapped/sampled, not audited line by line.

Use `.venv/Scripts/python.exe`. Bare `python` currently resolves to
`C:\Python314\python.exe` without scientific dependencies. The project venv has
NumPy, pytest, matplotlib, pypdf, Pillow, and Playwright. On Windows use
`Get-Content -Encoding utf8` for documents and `rg -g '*.py' directory` rather
than assuming shell wildcard expansion.

When outputs are needed:

```powershell
.venv/Scripts/python.exe build_standalone.py
.venv/Scripts/python.exe figures/fig_qmedian_vs_Fdec.py
```

For the LF derivation, run `pdflatex -interaction=nonstopmode -halt-on-error
luminosity_function_derivation.tex` twice from `docs`. The pgfplots custom
function key is `/pgf/declare function`, not `/pgfplots/declare function`.
Treat build success, numerical agreement, and visual correctness separately.
