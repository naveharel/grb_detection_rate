# Why the model overpredicted ZTF, what was changed, and what remains

**Companion script:** [`ztf_validation.py`](ztf_validation.py) (regenerate the numbers with
`.venv/Scripts/python analysis/ztf_validation.py`; the run used below is saved in
[`validation_output.txt`](validation_output.txt)).

## Executive summary

Evaluated at the *actual* ZTF observing modes (not the app's old `i = 10` default), the
model with its old defaults overpredicted the serendipitous afterglow rate by **×100–700**.
Three causes, in decreasing order of importance:

1. **The canonical afterglow was a bright-tail event assigned to every burst** (~×70 in
   rate for the public mode). Fixed by recalibrating single-value defaults to the *median
   detected event* (ε_B: 10⁻² → 10⁻³·⁴) and the observed all-sky prompt rate
   (D_Euc: 5.28 → 4.55 Gpc).
2. **The detection-window criterion credited detections to the pre-peak interval of
   off-axis events** (~×2, and it skewed the predicted population off-axis). Fixed by two
   new off-by-default Settings switches (`win_from_peak`, `win_i_minus_one`) whose
   combination gives the corrected criterion t₊ − t_p ≥ (i−1)·t_cad.
3. **Detection ≠ identification** (~×1.5–3 depending on mode/cadence): the existing
   s_fade/s_rise filters must be on at ZTF-like values (0.3 / 0.5 mag/day), and nominal
   footprints must be derated by an effective-coverage factor ε_cov ≈ 0.35
   (weather/moon/|b|-cut/reference losses; anchored to Ho et al. 2022's field-night audit).

With all of this, **Mode A (public survey) predicts 3.3–22 events/yr against 2 observed** —
the low (conservative-window) end matches to ×1.7 — with the detected population
on-axis-consistent (78% at q < 1.5, q_med ≈ 1.2), as observed. **Mode B (high-cadence)
remains ×5 over** and the Ho strict benchmark ×40: these probe the volume/horizon-limited
regime where *no* single luminosity value can be right (see "Measured residual" below) —
the quantified evidence for the deferred last-resort of a 1-D spread on F_dec.

## Observational targets

| Target | Value | Source |
|---|---|---|
| Mode A — ZTF-II public: ~15,000 deg², t_cad = 2 nights, m_lim ≈ 20.5, i ≈ 2 | ~2 /yr (AT2021cwd, AT2021lfa) | Ho et al. 2022 |
| Mode B — high-cadence/ZUDS: ~2,500 deg², 6 visits/night, i ≈ 6 | ~2–3 /yr (AT2020kym, AT2020blt, AT2021any, AT2021qbd) | Ho et al. 2022; Andreoni et al. 2021 |
| Strict intranight benchmark: 19,190 field-nights / 2 yr | λ = 1.04 / 2 yr expected (3 seen) | Ho et al. 2022 §4.1 |
| Viewing angles | all events on-axis-consistent; ≤1–2/13 slightly off-axis | Ho et al. 2022 §3.4; Li et al. 2025 |
| Redshifts | z = 0.876–2.9 (median ≈ 1.15) | Ho et al. 2022 Table 1 |
| Detected-event luminosity | νL_ν(1 d) ≈ (1–4)×10⁴⁴ erg/s, median 1.9×10⁴⁴ | Ho et al. 2022 Table 5 |
| Counterpart energetics | E_γ,iso = (1.4–4.8)×10⁵³ erg | Ho et al. 2022 Table 4 |
| All-sky on-axis LGRB rate | ≈ 511 /yr | Ho et al. 2022 §4.1 (Swift/BAT) |

## Diagnosis

### D0 — The mismatch, and the anatomy of the old apparent "match"

At the real Mode A configuration the old model gives **1,380/yr raw** (×690 over; ×240
even after coverage derating); Mode B gives 126/yr (×50). The thesis §3.3 "match"
(7.1/yr predicted vs 6.5/yr observed) is reproduced by the script at **9.9/yr** — but the
same configuration at the *real* i = 2 gives **333/yr**. The agreement was a coincidental
cancellation: demanding k = 10 detections and using only the 3,000 deg² footprint
suppressed the rate by roughly the same ×85 that the missing physics inflated it. The
app's old `i_det = 10` default hid the same gap (sensitivity check: i = 3 → i = 2 alone
is a ×7 rate change at Mode A).

### D1 — The canonical afterglow was a bright-tail event (dominant effect)

With ε_B = 10⁻², E = 10⁵³ erg, n = 1 cm⁻³ the on-axis light curve has
**νL_ν(1 day) = 3.05×10⁴⁵ erg/s** — ~16× brighter than the *median detected* event
(1.9×10⁴⁴, Ho Table 5) and ~30× the typical LGRB afterglow (~10⁴⁴; Racusin et al. 2011).
In the flux-limited regime R ∝ F_dec^{3/2}, so assigning this brightness to every burst
inflates rates by ~×60. This is also the structural reason no single sensible value ever
fit: matching the **rate** pulls the luminosity down toward the population median, while
matching the **detected events' properties** (E_γ,iso ≈ 2×10⁵³ median, z up to 2.9 —
beyond the old model's own detection horizon) pulls it up to the bright tail.

**Resolution (single values, per the working rule "calibrate to the median of the
detected events, since they dominate the rate integral"):** lower ε_B so the light curve
passes through the detected-median luminosity. The calibration echo:

| | νL_ν(1 d) [erg/s] | F_dec = F_ν(t_dec, D_Euc) | on-axis all-sky |
|---|---|---|---|
| old (ε_B = 10⁻², D_Euc = 5.28 Gpc) | 3.05×10⁴⁵ | 2.33 Jy (log = +0.37) | 802 /yr |
| new (ε_B = 10⁻³·⁴, D_Euc = 4.55 Gpc) | 1.82×10⁴⁴ | 187 mJy (log = −0.73) | 513 /yr |
| target | 1.9×10⁴⁴ | — | 511 /yr |

### D2 — Detection-window accounting (fixes the q-distribution)

The legacy criterion is the rectangle {q ≤ q_i} × {D ≤ D_i}: the peak must *arrive*
within i·t_cad and the peak flux must reach F_lim. But the off-axis flux is negligible
before the peak, so at the dominant corner (q_i, D_i) the true above-threshold window
t₊ − t_p is ≈ 0 while i detections were credited. Corrected criterion (both new switches
on): **t₊(D) − t_p,eff(q) ≥ (i−1)·t_cad**. The cadence cap becomes the q-dependent
D_w(q) = D(t₊ = t_p,eff + T_req), which unifies the old boundaries (t₊ = T_req gives D_i;
t₊ = t_p gives D_max(q)). Effects at Mode A: rate ×0.49, predicted q_med 1.84 → 1.27.

The two switches are deliberately separate and off by default:
- `win_from_peak` alone (T_req = i·t_cad, from the peak) is the **conservative** endpoint —
  guaranteed count of i visits inside the window;
- adding `win_i_minus_one` (T_req = (i−1)·t_cad) is the **optimistic** endpoint — i visits
  fit only at perfect phase.
The uniform-phase truth lies between them (probability clip(w/t_cad − (i−1), 0, 1) ramps
across exactly this interval), so the pair should be read as a bracket. A future P-ramp
weight would replace the bracket by its interior; not implemented in this pass.

### D3 — Identification and coverage

Real candidates must rise ≥0.5–1 mag/day from a recent non-detection and fade
≥0.3 mag/day — the existing s_fade/s_rise filters at (0.3, 0.5). At 2-day cadence they
remove ~30–40% (composing with the window settings); at sub-night cadence they are nearly
toothless in the model (the implied margins η → 1), while real intranight vetting is not —
part of Mode B's residual. Nominal footprints must also be derated: Ho's audit finds only
~1,240 deg²/night of strict-quality coverage against a 2,500 deg² nominal HC footprint.
Adopted single value: **ε_cov = 0.35** (weather ~0.7 × moon/quality ~0.8 × |b|>10° ~0.8 ×
reference availability ~0.8), applied outside the engine.

### D4 — Normalization

ρ = 260 Gpc⁻³ yr⁻¹ is the cited local rate and is kept; D_Euc is the calibration knob for
the observable all-sky prompt rate, so it moves 5.28 → 4.55 Gpc (802 → 513/yr on-axis
all-sky vs the observed ≈511).

## Changed defaults (explicit list with motivation)

| Parameter | Old | New | Physical motivation |
|---|---|---|---|
| log ε_B | −2 | **−3.4** | Puts the PLS-G on-axis curve through the median detected event, νL_ν(1 d) = 1.9×10⁴⁴ erg/s (Ho Table 5). ε_B ≈ 4×10⁻⁴ is central in broadband-modelling determinations (Santana et al. 2014; Barniol Duran 2014: 10⁻⁵–10⁻²); 10⁻² was the optimistic extreme. Slider range widened to [−5, −1]. |
| D_Euc | 5.28 Gpc | **4.55 Gpc** | All-sky on-axis LGRB rate = 513/yr vs observed ≈511/yr (Ho §4.1), with ρ = 260 Gpc⁻³ yr⁻¹ unchanged (cited local rate). 5.28 Gpc (z ≈ 2 comoving) remains documented in the slider tooltip. |
| i_det (UI default) | 10 | **2** | Matches the public-survey detected events (2 alert-stream epochs); i = 10 was unphysical and masked the rate gap. |
| E_k,iso, n₀, ε_e, p, Γ₀, θ_j, ν | — | unchanged | E = 10⁵³ erg is consistent with the detected counterparts' median E_γ,iso ≈ 2×10⁵³ for γ-efficiency ~1/3–1/2; the νL_ν calibration is degenerate along (E, n, ε_B), and ε_B is the parameter with the weakest independent prior. |
| F_ν(t_dec, D_Euc) (derived) | 2.33 Jy | 187 mJy | Follows from the above; the TEMP F_dec override slider default moves to log = −0.73. |

## Waterfall (script output)

Mode A — public 2-night (observed ≈ 2/yr); "× ε_cov" applies the 0.35 coverage factor:

| step | R_raw [/yr] | × ε_cov | q_med | D_med [Gpc] |
|---|---|---|---|---|
| old defaults, legacy criterion | 1,380 | 481 | 1.84 | 3.60 |
| recalibrated defaults | 20.0 | 7.0 | 1.84 | 0.88 |
| + window from peak | 9.83 | 3.44 | 1.27 | 0.80 |
| + id cuts (0.3/0.5) | 9.48 | **3.3** | 1.27 | 0.82 |
| + (i−1)·t_cad [optimistic] | 63.7 | 22.3 | 1.20 | 1.65 |

Window bracket with cuts: **[3.3, 22] /yr after coverage** vs 2 observed — the
conservative endpoint agrees to ×1.7; the geometric mean (8.6/yr) is ×4 over,
consistent with the un-modelled P-ramp midpoint plus an identification-margin factor
(needing ~0.5 mag above the 5σ limit for a *measurable* fade is another ~×2 in the
flux-limited regime).

Mode B — high-cadence (observed ≈ 2.5/yr):

| step | R_raw [/yr] | × ε_cov |
|---|---|---|
| old defaults, legacy criterion | 126 | 44 |
| recalibrated defaults | 43.5 | 15.2 |
| + windows + id cuts | 38.1–39.1 | **≈ 13** |

Residual **×5** — see below.

## Predicted detected population (final configuration)

| | q_med | q < q_dec | q < 1.5 | D_med | D_90 |
|---|---|---|---|---|---|
| Mode A | 1.17 | 39% | **78%** | 1.66 Gpc | 2.33 Gpc |
| Mode B | 1.23 | 35% | **75%** | 3.38 Gpc | 4.35 Gpc |

Matches the observed on-axis-dominated sample (all 13 events on-axis-consistent, with at
most 1–2 grazing/slightly off-axis — cf. 75–78% at q < 1.5 plus the strict-on-axis 35–39%;
the pre-fix model put q_med at 1.8–2.3 with a heavy tail). Mode A's D_med is on the low
side of the observed z-distribution — the known cost of the single-luminosity model with a
hard D_Euc wall (tolerated limitation; the observed z = 2.9 event lies beyond any
single-median-L horizon by construction).

## Measured residual — where a single luminosity value provably cannot fit

Mode B after every single-value fix sits at **≈13/yr vs 2.5 observed (×5)**, and the Ho
strict intranight benchmark at **20.5/yr vs 0.52 (×40)**. Both configurations are
volume/horizon-limited (Mode B: D_med = 3.4, D_90 = 4.35 ≈ D_Euc): the median-calibrated
event is detectable through most of the calibration volume, so the model counts ~100% of
bursts there. In reality the luminosity function puts half the events *below* the median —
they lose detection volume as L^{3/2} — while the bright half gains almost nothing because
its extra horizon is cut off by the D_Euc wall and by the survey's own depth. This
asymmetry cannot be absorbed by any single L: lowering it to fix Mode B (×5 needs
~−0.9 dex in F_dec, per the sensitivity table) would push Mode A's conservative endpoint
to ~×0.1 of the observed rate and the detected-event brightness ~1 mag below the events
actually seen. Ho et al.'s own MC — real light-curve *distribution* folded through real
logs — reproduces the benchmark, which is the existence proof that the distribution is
what closes this gap.

**Deferred last resort (not implemented):** a 1-D spread on the single derived quantity
F_dec (outer quadrature over ~10 nodes with a lognormal weight, σ ≈ 1.2–1.5 mag anchored
to the Kann et al. 2010 / Cenko et al. 2009 1-day flux distributions), reusing the
existing F_dec-override machinery. Expected effect: ÷3–7 in the horizon-limited
configurations, ~×1 in the flux-limited ones, and a D-distribution tail extending to and
beyond D_Euc. Revisit if/when Mode-B-type predictions matter quantitatively.

## Other known, tolerated limitations (unchanged by design)

- **Top-hat jet with sharp off-axis joining** — accepted assumption; peak assumed at t_on.
- **Hard D_Euc cutoff, Euclidean geometry, no (1+z) factors** — the D-distribution is not
  expected to match in detail; events beyond D_Euc (z = 2.9) are outside the model by
  construction.
- **PLS G only**; single θ_j and Γ₀.
- **No detection-probability ramp** — the two window switches bracket it (see D2).
- Sub-night identification cuts are structurally weak in the model (η → 1 as t_cad → 0);
  real intranight vetting efficiency is folded into ε_cov only crudely.

## Sensitivity (Mode A final config, one-at-a-time)

| variation | R_raw | ×baseline |
|---|---|---|
| ε_B −0.5 dex | 14.0 | 0.22 |
| ε_B +0.5 dex | 288 | 4.5 |
| s_rise = 1.0 | 31.0 | 0.49 |
| s_rise = 0.3 | 77.4 | 1.22 |
| no id cuts | 89.4 | 1.40 |
| i = 3 | 8.95 | 0.14 |

ε_cov ∈ [0.2, 0.5] scales all rows linearly. The two largest levers are the luminosity
normalization (ε_B) and the detection count i — exactly the two quantities that the old
defaults had at bright-tail / masking values.

## What changed in the code (this pass)

- `grb_detect/detection_rate.py`: `win_i_minus_one` / `win_from_peak` model flags,
  `_t_req_s()`, `D_from_t_plus()`, `_D_eff_window_cm()`; q-dependent window cap in the
  full integral, numerical medians, dR/dq, dR/dD; `rate_log10` routes through the
  q-integral under `win_from_peak`. Tests: `tests/test_window_toggles.py` (19 tests incl.
  bit-parity with both flags off).
- Defaults: `grb_detect/params.py` (ε_B, D_Euc), sliders/inputs
  (`build_standalone.py`, `web/template.html`): i → 2, ε_B range [−5, −1], D_Euc marks,
  s_fade/s_rise 0.5 reference marks, F_dec-override default.
- App: two Settings switches for the window settings; preset dropdown split into
  **ZTF public (2-night)** / **ZTF high-cadence (6/night)** (+ Rubin); two ZTF reference
  markers on the surface (coral circle = public, violet square = high-cadence, hidden when
  infeasible); Python preset mirror `figures/figlib/presets.py` updated
  (existing `presets.ZTF` now aliases the public mode — regenerate figures through the
  visual-inspection gate before next use).
- Docs: `implementation_reference.tex` §"Detection-Window Settings",
  `physics_model.md` (window settings + calibrated defaults), `ui_reference.md`.
