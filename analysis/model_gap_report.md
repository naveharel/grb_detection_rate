# Why the model overpredicted ZTF, what was changed, and what remains

**Companion script:** [`ztf_validation.py`](ztf_validation.py) (regenerate the numbers with
`.venv/Scripts/python analysis/ztf_validation.py`).

> **Staleness note (2026-08-14):** the tables in the body below quote the pre-`6ca135f`
> run (ε_B = 10⁻³·⁴, p = 2.5). [`validation_output.txt`](validation_output.txt) has been
> regenerated at the current defaults (ε_B = 10⁻⁴, p = 2.2) and matches the
> [distance-distribution assessment](#distance-distribution-assessment-2026-08-14) at the
> end of this report, which carries the up-to-date numbers.

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

*Pre-`6ca135f` numbers — the current-defaults rerun lives in the assessment section at
the end of this report.*

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
**Smoke-tested 2026-08-14** — see the distance-distribution assessment at the end of
this report.

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

## Distance-distribution assessment (2026-08-14)

**Question asked:** the model's predicted detected-population distances (median D) sit
well below the distances implied by the observed redshifts (project notes 14.8.26,
Table 18). Is there a simple, observationally motivated fix that preserves the Euclidean
assumption? **Decision: assessment only** — findings recorded here; no code, default, or
engine change. Predictions below come from a fresh `ztf_validation.py` run at the current
defaults (saved to `validation_output.txt` the same day); methods for the observed-side
conversion and the spread study are in "Reproduction sketch" below.

### Which distance a Euclidean D should be compared against

The model has no redshift; its D plays two roles at once — volume coordinate for the
counts (ρ·D² dD) and flux-dilution distance (F = L/4πD²). In FRW cosmology those are
*different* distances, so a yardstick must be chosen (decision 2026-08-14: report both of
the first two, comoving primary):

- **Comoving D_C (primary).** The model is rate-calibrated — ρ·(4π/3)·D_Euc³ is pinned to
  the observed all-sky rate — so D is a comoving-volume coordinate. Thesis precedent: its
  own z = 2.9 cross-check quotes "6.4 Gpc" = D_C(2.9), and the original wall
  D_Euc = 5.28 Gpc is the comoving distance of z ≈ 2.
- **Flux-equivalent d_eff (secondary).** d_eff ≡ D_L·(1+z)^{−(1+β−α)/2}: the distance at
  which a Euclidean universe receives the same flux from the same afterglow (K-correction
  + time dilation for F_ν ∝ t^α ν^β). At p = 2.2 the exponent is 0.65 in phase II
  (β = −0.6, α = −0.9) and 1.3 in phase III (α = −p) — the yardstick itself is ~±30%
  ambiguous, which bounds how precisely any Euclidean model can be graded on distances.
- **Luminosity D_L (shown for reference, not a yardstick).** The model has no (1+z)
  dimming, so comparing its D to D_L overstates the gap by construction.

### Observed side — Table 18 redshifts → distances

Flat ΛCDM, H₀ = 70 km/s/Mpc, Ω_m = 0.3 (the same anchor cosmology behind the
"5.28 Gpc ≈ z = 2" identification). Of the 13 events, 11 have redshifts; GRB 190106A is
excluded ("not part of the main sample" per the notes); AT2020sev and AT2021cwd have no z.

| event | z | cadence group | D_C [Gpc] | d_eff(II) [Gpc] | D_L [Gpc] |
|---|---|---|---|---|---|
| AT2019pim | 1.2592 | low (ToO tiling / TESS field) | 3.89 | 5.17 | 8.79 |
| AT2020blt | 2.9 | high (ZUDS 6/night) | 6.26 | 10.08 | 24.41 |
| AT2020kym | 1.256 | high (ZUDS 6/night) | 3.88 | 5.16 | 8.76 |
| AT2020yxz | 1.105 | unspecified | 3.55 | 4.61 | 7.48 |
| AT2021any | 2.5131 | high (partnership 6/night) | 5.84 | 9.07 | 20.53 |
| AT2021buv | 0.876 | low (TESS 1-day) | 2.99 | 3.73 | 5.61 |
| AT2021lfa | 1.0624 | low (public 2-day) | 3.45 | 4.45 | 7.12 |
| AT2021qbd | 1.1345 | high (partnership 6/night) | 3.62 | 4.72 | 7.72 |
| AT2023lcr | 1.0272 | high (partnership 6/night) | 3.37 | 4.31 | 6.83 |
| AT2023sva | 2.280 | low (wide nightly, ~10⁴ deg²) | 5.56 | 8.43 | 18.24 |
| (GRB 190106A | 1.861 | excluded | 4.97 | 7.18 | 14.23) |

Medians per group (flux-equivalent quoted for the phase-II / phase-III exponents):

| group (N) | med z | med D_C | med d_eff (II / III) | med D_L |
|---|---|---|---|---|
| High-cadence (5) | 1.256 | **3.88 Gpc** | 5.16 / 3.04 | 8.76 |
| Low-cadence (4) | 1.161 | **3.67 Gpc** | 4.81 / 2.91 | 7.95 |
| All main (10) | 1.195 | 3.75 | 4.94 / 2.96 | 8.24 |

The observed low- vs high-cadence medians are statistically indistinguishable
(×1.06 apart, N = 4 and 5).

### Predicted side — current defaults (fresh run, post-`6ca135f`)

| | rate × ε_cov [con, opt] | observed | q_med | D_med [Gpc] | D_90 [Gpc] |
|---|---|---|---|---|---|
| Mode A (public 2-night) | [0.99, 4.7] /yr | 2.0 | 1.17 | **0.56–1.00** | 1.41 |
| Mode B (high-cad 6/night) | [3.8, 4.6] /yr | 2.5 | 1.06 | **2.28–2.47** | 3.28 |

(D_med range = conservative i-window … optimistic (i−1)-window; q_med/D_90 at the
optimistic endpoint.) The `6ca135f` recalibration (ε_B 10⁻³·⁴ → 10⁻⁴, p 2.5 → 2.2) made
both modes flux/cadence-limited *below* the wall — D_90 = 3.28 < D_Euc = 4.55 Gpc — so
the wall no longer touches either mode's distance distribution, and the detected
νL_ν(1 d) = 5.8×10⁴³ erg/s sits ×3.3 below the observed detected median (deliberate: the
selection-bias argument of that commit, quantified below).

### Verdict

| | predicted D_med | obs D_C (ratio) | obs d_eff II (ratio) | obs d_eff III (ratio) |
|---|---|---|---|---|
| Mode A vs low-cad | 0.56–1.00 | 3.67 (×3.7–6.6) | 4.81 (×4.8–8.6) | 2.91 (×2.9–5.2) |
| Mode B vs high-cad | 2.28–2.47 | 3.88 (×1.6) | 5.16 (×2.1) | 3.04 (×1.2–1.3) |

The discrepancy is real under every yardstick — mildest for Mode B on the phase-III flux
yardstick (×1.2), severest for Mode A (×3–9). And where the observed groups are nearly
equal (×1.06), the model splits them by ×2.5.

### Why no single-luminosity parameter can fix it

In every regime the detected-distance density is p(D) ∝ D² up to a horizon H (= D_i in
the cadence/flux-limited regimes, D_Euc when wall-limited), so **D_med ≈ 0.794·H**, while
the mode rate is R ∝ ρ·q_cap²·H³. The *observed rate therefore pins the horizon*:
matching R_A ≈ 2/yr forces H_A ≈ 1.0–1.3 Gpc, hence D_med,A ≈ 0.8–1.0 Gpc. Every dial
fails:

- **Brightness (ε_B, E_k,iso, n₀, ν, p — all acting through F_dec):** moves H and R
  together (R ∝ H³ ∝ F_dec^{3/2}). Pushing Mode A's median to its wall-capped maximum
  (≈3.5 Gpc) costs ×(3.6)³ ≈ ×47 in rate → ~220/yr effective vs 2 observed.
- **Normalization (ρ, D_Euc):** the 511/yr all-sky lock fixes ρ·D_Euc³ (= R_int), and the
  physical horizons are D_Euc-invariant (F_dec ∝ D_Euc⁻² cancels exactly), so these move
  R (∝ ρ at fixed H) but never D_med. Rescuing the ×47 overshoot with ρ alone needs
  ρ ÷ 50–100 → on-axis-equivalent local rate 0.01–0.03 Gpc⁻³ yr⁻¹ (vs Wanderman & Piran
  2010's 1.3₋₀.₇⁺⁰·⁶) and, via the lock, D_Euc ≈ 16–22 Gpc — both far outside the
  literature and the Euclidean approximation's domain.
- **D_min:** raises the median trivially by excising the near volume, but models no real
  selection effect (nearby afterglows are easier to identify, not harder) — a fudge, not
  a fix.

Within the single-L model the distance medians are not free parameters — they are
consequences of the observed rates. The gap is structural Malmquist bias: the real events
seen at 3.5–6 Gpc are the bright tail of a luminosity function the model does not have.

### What a rate-locked rebalance would (and would not) do — not applied

(ρ, D_Euc) = (260, 4.55) → (**166.4 Gpc⁻³ yr⁻¹, 5.28 Gpc**) keeps the all-sky on-axis
rate at 513/yr. Verified numerically:

| | rate × ε_cov | observed | D_med |
|---|---|---|---|
| Mode A | [0.63, 3.01] /yr | 2.0 | 1.00 Gpc (unchanged) |
| Mode B | [2.41, 2.96] /yr | 2.5 ✓ | 2.47 Gpc (unchanged) |

It fixes Mode B's ×1.6 rate residual, restores the thesis wall (≈ comoving z = 2), and
keeps ρ within the Wanderman & Piran 1σ band (on-axis-equivalent 0.83 vs 1.3₋₀.₇⁺⁰·⁶
Gpc⁻³ yr⁻¹) — but, per the invariance above, **no median moves**. Recommended companion
to the F_dec spread if defaults are ever revisited; **not applied** (assessment-only
decision 2026-08-14).

### F_dec-spread smoke test (the deferred fix, quantified)

The deferred lognormal F_dec spread (see "Measured residual" above) was smoke-tested
through the existing `F_dec_override_Jy` bridge hook (contract:
`tests/test_fdec_override_bridge.py`): Gauss–Hermite 5-node quadrature over log₁₀ F_dec,
median-anchored, σ in magnitudes, final-config windows + id cuts. On the *rebalanced*
calibration (ρ = 166.4, D_Euc = 5.28):

| σ [mag] | A rate×ε_cov | A D_med | A D_90 | B rate×ε_cov | B D_med | B D_90 | detected νL_ν(1 d) |
|---|---|---|---|---|---|---|---|
| 0 | [0.63, 3.01] | 1.00 | 1.40 | [2.41, 2.96] | 2.47 | 3.28 | 5.8×10⁴³ |
| 1.0 | [1.64, 7.67] | 1.79 | 3.63 | [4.36, 4.77] | 3.16 | 4.88 | **2.0×10⁴⁴** |
| 1.4 | [4.08, 13.6] | 2.42 | 4.01 | [5.08, 5.45] | 3.30 | 4.93 | 3.3×10⁴⁴ |

(rates [conservative, optimistic] windows in /yr; D in Gpc, mixtures at the optimistic
window. Same σ on the current calibration: slightly lower, capped by the 4.55 wall —
σ = 1.4 gives A 2.33 / B 2.97, D_90 3.5 / 4.2.)

1. **The detected-median luminosity lands on target:** 2.0×10⁴⁴ erg/s at σ = 1.0 vs the
   observed 1.9×10⁴⁴ (Ho Table 5), while the population median stays 5.8×10⁴³ — the
   `6ca135f` selection-bias rationale made quantitative.
2. **Distances move most of the way:** Mode A 1.0 → 1.8–2.4 Gpc, Mode B 2.5 → 3.2–3.3;
   the spurious A:B split narrows ×2.5 → ×1.4 (observed ×1.06); D_90 reaches ~4.9–5.0 Gpc
   — the tail now extends to the wall, covering the z ≈ 2.3–2.5 events (D_C = 5.6–5.8).
   The z = 2.9 event (D_C = 6.26 Gpc) remains beyond any rate-consistent wall — a
   permanent Euclidean limitation.
3. **Rates rise ×2–7**, so σ must be fitted jointly with the median brightness (ε_B or
   the F_dec anchor) against {R_A, R_B, detected νL_ν, D_med} — two knobs against four
   observables, i.e. genuinely falsifiable. From the per-node responses the plausible
   landing is Mode A D_med ≈ 2.5–3 Gpc (still ~25–35% short of 3.67) and
   Mode B ≈ 3.3–3.6 vs 3.88.
4. **Quadrature caveat:** the top node (14–40× the median, weight 0.011) carries a large
   share of the detections — Malmquist at work, but also a warning that a production
   version needs ≥7–9 nodes and an explicit bright-end truncation, anchored to the
   Kann et al. 2010 / Cenko et al. 2009 1-day flux distributions (σ ≈ 1.2–1.5 mag).

### Reproduction sketch

Predictions: `.venv/Scripts/python analysis/ztf_validation.py` (run of 2026-08-14 saved
in `validation_output.txt`). Observed side: D_C = (c/H₀)∫₀^z dz′/E(z′) with
E = √(0.3(1+z)³ + 0.7); D_L = (1+z)·D_C; d_eff = D_L·(1+z)^{−0.65} (phase II) or
(1+z)^{−1.3} (phase III). Spread: nodes log₁₀ F_dec = log₁₀ F_dec,0 + √2·(σ/2.5)·ξ_k
with 5-point Gauss–Hermite (ξ_k, w_k/√π), evaluated by passing
`params["F_dec_override_Jy"] = s_k·F_dec,0` through `bridge._build_models` (copy-on-write;
co-scales F_j/F_nr; cache-safe); mixture statistics are w_k-weighted sums of the per-node
`dR_dD_full_integral` / `dR_dq_full_integral` arrays (grids identical at fixed D_Euc);
detected-luminosity statistics use weights w_k·R_k with L_k = s_k·νL_ν,0(1 d).

### Bottom line

The distance discrepancy is **not fixable by any well-motivated parameter choice within
the single-luminosity Euclidean model** — the observed rates pin the horizons, and the
horizons pin the medians. Three statements survive scrutiny and belong in the paper:
(i) part of the *apparent* gap is the yardstick — against comoving distance (the
appropriate measure for a rate-calibrated Euclidean model) the observed medians are
3.7–3.9 Gpc, not the 8–9 Gpc suggested by luminosity distance; (ii) most of the rest is
the luminosity function — a lognormal F_dec spread (σ ≈ 1.0–1.4 mag, literature-anchored)
moves the predicted medians most of the way, reproduces the detected-median luminosity,
and largely removes the model's spurious low-vs-high-cadence median split, at the cost of
a joint (σ, ε_B) refit of the rates; (iii) the residual ~25–35% shortfall on the
public-survey mode and the unreachable z = 2.9 event are honest, quantified limits of
Euclidean compression with a hard wall — to be stated, not fixed.
