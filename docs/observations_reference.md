# Observations Reference — ZTF GRB-Afterglow Sample

Tier-2 reference (per [`../CLAUDE.md`](../CLAUDE.md)'s source hierarchy): the 13-event ZTF
afterglow/candidate sample originally compiled as **Table 18** of the Tier-3 "Project notes"
PDF (`sources/Project notes 14.8.26.pdf`, p. 52), cross-verified here directly against the six
primary papers it cites (arXiv HTML/PDF, read in full — not secondhand summaries). Built
2026-08-29 to support the schedule-cadence model review; see
[`../analysis/model_gap_report.md`](../analysis/model_gap_report.md) and
[`../analysis/simple_mode_check.py`](../analysis/simple_mode_check.py) for the modeling side.

**This document supersedes Table 18 as the citable source for per-event facts** — three
corrections were found in the process (see "Corrections" below). Distances (D_C/d_eff/D_L from
redshift) are **not** recomputed here; see the existing table in `model_gap_report.md` (lines
284–313) and link back to it.

## Bibliography

| Ref | Paper | arXiv | Events covered here |
|---|---|---|---|
| Ho+22 | Ho, Perley, Yao, Svinkin, de Ugarte Postigo, R. A. Perley, Kann, Burns, Andreoni, Bellm, et al. 2022, "Cosmological Fast Optical Transients with the Zwicky Transient Facility: A Search for Dirty Fireballs" (submitted ApJ) | [2201.12366](https://arxiv.org/abs/2201.12366) | AT2020kym, AT2021cwd, AT2021lfa, AT2021qbd (+ Table 5/§4.1 population stats) |
| Ho+20c | Ho et al. 2020, "ZTF20aajnksq (AT2020blt): A Fast Optical Transient at z≈2.9 With No Detected Gamma-Ray Burst Counterpart" | [2006.10761](https://arxiv.org/abs/2006.10761) | AT2020blt (actual discovery paper — **not** Ho+22, see Corrections) |
| Andreoni+21 | Andreoni, Coughlin, Kasliwal, et al. 2021, "Fast-transient Searches in Real Time with ZTFReST..." ApJ 918, 63 | [2104.06352](https://arxiv.org/abs/2104.06352) | AT2020sev, AT2020yxz, AT2021buv (+ HC survey description) |
| Perley+25 | Perley et al. 2025, "The Luminous, Slow-Rising Orphan Afterglow AT2019pim as a Candidate Moderately Relativistic Outflow," MNRAS (doi:10.1093/mnras/staf125) | [2401.16470](https://arxiv.org/abs/2401.16470) | AT2019pim |
| Li+25 | Li, Ho, Ryan, Perley, et al. 2025, "The Nature of Optical Afterglows Without Gamma-ray Bursts: Identification of AT2023lcr and Multiwavelength Modeling" | [2411.07973](https://arxiv.org/abs/2411.07973) | AT2023lcr (new), AT2021any, AT2021lfa (jet-fit only; photometry cited from Ho+22) |
| Srinivasaragavan+25 | Srinivasaragavan et al. 2025, "Multi-Wavelength Analysis of AT 2023sva: a Luminous Orphan Afterglow With Evidence for a Structured Jet," MNRAS (doi:10.1093/mnras/staf290) | [2501.03337](https://arxiv.org/abs/2501.03337) | AT2023sva |

**Not independently fetched**: GRB190106A's sources (Andreoni et al. 2020 / Zhu et al. 2023) —
that event is already excluded from the main sample in `model_gap_report.md`; AT2020blt's Ho+20c
paper (identified but not separately deep-dived — Ho+22 only adds a Keck host-galaxy
non-detection for this event, no new photometry).

## Survey-mode facts (corrected against the primary sources)

- **Public ZTF-II all-sky**: 15,000 deg² in g and r every 2 nights (Ho+22, p.4, exact quote).
  Realized per-night pattern in the two events checked (AT2021cwd, AT2021lfa): **one g point +
  one r point**, ~1.1–1.9 h apart — not two same-filter visits.
- **High-cadence partnership**: g+r exposures **6 times per night**, **nightly** cadence
  (Andreoni+21, §II, exact quote) — confirmed. **Footprint ≈3,000 deg², not 2,500 deg²**
  (Andreoni+21 states this explicitly) — see Corrections #1. i-band is not part of the nominal
  survey design per this paper (occasional i-band points come from other programs). Realized
  per-night visit counts in the tables actually inspected: mostly 2–5 sub-nightly epochs, not
  the full nominal 6 (weather/scheduling losses — consistent with the coverage-efficiency
  factor `EPS_COVERAGE` already used in `ztf_validation.py`).
- Same telescope (P48) confirmed for both modes (all rows in every table checked are
  "P48+ZTF").

## Calibration-target honesty

`ztf_validation.py`'s docstring and `analysis/model_gap_report.md` treat "~2/yr" (public) and
"~2.5/yr" (high-cadence) as observed annual detection-rate targets. **Neither is actually a rate
the source papers state:**
- Ho+22 (p.4) says only that **2 of 10** total ZTF afterglows *to date* (as of that paper, ZTF's
  first ~3–4 years of operation) came from the public survey — a cumulative count, not a rate.
- Andreoni+21's abstract reports **3 new GRB-associated transients in 13 months** of ZTFReST
  operation (≈2.8/yr by naive extrapolation) — the "~2.5/yr" figure is this project's own
  extrapolation from that, not a number Andreoni+21 states as a rate.

This doesn't necessarily invalidate the working targets, but they should be documented as
project-derived assumptions, not literature-quoted rates. **Flagged as an open question — not
changed here** (see Corrections #2).

## Verified benchmark numbers (no change needed)

- **Ho+22 Table 5** median νLν(1 day) = **1.9×10⁴⁴ erg/s**, n = 9 events (iPTF14yb, AT2019pim,
  AT2020blt, AT2020kym, AT2020yxz, AT2021any, AT2021buv, AT2021lfa, AT2021qbd) — exact match to
  `docs/physics_model.md`'s calibration citation.
- **Ho+22 §4.1 strict MC benchmark** — exact: 19,190 field-nights, 35,171 unique observations
  (from the 2020–2021 ZTF log via `ztfquery`; criteria: limiting mag > 20, ≥2 same-night
  r-band obs, an r-band obs the previous night also < 20 mag limit, |b| > 15°). Detection
  criterion: first detection 0.5 mag above the limiting magnitude, plus a same-night pair with
  0.02 d < Δt < 0.6 d (≈29 min–14.4 h). Result: **λ = 1.04 per 2 yr** (Fig. 12), P(≥1) = 60%.
  Matches `ztf_validation.py`'s benchmark mode exactly.
- **All-sky on-axis LGRB rate = 511/yr** — exact quote (Ho+22 §4.1: from a mean Swift/BAT rate
  of 73 LGRB/yr, corrected for field-of-view/duty-cycle "in the same way as Cenko et al. 2013").
  This is the number `docs/physics_model.md` uses to calibrate D_Euc = 4.55 Gpc — now confirmed
  against the primary source rather than a note-of-a-note.
- Redshift range 0.876–2.9 across the Table-5 sample — confirmed (min AT2021buv, max AT2020blt).
  Median comoving distance (~3.7–3.9 Gpc, used in `model_gap_report.md`) is **not** a number
  Ho+22 states — the paper never uses comoving distance (only z and D_L via flat ΛCDM,
  H₀=67.7, Ωₘ=0.307); the D_C figures are this project's own derivation from the redshifts,
  which `model_gap_report.md` already correctly attributes as such.

## Per-event table

q-estimate uses this project's q ≡ θ_obs/θ_j convention; for structured/Gaussian jets the
paper's core width θ_c stands in for θ_j (an approximation, noted per row). "—" = not
independently constrained in the fetched source.

| Event | z | Cadence group | Discovery survey | q-estimate (source) | Distances |
|---|---|---|---|---|---|
| GRB190106A | 1.861 | excluded (not main sample) | — | — | see model_gap_report.md (excluded row) |
| AT2019pim | 1.2592 | GW ToO tiling (not periodic) | S190901ap tiling | ≈1.4 (Perley+25 Model C, illustrative, degenerate w/ on-axis low-Γ) | D_C 3.89 Gpc |
| AT2020blt | 2.9 | high (ZUDS 6/night) | — (Ho+20c, not independently fetched) | — | D_C 6.26 Gpc |
| AT2020kym | 1.256 | high (ZUDS 6/night) | ZUDS | — (not reported) | D_C 3.88 Gpc |
| AT2020yxz | 1.105 | unspecified | daily-morning ZTFReST scan | — (not reported) | D_C 3.55 Gpc |
| AT2020sev | unknown | low-cadence (per notes) | retrospective/science-validation | — (not reported) | no z |
| AT2021any | 2.5131 | high (partnership 6/night) | — | ≈0.5 (Li+25 free-t0 fit, on-axis-consistent) | D_C 5.84 Gpc |
| AT2021buv | 0.876 | TESS-shadowing 1-day | real-time fast-transient filter | — (not reported) | D_C 2.99 Gpc |
| AT2021cwd | unknown | Public ZTF-II all-sky, 2-day | public alert stream | — (not reported) | no z |
| AT2021lfa | 1.0624 | Public, 2-day | public alert stream | ≈0.5 (top-hat) / ≈0.75 (Gaussian) / 2.0 (off-axis alt.) — genuinely degenerate (Li+25) | D_C 3.45 Gpc |
| AT2021qbd | 1.1345 | high (partnership 6/night) | — | — (not reported; **Li+25 does not cover this event — see Corrections #3**) | D_C 3.62 Gpc |
| AT2023lcr | 1.0272 | high (partnership 6/night) | GOTO-bracketed onset | 0.0 (Li+25, fixed on-axis top-hat) | D_C 3.37 Gpc |
| AT2023sva | 2.280 | wide nightly (~10⁴ deg²) | routine alert-stream (2-day gap) | ≈1.17 (Srinivasaragavan+25, structured jet posterior) | D_C 5.56 Gpc |

Distances (D_C, comoving, primary yardstick) copied from `model_gap_report.md:290-302` for
convenience only — that table is the source of record.

### Per-event detail

**AT2019pim** (Perley+25) — GW-triggered ToO tiling the night after the S190901ap alert; "all
candidates detected during the night with no previous history were scanned by eye" (a single
deep pass, not a routine multi-night revisit program). Last non-detection g>20.60 (MJD
58727.3161); first detection g=20.04±0.16 at MJD 58728.1798 (~0.86 d later), r=19.45±0.11 ~1.2 h
after that. Followed by GIT/LT/Keck/WHT over weeks. Jet models are explicitly degenerate: the
paper "cannot distinguish between an on-axis outflow with an intrinsically low initial Lorentz
factor versus... the high-latitude component of a structured jet seen partially off-axis" — no
formal posterior uncertainties on θ_j/θ_obs.

**AT2020kym** (Ho+22) — GRB200524A (Fermi-LAT), ZUDS high-cadence. Discovery MJD 58993.2863,
r=17.35±0.04 (bright), 1.8 h after the GRB trigger. **First night: 5 detections, all r-band,
within 29 min**, fading 0.35 mag over that span (~18 mag/day — very fast). Next epoch ~1 day
later shows a ~4 mag drop (g=21.60, r=21.33). Cleanest example of a genuine same-night
multi-visit intranight catch.

**AT2020sev** (Andreoni+21 Table 3) — possible counterpart to GRB200817A (tentative
association). First ZTF detection 2020-08-18 05:19 UT, r=19.14±0.05, after two prior
non-detecting nights (~1–1.5 h same-night pairs). Continued nightly through 2020-08-23 (~1.3 mag
faded in r over the first 2 days), 2–3 sub-nightly epochs per night. Only 3 total i-band epochs
(confirms Table 18's "sparse i-band coverage"). **Discovery was retrospective**: found during
science validation ~10 days after the first detection was already in the data — the paper notes
that in real-time operation this source would only have been flagged "after the second night
post-discovery," not the "second visit of the night" phrasing in the internal notes.

**AT2020yxz** (Andreoni+21 Table 4) — GRB201103B. Several non-detecting nights (Oct 29–Nov 3),
then first table-detection 2020-11-04 08:47 UT g=19.45±0.13, r=19.23±0.11 ~1.7 h later. Faded
~0.7 mag in g over ~0.9 days. **Flagged by the ZTFReST pipeline on 2020-11-05** — about 2 days
after the data already showed a detection (the paper's own narrative text and Table 4 disagree
on the literal "first detection" timestamp by one day, likely a documented internal
inconsistency in the source paper). LCO ToO promptly triggered; VLT spectroscopy (z=1.105) 2.3 d
later; Swift confirmed an X-ray counterpart. The "mixed r′/R proxy" noted in Table 18 comes from
externally published GCN photometry overplotted in the paper's Fig. 3, not from the P48 table
itself.

**AT2021buv** (Andreoni+21 Table 6, primary P48 source — full multiwavelength paper is Kumar et
al., in prep, not Ho+22) — GRB210204A. The pre-trigger night (2021-02-03) has **5 non-detections
within ~2 h**, the closest match among all fetched events to the nominal "6 visits/night" HC
design. Last non-detection 2021-02-04 05:14 UT (g, lim 18.3); **first detection 07:07 UT,
r=17.16±0.03 — a 1 h 53 min gap**, exactly matching the paper's stated explosion-time constraint
of 1.9 h. This is the **cleanest real "fresh, same-night" catch** in the whole sample:
independently discovered by the standard real-time fast-transient filter (Ho et al. 2020d),
confirmed via IPN (Kool et al. 2021; Hurley et al. 2021), z=0.876 (Xu et al. 2021).

**AT2021cwd** (Ho+22) — GRB210212B, public survey. Discovery MJD 59257.3697, g≈19.57; **only two
ZTF points the first night** (one g, one r, ~1.1 h apart) — the paper states explicitly this
event "had only one g-band and one r-band measurement in the alert stream." Confirming follow-up
came from the Liverpool Telescope ~0.6 d later (rapid 2.4 mag/day fading, red color) — LT, not a
second ZTF visit, is what actually triggered classification. GRB trigger 6.2 h before first ZTF
detection (IPN, 1286 deg², chance-coincidence probability 0.09). No redshift obtained.

**AT2021lfa** (Ho+22 photometry, Li+25 jet fit) — orphan (no GRB). Discovery MJD 59338.2324,
r=18.60±0.08; g=18.80±0.11 ~1.92 h later, same night — "detected in both r and g band the first
night." Next visit ~0.58 d later via LT (multiband), then LT/SEDM at Δt ≈ 1.7–6.6 d. Single
power-law post-jet-break decline, α=2.54±0.02. Jet fit genuinely degenerate between on-axis
low-Γ₀ (5–13) and off-axis high-Γ₀ (θ_obs=2θ_jet, Γ₀≈100) — matches Table 18 exactly.

**AT2021qbd** (Ho+22) — GRB210610B, partnership HC. Discovery MJD 59376.2325, g=18.49±0.10, 9.7 h
after the Swift/Fermi-GBM/Konus-Wind trigger. Rise rate >2.1 mag/day. **First night: 4 points
(g,r,r,g) over ~3.1 h** (0.32 mag fade = 2.5 mag/day); second night 4 points within ~1.5 h; third
night 2 points — the clearest genuine same-night multi-visit HC pattern besides AT2020kym.
**Table 18's citation of Li et al. 2025 for this event could not be verified — see Corrections
#3.**

**AT2023lcr** (Li+25 Table 15, this paper's headline new event) — no GRB. First ZTF detection
2023-06-18 06:36:27 UT, rising >1.2 mag/day in r. GOTO detected it ~5 h earlier (L=18.77±0.06);
not present in the immediately preceding GOTO epoch, bracketing onset to **1 h 38 min** — this is
the exact "GOTO nondetection-detection onset bracket" Table 18's notes column references. Jet
fit: on-axis top-hat, Γ₀≈166, θ_v fixed at 0, θ_c≈0.02 rad (1.15°, highly collimated).

**AT2023sva** (Srinivasaragavan+25 Table 1) — no GRB. Discovery 2023-09-17 09:38:31 UT via
routine alert-stream filtering; last non-detection 2023-09-15 (r>20.36) — a **~2-day gap**,
consistent with the ordinary public-survey cadence rather than a triggered follow-up. Rise
>1.3 mag/day inferred; post-peak decline ~3 mag/day. 28 total photometric epochs across
ZTF/SEDM/NOT/GIT. Power-law structured jet strongly preferred over top-hat (ΔlogZ ≈ 9); posterior
θ_v = 0.07±0.02 rad, θ_c = 0.06±0.02 rad — matches Table 18 exactly. Independent cross-check via
interstellar scintillation at 72 days post-explosion.

## Corrections (flagged — not applied to code/defaults here)

1. **High-cadence footprint is ≈3,000 deg², not 2,500 deg².** Andreoni+21 states this
   explicitly. `analysis/ztf_validation.py` (`omega_srv_deg2=2500.0` in both `MODES` and
   `MODES_OLD_ENCODING`) and the app's `ztf_hc` preset (`web/app.js`) both currently use 2,500 —
   a ~20% error in N_exp for HC mode if corrected. **Needs a decision**: whether to update the
   constant (would shift the HC-mode calibration numbers throughout `ztf_validation.py`,
   `simple_mode_check.py`, and the app preset).
2. **The "~2/yr" / "~2.5/yr" observed-rate targets are project-derived, not literature-quoted
   rates** (see "Calibration-target honesty" above). **Needs a decision**: keep as documented
   working assumptions, or attempt a more careful rate estimate from the papers' cumulative
   counts (e.g. Ho+22's "2 events in ~3–4 yr of public-survey operation" ≈ 0.5–0.7/yr, notably
   lower than the current 2.0 target).
3. **Table 18's "Li et al. 2025" citation for AT2021qbd appears to be an error** — two
   independent full-text searches of arXiv:2411.07973 found no mention of AT2021qbd,
   ZTF21abfmpwn, or GRB210610B; that paper's own abstract lists only AT2023lcr, AT2020blt,
   AT2021any, AT2021lfa as its subjects. **Needs a decision**: correct Table 18 in the source
   notes, or identify the actual intended citation.
4. Minor: Ho et al. 2022's author list has no "Goldstein" (an internal-notes attribution error);
   AT2020blt's real discovery paper is Ho et al. 2020c (arXiv:2006.10761), not the 2022 paper.
