# q_min / D_min Filter Verification — Layer C (UI) and Layer D (thesis)

Companion to `tests/test_filters.py`. Layer B (Python tests) runs automatically.
Layers C and D are manual and recorded here.

The full plan is at
`C:/Users/naveh/.claude/plans/can-you-make-a-sharded-thacker.md`.

---

## Layer C — Interactive UI walkthrough

Run the app and execute each step in order. Tick the box and record any deviation
from the expected outcome in the "Result" column.

```powershell
.venv\Scripts\python plot_3d.py
```

Reference values for the default parameter set (theta_j=0.1):
- `q_dec` ≈ 1.0316
- `q_j` = 2.0
- `q_nr` ≈ 14.142
- `D_Euc` = 5.28 Gpc

| Step | Action | Expected | Result |
|---|---|---|---|
| C.1 | Reset all sliders, toggle full-integral on/off | Surface shape changes (full integral higher off-axis); q_med/D_med at A1 cells differ by < 5% |  |
| C.2 | q_min: 0 → 0.001 | Visually identical surface (Δlog10 ≈ 5e-7) |  |
| C.3 | q_min slider to q_nr ≈ 14.14 | Surface entirely NaN; R_opt = N/A; N-slice and t-slice empty |  |
| C.4 | q_min = 1.0 | A1 region contracts; R_opt drops; hover q_med ≥ 1 everywhere |  |
| C.5 | q_min = 2.0 (= q_j) | A2/A3 cells vanish where q_E ≤ 2; visible surface mostly A1 and high-q_max cells |  |
| C.6 | D_min = 2.64 Gpc (= D_Euc/2) | A1/A2/A3 cells drop **uniformly** by log10(7/8) ≈ −0.058 dex (hover several A1 cells) |  |
| C.7 | D_min = 5.28 Gpc (= D_Euc) | A1/A2/A3 vanish; A4/A5/A6 vanish where D_i ≤ D_Euc; surface mostly empty |  |
| C.8 | q_min = 1, D_min = 1 Gpc | Δlog10 ≈ Δ(q_min=1 alone) + Δ(D_min=1 alone) at same hover cell (filter separable) |  |
| C.9 | Toggle full-integral, repeat C.4 + C.6 | No new NaN holes vs. dominant mode; magnitudes shift slightly |  |
| C.10 | Hover an A3 cell (q_E ≈ 5), sweep q_min 0→3 | Hover q_med moves from 5/√2 ≈ 3.54 up to √((25+9)/2) ≈ 4.12 |  |
| C.11 | Apply q_min = 2; note (N_opt, t_cad_opt) before/after | Optimizer moves toward larger t_cad or smaller N_exp (filter kills flux-limited cells) |  |
| C.12 | Apply q_min = 1, D_min = 1; export CSV | Z_raw values at known cells match hover; NaN cells exported as null/empty |  |
| C.13 | Apply q_min = 1; switch ZTF preset → Rubin → manual | q_min slider stays at 1 (presets do not touch the filter) |  |

Save before/after screenshots for **C.3, C.5, C.7** — these are the highest-signal
visual diagnostics.

---

## Layer D — Thesis / source cross-check

Documents to consult, listed in [sources/](../sources):
- `Detection Rates of Gamma-Ray Burst Afterglows in Optical Surveys - An Analytic Approach.pdf` (the bachelor's thesis — Tier 1)
- `Tentative Paper structure.pdf` (Tier 3; treat with skepticism)
- `מצגת לפרויקט מתקדם - הראל נוה.pptx` (advanced project presentation)

### D.1 — Unfiltered rate match

| Element | Code reference | Thesis reference | Match? |
|---|---|---|---|
| Full integrand `R ∝ ∫ q · D_eff(q)³ dq` | `detection_rate.py:481` and `docs/implementation_reference.tex` Eq. (full_integral) | thesis Eq. 39 or 62 (per code docstring) |  |
| Regime A1: R = f_Ω · R_int | dominant formula at q_min=D_min=0 (verified by test B.1) | thesis Section on saturation regime |  |
| Regime A2/A3: R = ½ · f_Ω · θ_j² · q_E² · R_int | dominant formula | thesis flux-limited regime |  |
| Regime A4: R = ½ · f_Ω · θ_j² · q_nr² · (D_i/D_Euc)³ · R_int | dominant formula | thesis cadence-limited saturation |  |
| Medians q_med = q_max/√2, D_med = D_eff·2^(-1/3) | `compute_medians_analytic`, verified by test B.9 | thesis median derivation |  |

### D.2 — Filtered form documented

The new section "Detection-Rate Filters: q_min and D_min" in
`docs/implementation_reference.tex` (Sec. 6 in the rendered doc) derives the
filtered formulas symbolically from the joint PDF `p(q,D) ∝ q · D²`.

### D.3 — The measure `q dq` (most load-bearing assumption)

The rate integrand is `q · D_eff(q)³ dq`, NOT `(q−1) dq` or `sin(θ_obs) dθ_obs`.
This is consistent with the thesis IF the underlying solid-angle factor
`θ_j² q dq` corresponds to the projected solid angle of the jet axis on the
unit sphere of viewing directions. The identity `θ_j² q_nr² / 2 = 1`
(via `q_nr = √2/θ_j`) is the smoking gun: it makes `∫₀^{q_nr} θ_j² q dq = 1`
exactly, so the regime A1 rate `f_Ω · R_int` requires no extra beaming prefactor.
This identity is explicitly noted in `docs/implementation_reference.tex` Sec. 3.2
("Beaming identity").

**Action**: open the bachelor's thesis and locate the derivation step where the
solid-angle integral is rewritten in terms of `q`. Confirm the result equals
`θ_j² q dq`. If the thesis uses `q̃ dq̃` with `q̃ = q − 1`, the implementation
disagrees with the thesis at low q (near `q_dec`) and every regime's rate is
off by a small constant factor.

### D.4 — Legacy "off-axis ON" equivalence

CLAUDE.md states `q_min ≈ q_dec ≈ 1.03` corresponds to the old "off-axis only"
toggle. The legacy `R_off = R_total − R_on` code path has been removed (verified
by `grep R_on|R_off|off_axis`). The new q_min implementation does NOT subtract
an on-axis term — it directly excludes the q ∈ [0, q_min] shell from each
regime's integral. Setting q_min = q_dec yields:

- A7 (q_max = q_dec) contributes zero (filter zeros all weight).
- Other regimes have their lower q-bound raised to q_dec.

This is mathematically *not* identical to `R_total − R_on` (which integrated
the full domain then subtracted the on-axis cap), but it is physically the
same idea — exclude the on-axis core.

**Action**: pick a cell where the old "off-axis ON" toggle was known to give a
specific R value (from git history or screenshots). Run the current model with
`q_min = q_dec` at the same parameters; compare. Discrepancy < 1% is expected;
larger would warrant investigation.

### D.5 — Hebrew presentation

Open `sources/מצגת לפרויקט מתקדם - הראל נוה.pptx` and locate the slide showing
the detection volume in (q, D) coordinates. Confirm:

- The unfiltered region is the shell `[0, q_nr] × [0, D_eff(q)]`.
- The filter `q_min, D_min` carves out the smaller shell `[q_min, q_nr] × [D_min, D_eff(q)]`.

If no such slide exists, this verification suggests adding one for the thesis.

---

## Open questions for the user

The plan's five open questions, recorded here for ongoing tracking:

1. **`q dq` vs `(q−1) dq` measure** (D.3) — needs thesis re-read.
2. **PLS H discontinuity at q_j** (B.13 covers PLS G; PLS H is known to be
   discontinuous, see Sec. 3.5 of the .tex). Is this intentional?
3. **`q_min ≈ q_dec` ↔ legacy "off-axis ON" equivalence** (D.4) — explicit
   numerical check pending.
4. **Indicator bias at q_min** (B.12 quantifies as `O(1/N_q) ≈ 2e-3`).
   Acceptable, or worth increasing N_q?
5. **D<sub>min</sub> tooltip wording** — the current text reads "Sources closer
   than D_min are excluded from the detection rate." This is technically wrong;
   the filter excludes bursts whose *maximum detectable distance* falls below
   D_min, not bursts physically nearer than D_min. Recommend rewording to:
   "Excludes bursts that cannot be detected beyond D_min."
