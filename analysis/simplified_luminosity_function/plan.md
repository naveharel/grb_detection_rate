# Approved plan: Simplified luminosity function

Approved by the user on 2026-10-01. Save this plan before implementing the study.

## Summary

Produce a standalone report, **Simplified Luminosity Function**, matching the
notation and explicit derivations of the existing LF document. Include analytic
results, reproducible numerical tests, figures, and the shared-parameter ZTF
comparison. Leave the app, physics engine, existing LF manuscript, and current
local edits unchanged.

## Physical model and derivation

Use dRcal/dL = A/L_ref (L/L_ref)^alpha for 0 < L < infinity, with
L = L_nu(t_dec) and fixed L_ref = 10^32 erg s^-1 Hz^-1. The two adjustable
parameters are A (volumetric event rate per natural-log luminosity interval at
the reference) and alpha.

- Explain the infinite intrinsic total rate and finite detected rate for
  -5/2 < alpha < -1.
- Derive the full luminosity-distance-angle integral, integrate luminosity and
  distance analytically, and give explicit angular expressions and log limits.
- Establish normalization, limiting-flux, and Euclidean-radius scalings.
- Identify the Figure 15 threshold L_dec = 4 pi D_Euc^2 F_lim, and derive the
  higher thresholds for repeated detections.
- Test concentration near q = 1 with logarithmic distributions, medians, and
  enclosed fractions; distinguish a maximum from concentration.
- Derive detected L, D, and q distributions, including
  P(D<d) = (d/D_Euc)^(2 alpha+5) and
  D_med = D_Euc 2^[-1/(2 alpha+5)].
- Compare continuous detection, the thesis cadence approximation, and random
  phase selection retaining peak time. Use the thesis Newtonian decline for
  late-time sensitivity. Keep physical luminosities in the main derivation and
  explicitly document relevant source inconsistencies.

## Numerical study and ZTF comparison

Build an isolated analysis package importing the unchanged engine.

- Use thesis parameters for analytic examples and the previous study's current
  parameters for ZTF, explicitly converting deceleration spectral luminosity
  to one-day luminosity.
- Cover alpha = -2.4, -2, -1.75, -1.25, -1.05; convergence boundaries;
  jet-break transitions; continuous and repeated detections.
- Reproduce both previous ZTF modes, both peak-window prescriptions, joint
  rise/fade selection, and coverage 0.35.
- At each slope choose one shared A minimizing the largest logarithmic rate
  discrepancy across all four survey/window cases. Scan the convergent slope
  interval and refine candidate minima and acceptance boundaries to 0.001.
- Retain factor-three rate and 35% distance tolerances. Report endpoint-specific
  alternatives separately.
- Report rates, L/D/q medians, and q<1, q<1.5, q>2 fractions. Check the common
  normalized distance distribution of both surveys.
- Treat comparisons as diagnostic project benchmarks; retain the selection and
  viewing-angle limitations in Ho et al. (2022), arXiv:2201.12366v2, and
  Li et al., arXiv:2411.07973v2.

## Verification and acceptance

- Independent quadrature vs analytic expressions: relative tolerance 1e-7 for
  smooth controlled cases.
- Verify A, flux, distance and pivot invariance; CDF normalization and medians.
- Demonstrate boundary divergences and truncated-to-infinite convergence at
  fixed differential amplitude, with quantified omitted tails.
- Compare finite-cutoff references to unchanged engine: agreement within 1%
  after resolving numerical error.
- Double resolution for highlighted cases: rates/D medians stable within 1%,
  q medians within 0.01.
- Quantify corner-approximation and cadence-prescription errors.

## Deliverables

Create docs/simplified_luminosity_function.tex and compiled PDF. Put reproducible
scripts, tests, CSV/JSON results, and this plan under
analysis/simplified_luminosity_function/.

Report: summary, notation, complete derivation, test methods, numerical tables,
contribution and approximation-error figures, ZTF comparison, and clear
conclusions. Follow project figure standards, visually inspect every final PDF
page, and restore TeX/PDF-rendering tools if needed. No app or engine changes.
