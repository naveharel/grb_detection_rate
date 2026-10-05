# Approved plan: luminosity function versus rates and medians

Approved by the user on 2026-09-17. This records the requested study before its
implementation; preliminary numbers below are hypotheses to reproduce, not the
final search result.

## Question and preliminary comparison

Can one shared luminosity function (LF) give approximately correct rates and
detected distance distributions for both existing ZTF comparison modes, while
retaining an observationally plausible viewing-angle distribution?

| Survey | LF | Effective rate [year^-1] | Median distance [Gpc] | Median q |
|---|---|---:|---:|---:|
| Public | Off | 0.99-4.70 | 0.56-1.00 | 1.17-1.23 |
| Public | Default | 0.74-2.42 | 1.73-2.17 | 1.22-1.24 |
| High cadence | Off | 3.76-4.62 | 2.27-2.47 | 1.07-1.09 |
| High cadence | Default | 0.92-1.03 | 2.43-2.46 | 1.17-1.19 |

Ranges cover the two detection-window prescriptions; effective rates include
coverage 0.35. Here q = theta_obs/theta_j. The default LF improves public-survey
distances, but these preliminary values do not demonstrate a joint match.

## Criteria and fixed assumptions

- Require effective rates within a factor of three of 2/year (public) and
  2.5/year (high cadence), and median distances within 35% of 3.67 and 3.88 Gpc,
  respectively. Both survey modes must pass with the same LF and window
  prescription.
- These are diagnostic project benchmarks, not a validated observational fit.
  The distance samples mix selections, and the modeled detection requirements
  differ from the actual search. See [the observation note](lf_rate_medians_observations.md)
  and [Ho et al. (2022)](https://arxiv.org/html/2201.12366v2).
- Keep current physics, jet geometry, intrinsic rate, survey settings, and
  coverage fixed. Retain the existing comparison's identification cuts and
  window measured from the peak. Compare the i*t_cad and (i-1)*t_cad endpoints
  separately, without mixing endpoints between surveys.
- Use L = nu L_nu(1 day), in erg/s, with phi(L) proportional to L^alpha.
  Report median q, median physical viewing angle, and fractions q<1, q<1.5,
  and q>2. Angle plausibility is assessed separately, without inventing a hard
  population-median threshold from [Li et al.](https://arxiv.org/html/2411.07973v2).

## LF-only search

- Search alpha from -3.5 to 0 in steps of 0.25; log10(L_min/[erg/s]) from
  41 to 46, and log10(L_max/[erg/s]) from 42 to 47.5, both in steps of 0.5.
  Retain ordered lower and upper bounds.
- Screen full-integral rates first, then compute medians from the detected
  distributions for rate-compatible candidates. Never average component
  medians.
- Rank candidates by their largest normalized rate or distance discrepancy.
  Refine around the ten best candidates at half the original grid spacing.
- Classify each LF as passing both window endpoints, passing one, or failing.
  Explicitly report boundary solutions and near-misses.
- Repeat highlighted examples at coverage 0.2 and 0.5 as sensitivity checks.
  Do not fit coverage; it scales rates without changing normalized medians.

## Verification and deliverables

- Reproduce LF-off/default comparisons and rerun relevant LF tests. The
  approved plan recorded 38 passing LF tests; distinguish that prior result
  from the actual implementation run.
- Recompute highlighted configurations with refined angle and distance grids,
  then double resolution. Require rates and distance medians stable within
  1%, q medians within 0.01, and integrated distributions consistent within 1%.
- Save this plan, reproducible analysis code, numerical results, and a concise
  report under analysis/. Do not change the engine, app, defaults, or public API.
- State whether a shared LF passes the selected rate/distance tolerances,
  whether its angle distribution is observationally plausible, and what remains
  uncertain because of sample selection and the fixed 4.55-Gpc Euclidean wall.

## Implementation clarification

Refinement includes the top ten configurations by the shared, both-endpoint
score and the top ten for each individual endpoint. Deduplicate those centers
before evaluating their half-spacing neighborhoods. This retains the approved
top-ten shared search while also resolving good solutions specific to either
window prescription; it does not change the comparison tolerances or LF bounds.
